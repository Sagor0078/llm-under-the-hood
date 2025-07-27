import triton
import triton.language as tl
import torch


@triton.jit
def _attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr,
    o_ptr, L_ptr,
    sm_scale,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_Lb, stride_Lh, stride_Lm,
    CAUSAL: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Pointer offset for current batch and head
    q_ptr += pid_b * stride_qb + pid_h * stride_qh
    k_ptr += pid_b * stride_kb + pid_h * stride_kh
    v_ptr += pid_b * stride_vb + pid_h * stride_vh
    o_ptr += pid_b * stride_ob + pid_h * stride_oh
    L_ptr += pid_b * stride_Lb + pid_h * stride_Lh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        mask=offs_m[:, None] < N_CTX,
        other=0.0
    )

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_curr = start_n + offs_n
        k = tl.load(
            k_ptr + offs_n_curr[None, :] * stride_kn + offs_d[:, None] * stride_kk,
            mask=offs_n_curr[None, :] < N_CTX,
            other=0.0
        )
        v = tl.load(
            v_ptr + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk,
            mask=offs_n_curr[:, None] < N_CTX,
            other=0.0
        )

        qk = tl.dot(q, k)
        qk *= sm_scale

        if CAUSAL:
            mask = offs_m[:, None] >= offs_n_curr[None, :]
            qk = tl.where(mask, qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        l_i_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        m_i = m_i_new
        l_i = l_i_new

    acc = acc / l_i[:, None]

    tl.store(
        o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
        acc.to(o_ptr.dtype.element_ty),
        mask=offs_m[:, None] < N_CTX
    )

    tl.store(
        L_ptr + offs_m,
        m_i + tl.log2(l_i),
        mask=offs_m < N_CTX
    )


def flash_attention_forward(q, k, v, sm_scale, causal=False):
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q.dtype == k.dtype == v.dtype
    assert q.is_cuda and k.is_cuda and v.is_cuda

    B, H, N_CTX, D = q.shape
    BLOCK_M = 128
    BLOCK_N = 128

    o = torch.empty_like(q)
    L = torch.empty((B, H, N_CTX), dtype=torch.float32, device=q.device)

    grid = (B, H, triton.cdiv(N_CTX, BLOCK_M))

    _attn_fwd_kernel[grid](
        q, k, v, o, L,
        sm_scale,
        N_CTX=N_CTX,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qm=q.stride(2), stride_qk=q.stride(3),
        stride_kb=k.stride(0), stride_kh=k.stride(1), stride_kn=k.stride(2), stride_kk=k.stride(3),
        stride_vb=v.stride(0), stride_vh=v.stride(1), stride_vn=v.stride(2), stride_vk=v.stride(3),
        stride_ob=o.stride(0), stride_oh=o.stride(1), stride_om=o.stride(2), stride_ok=o.stride(3),
        stride_Lb=L.stride(0), stride_Lh=L.stride(1), stride_Lm=L.stride(2),
        CAUSAL=causal,
        num_warps=8,
        num_stages=3,
    )
    return o, L


if __name__ == "__main__":
    # Config
    B, H, N, D = 2, 4, 1024, 64
    causal = False

    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    sm_scale = 1.0 / (D ** 0.5)

    out, logsumexp = flash_attention_forward(q, k, v, sm_scale, causal=causal)

    print("Output shape:", out.shape)
    print("Logsumexp shape:", logsumexp.shape)

    try:
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
        print("Reference shape:", ref.shape)
        print("Max absolute difference:", (out - ref).abs().max().item())
    except Exception as e:
        print("PyTorch reference failed:", e)
