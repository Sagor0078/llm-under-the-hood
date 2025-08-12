import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import asyncio
import pickle
import time
from dataclasses import dataclass

@dataclass
class KVCache:
    """Key‑Value cache for transformer attention"""
    keys: torch.Tensor
    values: torch.Tensor
    seq_len: int
    layer_idx: int

@dataclass
class PrefillResult:
    """Result from prefill phase"""
    kv_caches: List[KVCache]
    logits: torch.Tensor
    prompt_length: int
    model_config: Dict

class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None, layer_idx: int = 0) -> Tuple[torch.Tensor, KVCache]:
      if x.dim() == 2:
        x = x.unsqueeze(0)  # Add batch dimension if missing

      if x.dim() != 3:
        raise ValueError(f"Expected x to have 3 dimensions [B, T, D], but got {x.shape}")

      batch_size, seq_len, _ = x.shape

      q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
      k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
      v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)



      if kv_cache is not None:
          k = torch.cat([kv_cache.keys, k], dim=2)
          v = torch.cat([kv_cache.values, v], dim=2)
            # Optional: truncate cache
          max_cache_len = 1024
          k = k[..., -max_cache_len:, :]
          v = v[..., -max_cache_len:, :]

      scores = torch.matmul(q, k.transpose(-2,-1)) / (self.head_dim ** 0.5)
      attn_weights = F.softmax(scores, dim=-1)
      out = torch.matmul(attn_weights, v)
      out = out.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
      out = self.out_proj(out)

      new_cache = KVCache(keys=k, values=v, seq_len=k.size(2), layer_idx=layer_idx)
      return out, new_cache

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([AttentionLayer(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, kv_caches: Optional[List[KVCache]] = None) -> Tuple[torch.Tensor, List[KVCache]]:
        x = self.embed(input_ids)
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = layer(x, cache, layer_idx=i)
            new_caches.append(new_cache)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_caches

class PrefillNode:
    def __init__(self, model: SimpleLLM, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    async def prefill(self, input_ids: torch.Tensor) -> PrefillResult:
        start_time = time.time()
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            logits, kv_caches = self.model(input_ids)
            last_logits = logits[:, -1, :]
        print(f"Prefill completed in {time.time() - start_time:.3f}s")
        return PrefillResult(kv_caches=kv_caches, logits=last_logits,
                             prompt_length=input_ids.size(1),
                             model_config={"vocab_size":self.model.vocab_size,
                                           "d_model":self.model.d_model,
                                           "n_heads":self.model.n_heads,
                                           "n_layers":self.model.n_layers})

class DecodeNode:
    def __init__(self, model: SimpleLLM, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    async def decode_step(self, token_id: torch.Tensor, kv_caches: List[KVCache]) -> Tuple[torch.Tensor, List[KVCache]]:
        with torch.no_grad():
            token_id = token_id.to(self.device)
            if token_id.dim() == 0:
                token_id = token_id.unsqueeze(0)
            token_id = token_id.unsqueeze(0)  # shape (1,1)
            device_caches = [
                KVCache(keys=c.keys.to(self.device),
                        values=c.values.to(self.device),
                        seq_len=c.seq_len,
                        layer_idx=c.layer_idx)
                for c in kv_caches
            ]
            logits, new_caches = self.model(token_id, device_caches)
        return logits.squeeze(0).squeeze(0), new_caches

class KVCacheTransfer:
    @staticmethod
    def serialize_cache(cache: KVCache) -> bytes:
        return pickle.dumps({'keys': cache.keys.cpu(), 'values': cache.values.cpu(),
                             'seq_len': cache.seq_len, 'layer_idx': cache.layer_idx})
    @staticmethod
    def deserialize_cache(data: bytes) -> KVCache:
        cd = pickle.loads(data)
        return KVCache(keys=cd['keys'], values=cd['values'], seq_len=cd['seq_len'], layer_idx=cd['layer_idx'])
    @staticmethod
    async def transfer_caches(caches: List[KVCache]) -> List[bytes]:
        serialized = [KVCacheTransfer.serialize_cache(c) for c in caches]
        await asyncio.sleep(0.01)
        return serialized

class DisaggregatedInferenceEngine:
    def __init__(self, prefill_node: PrefillNode, decode_node: DecodeNode, eos_token_id: int = 0):
        self.prefill_node = prefill_node
        self.decode_node = decode_node
        self.eos_token_id = eos_token_id

    async def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0) -> List[int]:
        prefill = await self.prefill_node.prefill(input_ids)
        serialized = await KVCacheTransfer.transfer_caches(prefill.kv_caches)
        caches = [KVCacheTransfer.deserialize_cache(d) for d in serialized]
        generated = []
        logits = prefill.logits / temperature
        token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        for _ in range(max_new_tokens):
            generated.append(token.item())
            logits, caches = await self.decode_node.decode_step(token, caches)
            logits = logits / temperature
            token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            if token.item() == self.eos_token_id:
                break
        return generated

class MonolithicInferenceEngine:
    def __init__(self, model: SimpleLLM, device: str = "cpu", eos_token_id: int = 0):
        self.model = model.to(device)
        self.device = device
        self.eos_token_id = eos_token_id
        self.model.eval()

    async def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0) -> List[int]:
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            generated = []
            for _ in range(max_new_tokens):
                logits, _ = self.model(input_ids)
                logits = logits[:, -1, :] / temperature
                token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated.append(token.item())
                input_ids = torch.cat([input_ids, token.unsqueeze(0).unsqueeze(0)], dim=1)
                if token.item() == self.eos_token_id:
                    break
        return generated

async def demo_disaggregated_inference():
    vocab_size, d_model, n_heads, n_layers = 1000, 512, 8, 6
    pmodel = SimpleLLM(vocab_size, d_model, n_heads, n_layers)
    dmodel = SimpleLLM(vocab_size, d_model, n_heads, n_layers)
    dmodel.load_state_dict(pmodel.state_dict())
    prefill_node = PrefillNode(pmodel, device="cpu")
    decode_node = DecodeNode(dmodel, device="cpu")
    engine = DisaggregatedInferenceEngine(prefill_node, decode_node, eos_token_id=0)
    input_ids = torch.randint(1, vocab_size, (1,20))
    print("Running disaggregated inference...")
    start = time.time()
    gen = await engine.generate(input_ids, max_new_tokens=10)
    print(f"Generated tokens: {gen} in {time.time()-start:.3f}s")
    return gen

async def benchmark_comparison():
    vocab_size, d_model, n_heads, n_layers = 1000, 512, 8, 6
    m1 = SimpleLLM(vocab_size, d_model, n_heads, n_layers)
    m2 = SimpleLLM(vocab_size, d_model, n_heads, n_layers)
    m3 = SimpleLLM(vocab_size, d_model, n_heads, n_layers)
    m2.load_state_dict(m1.state_dict())
    m3.load_state_dict(m1.state_dict())
    disagg = DisaggregatedInferenceEngine(PrefillNode(m1), DecodeNode(m2), eos_token_id=0)
    mono = MonolithicInferenceEngine(m3, eos_token_id=0)
    input_ids = torch.randint(1, vocab_size, (1,20))
    print("=== Performance Comparison ===")
    start = time.time()
    mono_tokens = await mono.generate(input_ids, max_new_tokens=10)
    t_mono = time.time()-start
    print(f"Monolithic: tokens={mono_tokens} time={t_mono:.3f}s")
    start = time.time()
    disagg_tokens = await disagg.generate(input_ids, max_new_tokens=10)
    t_disagg = time.time()-start
    print(f"Disaggregated: tokens={disagg_tokens} time={t_disagg:.3f}s")
    print(f"Speedup: {t_mono/t_disagg:.2f}x, same tokens: {mono_tokens == disagg_tokens}")

# If you're in a notebook, run:
# await demo_disaggregated_inference()
# await benchmark_comparison()

# If in a standalone script:
# import nest_asyncio; nest_asyncio.apply()
# asyncio.run(demo_disaggregated_inference())
# asyncio.run(benchmark_comparison())


# Run the demo
if __name__ == "__main__":
    print("Prefill-Decode Disaggregation Demo")
    print("=" * 40)
    
    # Run basic demo
    await demo_disaggregated_inference()
    
    print("\n" + "=" * 40)
    
    # Run benchmark comparison
    await benchmark_comparison()