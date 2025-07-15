import time
import random
from collections import deque
import numpy as np 

# Simulate a simplified LLM for token generation 
class SimpleLLM:
    def __init__(self, processing_time_per_token=0.01):
        """
        Simulates an LLM's token generation time.
        processing_time_per_token: time taken to process one token (e.01 seconds).
        """
        self.processing_time_per_token = processing_time_per_token

    def generate_token(self):
        """Simulates generating a single token."""
        time.sleep(self.processing_time_per_token)
        return random.randint(0, 99) # Dummy token ID

    def process_sequence(self, sequence_length):
        """Simulates processing an entire sequence (e.g., prompt prefill)."""
        # Assume prefill is faster per token than decode, or a fixed overhead
        time.sleep(sequence_length * self.processing_time_per_token * 0.5)
        # Return dummy output as this sim focuses on timing, not content
        return [random.randint(0, 99) for _ in range(sequence_length)]

# Request Class 
class InferenceRequest:
    def __init__(self, request_id, prompt_length, max_new_tokens):
        self.request_id = request_id
        self.prompt_length = prompt_length
        self.max_new_tokens = max_new_tokens
        self.generated_tokens = []
        self.start_time = time.time()
        self.end_time = None  # Add end_time attribute
        self.status = "PENDING"

    def __repr__(self):
        return (f"Request(ID={self.request_id}, PromptLen={self.prompt_length}, "
                f"GenCount={len(self.generated_tokens)}/{self.max_new_tokens}, Status={self.status})")

# Static Batching Simulator 
def run_static_batching(llm, requests, batch_size, total_max_tokens_per_request):
    """
    Simulates static batching.
    All sequences in a batch are padded to the max length of the longest sequence in that batch.
    """
    print("\n--- Running Static Batching Simulation ---")
    request_queue = deque(requests)
    completed_requests = []

    total_time_taken = 0
    total_wasted_tokens = 0

    while request_queue:
        current_batch = []
        # Form a batch of fixed size
        for _ in range(batch_size):
            if request_queue:
                current_batch.append(request_queue.popleft())
            else:
                break

        if not current_batch:
            break

        # Determine max sequence length in the current batch (prompt + max_new_tokens)
        # This simulates padding for the entire potential generation
        max_seq_len_in_batch = 0
        for req in current_batch:
            max_seq_len_in_batch = max(max_seq_len_in_batch, req.prompt_length + req.max_new_tokens)

        print(f"\nProcessing Static Batch (size: {len(current_batch)}, max_seq_len: {max_seq_len_in_batch})")

        batch_start_time = time.time()

        # Simulate processing the entire padded batch
        # This is a simplification: in reality, it's token-by-token generation
        # but with padding applied to all steps.

        # Simulate prefill for all prompts in batch
        for req in current_batch:
            llm.process_sequence(req.prompt_length) # Simulate prompt processing

        # Simulate decode for all tokens up to max_seq_len
        for token_idx in range(max_seq_len_in_batch):
            for req in current_batch:
                if len(req.generated_tokens) < req.max_new_tokens:
                    llm.generate_token() # Simulate generating one token
                else:
                    # This is where padding waste occurs:
                    # The model still computes for this "padded" position
                    # even if the request is complete.
                    time.sleep(llm.processing_time_per_token) # Simulate processing a padding token
                    total_wasted_tokens += 1

        batch_end_time = time.time()
        total_time_taken += (batch_end_time - batch_start_time)

        for req in current_batch:
            req.status = "COMPLETED"
            req.end_time = time.time() # Set end_time when completed
            req.generated_tokens = [random.randint(0,99) for _ in range(req.max_new_tokens)] # Dummy generated tokens
            completed_requests.append(req)
            print(f"  {req.request_id} completed in {req.end_time - req.start_time:.4f}s")

    print(f"\n--- Static Batching Summary ---")
    print(f"Total requests completed: {len(completed_requests)}")
    print(f"Total simulation time: {total_time_taken:.4f}s")
    print(f"Simulated wasted tokens (due to padding): {total_wasted_tokens}")
    # Calculate average latency only if there are completed requests
    average_latency = np.mean([r.end_time - r.start_time for r in completed_requests]) if completed_requests else np.nan
    print(f"Average latency per request: {average_latency:.4f}s" if not np.isnan(average_latency) else "N/A")


# Dynamic/Continuous Batching Simulator 
def run_dynamic_batching(llm, requests, total_max_tokens_per_request):
    """
    Simulates dynamic/continuous batching.
    Requests are processed as they arrive, and GPU time is shared efficiently.
    """
    print("\n--- Running Dynamic/Continuous Batching Simulation ---")

    active_requests = deque()
    pending_requests = deque(requests)
    completed_requests = []

    total_time_taken = 0

    # Simulate a single "GPU cycle" where one token is generated for each active request
    # This is a very simplified event loop.

    # Simulate prefill for prompts first, then interleave decode
    # In a real system, prefill would be batched too.
    # For this simulation, we prefilled all at start for simplicity.
    for req in list(pending_requests): # Iterate over a copy to allow modification
        print(f"Prefilling prompt for {req.request_id} (length {req.prompt_length})")
        llm.process_sequence(req.prompt_length) # Simulate prompt prefill
        req.prefill_end_time = time.time() # Keep track of prefill end time if needed
        active_requests.append(req)
        pending_requests.remove(req) # Move from pending to active

    print("\nStarting interleaved token generation...")

    while active_requests: # Continue as long as there are active requests
        current_cycle_requests = list(active_requests) # Snapshot for current cycle

        # Simulate one token generation step for each active request
        # This is the "continuous" part: GPU processes one token for each active request
        cycle_start_time = time.time()

        requests_to_remove = [] # List to store requests to remove after the cycle

        for req in current_cycle_requests:
            if len(req.generated_tokens) < req.max_new_tokens:
                llm.generate_token() # Simulate generating one token
                req.generated_tokens.append(random.randint(0,99)) # Add dummy token

                # Check for completion (max tokens reached or EOS token)
                if len(req.generated_tokens) == req.max_new_tokens or req.generated_tokens[-1] == 0: # 0 as EOS
                    req.status = "COMPLETED"
                    req.end_time = time.time() # Set end_time when completed
                    completed_requests.append(req)
                    requests_to_remove.append(req) # Mark for removal
                    print(f"  {req.request_id} completed in {req.end_time - req.start_time:.4f}s")
            else:
                 # Should have been removed if max_new_tokens reached in a previous cycle
                 requests_to_remove.append(req)


        # Remove completed requests from the active queue
        for req in requests_to_remove:
             if req in active_requests:
                 active_requests.remove(req)


        cycle_end_time = time.time()
        total_time_taken += (cycle_end_time - cycle_start_time)

        # In a real system, new requests might arrive and be added to active_requests here.
        # For this simulation, all requests are known upfront and moved to active after prefill.


    print(f"\n--- Dynamic/Continuous Batching Summary ---")
    print(f"Total requests completed: {len(completed_requests)}")
    print(f"Total simulation time: {total_time_taken:.4f}s")
    # In dynamic batching, there's no explicit "wasted tokens" due to padding
    # as computation is only performed for active tokens.
    # Calculate average latency only if there are completed requests
    average_latency = np.mean([r.end_time - r.start_time for r in completed_requests]) if completed_requests else np.nan
    print(f"Average latency per request: {average_latency:.4f}s" if not np.isnan(average_latency) else "N/A")


# Main Simulation Execution 
if __name__ == "__main__":
    llm_simulator = SimpleLLM(processing_time_per_token=0.005) # Faster processing for quicker demo

    # Define a set of requests with varying prompt and generation lengths
    requests_static = [
        InferenceRequest("Req_S_1", prompt_length=10, max_new_tokens=50),
        InferenceRequest("Req_S_2", prompt_length=5, max_new_tokens=100),
        InferenceRequest("Req_S_3", prompt_length=20, max_new_tokens=30),
        InferenceRequest("Req_S_4", prompt_length=15, max_new_tokens=80),
        InferenceRequest("Req_S_5", prompt_length=8, max_new_tokens=60),
    ]

    requests_dynamic = [
        InferenceRequest("Req_D_1", prompt_length=10, max_new_tokens=50),
        InferenceRequest("Req_D_2", prompt_length=5, max_new_tokens=100),
        InferenceRequest("Req_D_3", prompt_length=20, max_new_tokens=30),
        InferenceRequest("Req_D_4", prompt_length=15, max_new_tokens=80),
        InferenceRequest("Req_D_5", prompt_length=8, max_new_tokens=60),
    ]

    STATIC_BATCH_SIZE = 2 # batch size
    TOTAL_MAX_TOKENS_PER_REQUEST = 150 # Upper bound for padding calculation in static batching

    # Run Static Batching
    # We need to deep copy requests_static because they will be modified
    import copy
    run_static_batching(llm_simulator, copy.deepcopy(requests_static), STATIC_BATCH_SIZE, TOTAL_MAX_TOKENS_PER_REQUEST)

    print("\n" + "="*50 + "\n")

    # Run Dynamic/Continuous Batching
    run_dynamic_batching(llm_simulator, copy.deepcopy(requests_dynamic), TOTAL_MAX_TOKENS_PER_REQUEST)