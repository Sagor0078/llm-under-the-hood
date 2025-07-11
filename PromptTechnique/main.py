import textwrap

def simulate_llm_response(prompt: str) -> str:
    """
    A placeholder function to simulate an LLM's response.
    In a real application, this would involve calling an LLM API.
    """
    print(f"\n--- Simulating LLM Response for Prompt ---\n")
    print(f"Prompt:\n{textwrap.indent(prompt, '    ')}\n")
    
    if "summarize" in prompt.lower():
        return "Simulated Summary: The provided text discusses key aspects of LLMs, focusing on their architecture, training, and underlying concepts like attention and tokenization. It highlights how prompt engineering helps guide these models."
    elif "translate" in prompt.lower() and "cat" in prompt.lower():
        return "Simulated Translation: Bird -> OISEAU"
    elif "act as a historian" in prompt.lower():
        return "Simulated Historian's Response: Ah, a fascinating query! From the annals of history, one might observe that the rise of large language models marked a pivotal shift in human-computer interaction, akin to the invention of the printing press in its dissemination of knowledge."
    elif "think step-by-step" in prompt.lower():
        return "Simulated CoT Response: Step 1: Identify the core problem. Step 2: Break it down into smaller, manageable sub-problems. Step 3: Solve each sub-problem sequentially. Step 4: Combine solutions for the final answer."
    else:
        return "Simulated Generic Response: I have processed your request. This is a generic simulated output."

def demonstrate_prompt_engineering():
    """
    Demonstrates various prompt engineering techniques by constructing prompts
    and showing their simulated LLM responses.
    """
    print("--- Demonstrating Prompt Engineering Techniques ---")

    # Direct Instruction Prompt
    print("\n=== Direct Instruction ===")
    direct_instruction_prompt = textwrap.dedent("""
    Summarize the following article in 50 words or less:
    Large Language Models (LLMs) are a class of artificial intelligence models that have revolutionized natural language processing (NLP). They are built upon the Transformer architecture and trained on enormous amounts of text data, allowing them to understand, generate, and manipulate human language. Key components include multi-head self-attention, positional encoding, and feed-forward networks. LLMs undergo pre-training, fine-tuning, and often reinforcement learning from human feedback (RLHF) to align with human preferences.
    """)
    response_direct = simulate_llm_response(direct_instruction_prompt.strip())
    print(f"LLM Response:\n{textwrap.indent(response_direct, '    ')}\n")

    # Role-Playing Prompt
    print("\n=== Role-Playing ===")
    role_playing_prompt = textwrap.dedent("""
    Act as a seasoned historian specializing in the digital age. Explain the significance of Large Language Models to a general audience, using historical analogies.
    """)
    response_role = simulate_llm_response(role_playing_prompt.strip())
    print(f"LLM Response:\n{textwrap.indent(response_role, '    ')}\n")

    # Few-Shot Prompt (In-Context Learning)
    print("\n=== Few-Shot Prompt (In-Context Learning) ===")
    few_shot_prompt = textwrap.dedent("""
    Translate English to French:

    Cat -> Chat
    Dog -> Chien
    Bird -> ?
    """)
    response_few_shot = simulate_llm_response(few_shot_prompt.strip())
    print(f"LLM Response:\n{textwrap.indent(response_few_shot, '    ')}\n")

    # Chain-of-Thought (CoT) Prompt
    print("\n=== Chain-of-Thought (CoT) ===")
    cot_prompt = textwrap.dedent("""
    I have 3 apples, then I buy 2 more. My friend gives me 1 apple, but then I eat 4 apples. How many apples do I have left? Think step-by-step.
    """)
    response_cot = simulate_llm_response(cot_prompt.strip())
    print(f"LLM Response:\n{textwrap.indent(response_cot, '    ')}\n")

    print("\n--- Prompt Engineering Demonstration Complete ---")

if __name__ == "__main__":
    demonstrate_prompt_engineering()
