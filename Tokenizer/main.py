from transformers import AutoTokenizer

def demonstrate_tokenization(model_name: str, text: str):
    """
    Demonstrates subword tokenization (e.g., BPE) using a pre-trained tokenizer
    from the Hugging Face transformers library.

    Args:
        model_name (str): The name of the pre-trained model whose tokenizer to use
                          (e.g., "gpt2", "bert-base-uncased", "meta-llama/Llama-2-7b-hf").
                          Note: Llama-2 tokenizers might require specific setup/authentication.
                          "gpt2" is a good general-purpose choice for demonstration.
        text (str): The input text to be tokenized.
    """
    print(f"--- Demonstrating Tokenization with {model_name} Tokenizer ---")
    print(f"Input Text: '{text}'\n")

    try:
        # Load a pre-trained tokenizer
        # The 'use_fast=True' argument uses a Rust-based tokenizer for speed.
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print(f"Tokenizer loaded: {tokenizer.__class__.__name__}\n")

        # Encode the text: Convert text to token IDs
        # This is the primary step where text is broken into subwords and mapped to IDs.
        encoded_input = tokenizer.encode(text, add_special_tokens=False)
        print(f"Encoded Token IDs (numerical representation): {encoded_input}")

        # Decode the token IDs back to text
        # This reconstructs the text from the numerical IDs.
        decoded_text = tokenizer.decode(encoded_input)
        print(f"Decoded Text (reconstructed from IDs): '{decoded_text}'\n")

        # Get the actual tokens (subwords)
        # This shows the intermediate subword units.
        tokens = tokenizer.convert_ids_to_tokens(encoded_input)
        print(f"Individual Tokens (subword units): {tokens}\n")

        # Demonstrate handling of Out-Of-Vocabulary (OOV) words
        # A word like 'supercalifragilisticexpialidocious' might be split into subwords.
        # Note: If the word is very common in the tokenizer's training data, it might
        # still be a single token.
        oov_text = "This is an unsupercalifragilisticexpialidocious word."
        print(f"--- Demonstrating OOV Word Handling ---")
        print(f"OOV Input Text: '{oov_text}'")
        encoded_oov = tokenizer.encode(oov_text, add_special_tokens=False)
        tokens_oov = tokenizer.convert_ids_to_tokens(encoded_oov)
        print(f"OOV Tokens: {tokens_oov}\n")
        print(f"Decoded OOV Text: '{tokenizer.decode(encoded_oov)}'\n")

        # Demonstrate special tokens (if added)
        # LLMs often add special tokens like [CLS], [SEP] for specific tasks.
        print(f"--- Demonstrating Special Tokens ---")
        text_with_special = "Hello world!"
        # add_special_tokens=True will add CLS/SEP/BOS/EOS depending on model
        encoded_with_special = tokenizer.encode(text_with_special, add_special_tokens=True)
        tokens_with_special = tokenizer.convert_ids_to_tokens(encoded_with_special)
        print(f"Text with Special Tokens: '{text_with_special}'")
        print(f"Encoded with Special Tokens: {encoded_with_special}")
        print(f"Tokens with Special Tokens: {tokens_with_special}\n")


    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have the 'transformers' library installed (`pip install transformers`).")
        print("For Llama-2, you might need to log in to Hugging Face and accept its terms.")
        print("Try a more common model like 'gpt2' if you encounter issues.")


if __name__ == "__main__":
    # You can try different model names here
    # "gpt2" is a good default for general purpose BPE-like tokenization
    # "bert-base-uncased" uses WordPiece tokenization
    # "meta-llama/Llama-2-7b-hf" (requires Hugging Face login and access)

    # Note: For Llama-2 models, you might need to install 'accelerate' and 'sentencepiece'
    # and potentially log in to Hugging Face via `huggingface-cli login`
    # or pass a token to from_pretrained(token="hf_YOUR_TOKEN").
    
    demonstrate_tokenization(model_name="gpt2", text="Tokenization is a fundamental concept in Large Language Models.")
    print("\n" + "="*80 + "\n")
    demonstrate_tokenization(model_name="bert-base-uncased", text="Tokenization is a fundamental concept in Large Language Models.")
