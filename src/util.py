from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_path):
    """
    Load a model and tokenizer from local storage.

    Args:
        model_path (str): The path to the local Hugging Face model directory.

    Returns:
        tokenizer: The tokenizer for the model.
        model: The model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # Load the tokenizer and model from the specified path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        print("Model and tokenizer successfully loaded from:", model_path)
        return tokenizer, model

    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None


    
    