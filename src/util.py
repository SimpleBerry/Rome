from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model(model_path):
    """
    Load a model and tokenizer from local storage.

    Args:
        model_path (str): The path to the local Hugging Face model directory.

    Returns:
        tokenizer: The tokenizer for the model.
        model: The model.
    """
    try:
        # Load the tokenizer and model from the specified path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        print("Model and tokenizer successfully loaded from:", model_path)
        return tokenizer, model

    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None


    
    