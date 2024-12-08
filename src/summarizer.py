"""
    Imitate o1's summarizer
"""

from typing import List
from util import load_model

class Summarizer:
    """
        Final component to summarize/aggerate the solutions.
    """

    def __init__(self, llm_model_id="meta-llama/Llama-3.1-8B-Instruct", external_call=None, summary_field: str=None):
        self.llm_model = load_model(llm_model_id)
        self.external_call = external_call
        self.summary_field = summary_field

    def rank(self, solutions: List[str], order: int = 1) -> List[float]:
        """
            Rank the solutions from high to low (order = 1, default)
            Rank the solutions from low to high (order = -1, default)
        """
        pass

    def eval(self, solution: str) -> float:
        """
            Evaluate the quality of a solution
        """
        pass

    def summarize(self, solutions: List[str], chosen_num: int = 3) -> str:
        """
            Summarize the solutions (I prefer to summarize only first 3 solutions)
        """
        def summarize_text(tokenizer, model, text, max_input_length=512, max_output_length=150):
            """
            Summarize a given text using a preloaded model and tokenizer.

            Args:
                tokenizer: The tokenizer loaded from the summarization model.
                model: The summarization model.
                text (str): The text to be summarized.
                max_input_length (int): Maximum length of the input text (default: 512).
                max_output_length (int): Maximum length of the output summary (default: 150).

            Returns:
                str: The generated summary.
            """
            try:
                # Tokenize the input text
                inputs = tokenizer.encode(
                    text, 
                    return_tensors="pt", 
                    max_length=max_input_length, 
                    truncation=True
                )

                # Generate the summary
                summary_ids = model.generate(
                    inputs, 
                    max_length=max_output_length, 
                    min_length=30, 
                    length_penalty=2.0, 
                    num_beams=4, 
                    early_stopping=True
                )

                # Decode the summary tokens
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                return summary

            except Exception as e:
                print(f"Error during summarization: {e}")
                return None

        sumamry_prompt = f"""
        You are an expert that summarizes the solutions.
        Following the steps to do the summary: Critize, Rank, Summarize (which means you will learn from these, not only summarizing the answers)
        The solutions are: {", ".join(solutions[:chosen_num])}
        Please provide a summary of the solutions.
        """

        if self.external_call:
            answer = self.external_call(sumamry_prompt)
            # summary_field is a string, try to get the corresponding property
            if isinstance(self.summary_field, str):
                return getattr(answer, self.summary_field, None)
        else:
            # default using LLM model
            return summarize_text(self.llm_model[0], self.llm_model[1], sumamry_prompt)



# if you want to use self-consistency, just replace the summarize function