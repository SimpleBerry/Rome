"""
    Imitate o1's summarizer
"""

from typing import List

class Summarizer:
    """
        Final component to summarize/aggerate the solutions.
    """

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
        pass

    def train(self):
        """
            Train the summarizer
        """
        pass

    def save(self):
        """
            Save the summarizer
        """
        pass





    