from abc import ABC, abstractmethod
from typing import List

class Solver(ABC):

    @abstractmethod
    def solve(self, problem: str) -> str:
        """
            Returns a complete solution
        """
        pass

    @abstractmethod
    def step_solve(self, problem: str) -> List[str]:
        """
            Returns a list of partial solution
        """
        pass