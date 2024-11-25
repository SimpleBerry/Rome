from typing import List, Tuple


from .solver import Solver
from .config import NAME2SOLVER

@dataclass
class Router:

    solver_pool: List[Solver]

    # { problem : {solver name: solution} }
    solutions: Dict[str, Dict[str, Solver]]

    # { problem : {solver name: grade} }
    grades: Dict[str, Dict[str, float]]

    def __init__(self):
        self.solver_pool = []
        self.solutions = {}
        self.grades = {}

    def route(self):
        pass

    def solve(self, problem: str) -> None:
        for solver in self.solver_pool:
            # compatible with mcts-llm (https://github.com/NumberChiffre/mcts-llm)
            self.solutions[problem][solver[0]] = solver[1].solve(problem).answer if solver[0] == "mctsr" else solver[1].solve(problem)

    def ensemble(self, names: List[str]) -> None:
        for name in names:
            if name not in NAME2SOLVER.keys():
                raise ValueError(f"{name} is not a supported solver")
            self.solver_pool.append( (name, NAME2SOLVER[name]) )

    