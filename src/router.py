from typing import List, Tuple


from .solver import Solver
from .config import NAME2SOLVER

@dataclass
class Router:

    solver_pool: List[Solver]


    def __init__(self):
        pass

    def route(self):
        pass

    def ensemble(self, names: List[str]):
        for name in names:
            if name not in NAME2SOLVER.keys():
                raise ValueError(f"{name} is not a supported solver")
            self.solver_pool.append(NAME2SOLVER[name])

    