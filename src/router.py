from typing import List, Tuple, Dict
from dataclasses import dataclass
from config import NAME2SOLVER, Solver

@dataclass
class Router:
    def __init__(self, solver_pool: List[Tuple[str, Solver]] = [], solutions: Dict[str, Dict[str, str]] = {}, grades: Dict[str, Dict[str, float]] = {}, step_solutions: Dict[str, Dict[str, List[str]]] = {}):
        self.solver_pool = solver_pool  # solver_pool: List[Tuple[str, Solver]]
        self.solutions = solutions      # solutions: { problem : {solver name: solution} }
        self.step_solutions = step_solutions    # step_solutions: Dict[str, Dict[str, List[str]]]
        self.grades = grades        # grades: { problem : {solver name: grade} }

    def route(self):
        pass

    def solve(self, problem: str) -> None:
        for name, solver in self.solver_pool:
            solution = solver.solve(problem)
            self.solutions.setdefault(problem, {})[name] = solution[0] if isinstance(solution, list) else solution

    def step_solve(self, problem: str, max_steps: int = 3) -> None:
        for name, solver in self.solver_pool:
            self.step_solutions.setdefault(problem, {})[name] = solver.step_solve(problem, max_steps)[0]
            
    def ensemble(self, names: List[str]) -> None:
        for name in names:
            name = name.lower()
            if name not in NAME2SOLVER.keys():
                raise ValueError(f"{name} is not a supported solver")
            self.solver_pool.append( (name, NAME2SOLVER[name]) )

    