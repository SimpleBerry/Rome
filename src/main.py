from router import Router
from config import NAME2SOLVER
from summarizer import Summarizer


if __name__ == "__main__":
    solver_name = "mctsr"
    solver = NAME2SOLVER[solver_name]
    print(solver.solve("how many r in strawberry?"))