from router import Router
from config import NAME2SOLVER
from summarizer import Summarizer


if __name__ == "__main__":
    solver_name = "tot".lower()
    solver = NAME2SOLVER[solver_name]
    print(solver.solve("Is 9.11 larger than 9.9?")[0])