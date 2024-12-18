from router import Router
from config import NAME2SOLVER
from summarizer import Summarizer


if __name__ == "__main__":
    router = Router()
    router.ensemble(['tot', 'mctsr'])
    problem = f"trans-cinnamaldehyde was treated with methylmagnesium bromide, forming product 1. 1 was treated with pyridinium chlorochromate, forming product 2. 3 was treated with (dimethyl(oxo)-l6-sulfaneylidene)methane in DMSO at elevated temperature, forming product 3. how many carbon atoms are there in product 3?"
    router.solve(problem)
    summarizer = Summarizer()
    final_summary = summarizer.summarize(router.solutions[problem].values())
    print(final_summary)