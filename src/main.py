from router import Router
from config import NAME2SOLVER
from summarizer import Summarizer



if __name__ == "__main__":
    router = Router()
    router.ensemble(['tot', 'mctsr'])
    router.solve("Is 9.11 larger than 9.9?")
    print(router.solutions)