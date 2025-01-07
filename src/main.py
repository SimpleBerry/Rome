import argparse
from router import Router
from summarizer import Summarizer

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Chemistry problem solver')
    parser.add_argument('--problem', type=str, default='What is the result of 9 * 199?',
                      help='The chemistry problem to solve')
    parser.add_argument('--methods', nargs='+', default=['tot', 'mctsr'],
                      help='Ensemble methods to use (default: tot mctsr)')

    # Parse arguments
    args = parser.parse_args()

    # Initialize and run the router
    router = Router()
    router.ensemble(args.methods)
    router.solve(args.problem)

    # Summarize the solutions
    summarizer = Summarizer()
    final_summary = summarizer.summarize(router.solutions[args.problem].values())
    print(final_summary)

if __name__ == "__main__":
    main()