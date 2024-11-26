import dspy
from mcts_llm.mctsr import MCTSr


# use dspy to configure the language model
ollama = dspy.OllamaLocal(
        model="qwen2.5:7b-instruct",
        model_type="chat",
        temperature=1.0,
        max_tokens=1024,
        num_ctx=1024,
        timeout_s=600
    )
dspy.settings.configure(lm=ollama, experimental=True)

def create_mctsr():
    """
    Create and return a MCTSr model.
    https://arxiv.org/pdf/2406.07394
    """
    return MCTSr()

def create_tot() -> Solver:
    """
    Create and return a Tree of Thoughts (ToT) model.

    Returns:
        function: A ToT model function that can solve a problem given a question.
    """
    def tot_solver(question: str, max_depth: int = 3, num_children: int = 3, max_rollouts: int = 10) -> str:
        """
        Tree of Thoughts (ToT) solver function.

        Args:
            question (str): The question to solve.
            max_depth (int): Maximum depth of the tree.
            num_children (int): Number of child thoughts to generate per node.
            max_rollouts (int): Maximum number of rollouts.

        Returns:
            str: Final answer to the question.
        """
        rollouts = 0

        def query_model(prompt: str) -> str:
            """Queries the model with the given prompt."""
            return ollama.generate(prompt)

        def expand_node(thought: str, depth: int) -> List[str]:
            """Generate child thoughts."""
            nonlocal rollouts
            if depth >= max_depth or rollouts >= max_rollouts:
                return []
            rollouts += 1
            prompt = f"Based on the thought: '{thought}', generate {num_children} new thoughts to solve: {question}"
            response = query_model(prompt)
            return response.split('\n')[:num_children]

        def evaluate_node(thought: str) -> float:
            """Evaluate the quality of a thought."""
            prompt = f"Evaluate how likely this thought leads to the correct answer for the question: '{question}'\nThought: {thought}\nScore (0 to 1):"
            response = query_model(prompt)
            try:
                return float(response)
            except ValueError:
                return 0.0

        # Initial thought and evaluation
        root = query_model(f"Initial thought for solving: {question}")
        best_thought = root
        best_score = evaluate_node(root)

        # Expand and evaluate thoughts
        for depth in range(max_depth):
            children = expand_node(best_thought, depth)
            for child in children:
                score = evaluate_node(child)
                if score > best_score:
                    best_thought = child
                    best_score = score

        # Generate the final answer
        final_prompt = f"Based on the thought: '{best_thought}', provide a final answer to the question: {question}"
        return query_model(final_prompt)

    return tot_solver


NAME2SOLVER = {
    "mctsr": create_mctsr(),
    "tot": create_tot(),
}
