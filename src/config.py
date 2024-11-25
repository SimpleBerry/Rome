import dspy
from mcts_llm.mctsr import MCTSr


def create_mctsr():
    ollama = dspy.OllamaLocal(
        model="qwen2.5:7b-instruct",
        model_type="chat",
        temperature=1.0,
        max_tokens=1024,
        num_ctx=1024,
        timeout_s=600
    )
    dspy.settings.configure(lm=ollama, experimental=True)
    return MCTSr()

def create_tot():
    return TOT()


NAME2SOLVER = {
    "mctsr": create_mctsr(),
    "tot": create_tot(),
}
