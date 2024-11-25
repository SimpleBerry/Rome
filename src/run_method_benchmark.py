from crypt import methods
import re
import random
from typing import List, Tuple, Dict, Optional, Any
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import time
import math
from concurrent.futures import ThreadPoolExecutor
import socket
import pandas as pd
from collections import Counter
from grading import check,check_label,extract_label

# Model configurations
MODELS = {
    "llama3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # "gpt4o_mini": "gpt4o-mini"  # Replace with the actual model path or identifier
}

# Dataset configurations
DATASETS = ["gsm8k", "math500", "aime2024"]

ROLLOUTS = 8

clients = {model: [] for model in MODELS}
times = {model: time.time() for model in MODELS}

def create_client(line, model):
    global clients
    if len(line) < 3:
        return
    node, port, _ = line.split(',')
    ip = socket.gethostbyname(node)
    print(f"Creating client for {model} at {ip}:{port}")
    client = OpenAI(
        base_url=f"http://{ip}:{port}/v1",
        api_key="token-abc123",
    )
    try:
        client.chat.completions.create(
            model=MODELS[model],
            messages=[{"role": "user", "content": 'hi'}],
            temperature=0.95,
            timeout=15
        )
        print(f"Client {len(clients[model])+1} for {model} created")
        clients[model].append(client)
    except Exception as e:
        print(f"Error creating client for {model}: {e}")

def get_clients():
    global clients
    lines = open('./server.csv', 'r').readlines()
    for model in MODELS:
        with ThreadPoolExecutor() as executor:
            executor.map(lambda line: create_client(line, model), lines)

def get_client(model):
    global clients, times
    return random.choice(clients[model])

def load_datasets():
    """Load GSM8K and Math datasets."""
    gsm8k = load_dataset("gsm8k", "main", split="test")
    math500 = load_dataset("qq8933/MATH500", split="test")
    math500 = math500.map(lambda example: {"question": example["problem"], "answer": example["solution"]})
    aime2024 = load_dataset("AI-MO/aimo-validation-aime", split="train")
    aime2024 = aime2024.filter(lambda example: '2024' in example['url'])
    aime2024 = aime2024.map(lambda example: {"question": example["problem"], "answer": example["answer"]})
    return {"math500": math500,"gsm8k": gsm8k, 'aime2024': aime2024}


def query_model(model: str, prompt: str) -> str:
    """Query the language model using the OpenAI-like interface."""
    try:
        client = get_client(model)
        completion = client.chat.completions.create(
            model=MODELS[model],
            messages=[
                {"role": "system", "content": "You are a helpful assistant that solves math problems step by step. Always end your answer with 'Therefore, the final answer is: [number].'"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.15,
            max_tokens=1000,
            stop=["Question:", "Human:"]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error querying model {model}: {str(e)}"

# def extract_final_answer(response: str) -> str:
#     """Extract the final answer from the model's response."""
#     match = re.search(r'final answer is:\s*(-?\d+\.?\d*)', response, re.IGNORECASE)
#     if match:
#         return match.group(1)
#     # Fallback: look for the last number in the response
#     numbers = re.findall(r'-?\d+\.?\d*', response)
#     return numbers[-1] if numbers else response

# def evaluate_answer(answer: str, correct_answer: str) -> bool:
#     """Evaluate if the answer is correct, allowing for minor variations."""
#     def normalize(s):
#         # Extract the last number from the correct answer string
#         numbers = re.findall(r'-?\d+\.?\d*', s)
#         return numbers[-1] if numbers else s

#     normalized_answer = normalize(answer)
#     normalized_correct = normalize(correct_answer)
    
#     try:
#         return abs(float(normalized_answer) - float(normalized_correct)) < 1e-6
#     except ValueError:
#         return normalized_answer == normalized_correct

def extract_final_answer(response: str) -> str:
    """Extract the final answer from the model's response."""
    return extract_label(response,'MATH')

# def evaluate_answer(answer: str, correct_answer: str) -> bool:
#     """Evaluate if the answer is correct, allowing for minor variations."""
#     # def normalize(s):
#     #     # Extract the last number from the correct answer string
#     #     numbers = re.findall(r'-?\d+\.?\d*', s)
#     #     return numbers[-1] if numbers else s
#     normalize = extract_final_answer

#     normalized_answer = normalize(answer)
#     normalized_correct = normalize(correct_answer)
    
#     try:
#         return abs(float(normalized_answer) - float(normalized_correct)) < 1e-6
#     except Exception as e:
#         return normalized_answer == normalized_correct
    
#     return False

def evaluate_answer(answer: str, correct_answer: str) -> bool:
    """Evaluate if the answer is correct, allowing for minor variations."""
    return check(correct_answer, answer,'MATH')

def repeatead_sampling(model: str, question: str, num_samples: int = ROLLOUTS) -> str:
    """Implement repeated sampling technique."""
    all_labels = []
    label_to_answers = {}
    for _ in range(num_samples):
        prompt = f"Question: {question}\nProvide a step-by-step solution and final answer:"
        response = query_model(model, prompt)
        label = extract_final_answer(response)
        all_labels.append(label)
        if label in label_to_answers:
            label_to_answers[label].append(response)
        else:
            label_to_answers[label] = [response]
    
    counter = Counter(all_labels)
    most_common_label, _ = counter.most_common(1)[0]
    print(label_to_answers[most_common_label][0])
    return label_to_answers[most_common_label][0]

def self_refine(model: str, question: str, max_iterations: int = 2) -> str:
    # Initial solution
    prompt = f"Question: {question}\nProvide a detailed answer to this question."
    response = query_model(model, prompt)
    
    for _ in range(max_iterations):
        # Self-critique
        critique_prompt = f"""
Previous answer: {response}
Critically analyze the previous answer. Identify any mistakes, unclear explanations, or areas for improvement.
Provide a detailed critique:
"""
        critique = query_model(model, critique_prompt)
        
        # Check if no improvements are needed
        if "no improvements needed" in critique.lower():
            break
        
        # Self-refinement
        refine_prompt = f"""
Question: {question}
Previous answer: {response}
Critique: {critique}
Based on this critique, provide an improved and refined answer to the original question.
"""
        refined_response = query_model(model, refine_prompt)
        
        # Update the response
        response = refined_response
    
    # If the final answer is not clear, ask for it explicitly
    if "final answer is:" not in response.lower():
        final_prompt = f"""
Based on your previous response to the question: {question}
Provide a concise final answer. Start your response with "Therefore, the final answer is:"
"""
        response = query_model(model, final_prompt)
    
    return response

def tree_of_thoughts(model: str, question: str, max_depth: int = 3, num_children: int = 3, max_rollouts = ROLLOUTS) -> str:
    """Implement Tree of Thoughts technique."""
    rollouts = 0
    def expand_node(thought: str, depth: int) -> List[str]:
        nonlocal rollouts
        if depth >= max_depth or rollouts >= max_rollouts:
            return []
        rollouts += 1
        prompt = f"Based on the thought: '{thought}', generate {num_children} new thoughts to solve: {question}"
        response = query_model(model, prompt)
        return response.split('\n')[:num_children]

    def evaluate_node(thought: str) -> float:
        prompt = f"Evaluate how likely this thought leads to the correct answer for the question: '{question}'\nThought: {thought}\nScore (0 to 1):"
        response = query_model(model, prompt)
        try:
            return float(response)
        except ValueError:
            return 0.0

    root = query_model(model, f"Initial thought for solving: {question}")
    best_thought = root
    best_score = evaluate_node(root)

    for depth in range(max_depth):
        children = expand_node(best_thought, depth)
        for child in children:
            score = evaluate_node(child)
            if score > best_score:
                best_thought = child
                best_score = score

    final_prompt = f"Based on the thought: '{best_thought}', provide a final answer to the question: {question}"
    return query_model(model, final_prompt)

def mctsr_self_eval(model: str, question: str, num_simulations: int = ROLLOUTS, exploration_weight: float = 1.4,gamma = 0.8) -> str:
    """Implement naive Monte Carlo Tree Search."""
    class Node:
        def __init__(self, state: str, parent=None):
            self.state = state
            self.parent = parent
            self.children = []
            self.visits = 1
            self.value = 0

        def ucb1(self) -> float:
            if self.visits == 0:
                return float('inf')
            return self.value / self.visits + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

    root = Node(question)

    for _ in range(num_simulations):
        node = root
        path = []

        # Selection
        while node.children:
            node = max(node.children, key=lambda n: n.ucb1())
            path.append(node)

        # Expansion
        if node.visits > 0:
            prompt = f"Question: {question}\nPrevious answer: {node.state}\nReflect on your previous answer. Are there any mistakes or areas for improvement? Please provide an improved answer.'"
            new_state = query_model(model, prompt)
            new_node = Node(new_state, parent=node)
            node.children.append(new_node)
            path.append(new_node)
            node = new_node

        # Simulation
        prompt = f"Evaluate how likely this answer is equal to the correct answer for: {question}\nAnswer: {path[-1].state}\nScore (0 to 1):"
        result = query_model(model, prompt)
        try:
            score = float(result)
        except ValueError:
            score = 0.0

        # Backpropagation
        for node in path:
            node.visits += 1
            score = gamma * score
            node.value += score
            
    # Selection
    node = root
    while node.children:
        node = max(node.children, key=lambda n: n.value)
        path.append(node)

    # best_child = max(root.children, key=lambda n: n.visits)
    # final_prompt = f"Based on this path: {' -> '.join([n.state for n in path])}, provide a final answer to the question: {question}"
    response = path[-1].state# query_model(model, final_prompt)
    if "final answer is:" not in response.lower():
        final_prompt = f"Based on your solution to the question: {question} and Answer: {response.lower()}\nWhat is the final numeric answer? Please respond with 'Therefore, the final answer is: [answer].'"
        response = query_model(model, final_prompt)
    return response

def cot_mcts(model: str, question: str, num_simulations: int = ROLLOUTS, exploration_weight: float = 1.4,gamma = 0.8) -> str:
    """Implement naive Monte Carlo Tree Search."""
    class Node:
        def __init__(self, state: str, parent=None):
            self.state = state
            self.parent = parent
            self.children = []
            self.visits = 1
            self.value = 0

        def ucb1(self) -> float:
            if self.visits == 0:
                return float('inf')
            return self.value / self.visits + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

    root = Node(question)

    for _ in range(num_simulations):
        node = root
        path = []

        # Selection
        while node.children:
            node = max(node.children, key=lambda n: n.ucb1())
            path.append(node)

        # Expansion
        if node.visits > 0:
            prompt = f"Based on: '{node.state}', generate a new thought or step towards solving: {question}"
            new_state = query_model(model, prompt)
            new_node = Node(new_state, parent=node)
            node.children.append(new_node)
            path.append(new_node)
            node = new_node

        # Simulation
        prompt = f"Evaluate how likely this path leads to the correct answer for: {question}\nPath: {' -> '.join([n.state for n in path])}\nScore (0 to 1):"
        result = query_model(model, prompt)
        try:
            score = float(result)
        except ValueError:
            score = 0.0

        # Backpropagation
        for node in path:
            node.visits += 1
            score = gamma * score
            node.value += score
            
    # Selection
    node = root
    while node.children:
        node = max(node.children, key=lambda n: n.value)
        path.append(node)

    # best_child = max(root.children, key=lambda n: n.visits)
    final_prompt = f"Based on this path: {' -> '.join([n.state for n in path])}, provide a final answer to the question: {question}"
    return query_model(model, final_prompt)

def self_consistency(model: str, question: str, num_samples: int = ROLLOUTS) -> str:
    def sample_outputs() -> List[Tuple[str, str]]:
        sampled_outputs = []
        for _ in range(num_samples):
            prompt = f"Question: {question}\nProvide a step-by-step solution and final answer:"
            output = query_model(model, prompt)
            reasoning_path, answer = parse_output(output)
            sampled_outputs.append((reasoning_path, answer))
        return sampled_outputs
    
    def parse_output(output: str) -> Tuple[str, str]:
        answer = extract_final_answer(output)
        reasoning_path = output.split("final answer is:", 1)[0].strip() if "final answer is:" in output.lower() else output
        return reasoning_path, answer if answer is not None else ""

    def get_rejection_reasons(sampled_outputs: List[Tuple[str, str]]) -> Dict[str, Any]:
        def check_rejection_reason(reasoning_path: str, answer: str) -> Optional[Any]:
            if len(reasoning_path.split()) < 10:
                return "Too short reasoning"
            if not answer:
                return "No answer provided"
            return None

        rejection_reasons = {}
        for reasoning_path, answer in sampled_outputs:
            reason = check_rejection_reason(reasoning_path, answer)
            if reason:
                rejection_reasons[reasoning_path] = reason
        return rejection_reasons

    def adjust_outputs(sampled_outputs: List[Tuple[str, str]], rejection_reasons: Dict[str, Any]) -> List[Tuple[str, str]]:
        def adjust_reasoning_path(reasoning_path: str, reason: Any) -> str:
            if reason == "Too short reasoning":
                return reasoning_path + " This reasoning path was expanded to provide more detail."
            if reason == "No answer provided":
                return reasoning_path + " An answer was inferred based on the reasoning."
            return reasoning_path

        adjusted_outputs = []
        for reasoning_path, answer in sampled_outputs:
            if reasoning_path not in rejection_reasons:
                adjusted_outputs.append((reasoning_path, answer))
            else:
                adjusted_reasoning_path = adjust_reasoning_path(reasoning_path, rejection_reasons[reasoning_path])
                adjusted_outputs.append((adjusted_reasoning_path, answer or "Inferred answer"))
        return adjusted_outputs

    def aggregate_answers(adjusted_outputs: List[Tuple[str, str]]) -> List[str]:
        return [answer for _, answer in adjusted_outputs]

    def find_most_consistent_answer(aggregated_answers: List[str]) -> str:
        counter = Counter(aggregated_answers)
        most_consistent_answer, _ = counter.most_common(1)[0]
        return most_consistent_answer

    sampled_outputs = sample_outputs()
    rejection_reasons = get_rejection_reasons(sampled_outputs)
    adjusted_outputs = adjust_outputs(sampled_outputs, rejection_reasons)
    aggregated_answers = aggregate_answers(adjusted_outputs)
    return find_most_consistent_answer(aggregated_answers)

methods = {
    # "repeatead_sampling": repeatead_sampling,
    "self_refine": self_refine,
    "tree_of_thoughts": tree_of_thoughts,
    # "mctsr_self_eval": mctsr_self_eval,
    # "cot_mcts": cot_mcts,
    "self_consistency": self_consistency
}

import concurrent.futures
from typing import Any, Dict
import random
from tqdm import tqdm

import concurrent.futures
from typing import Dict, Any

def process_question(model: str, item: Dict[str, Any], techniques: list) -> Dict[str, Dict[str, int]]:
    question = item['question']
    correct_answer = item['answer']
    results = {technique: {'correct': 0, 'total': 0} for technique in techniques}

    def process_technique(technique):
        answer = methods[technique](model, question)
        if answer is not None:
            is_correct = evaluate_answer(answer, correct_answer)
            print(f"{technique} - Question: {question}\nAnswer: {answer}\nCorrect: {is_correct}")
            if is_correct:
                results[technique]['correct'] += 1
            results[technique]['total'] += 1
        else:
            print(f"{technique} failed to provide a valid answer.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(process_technique, techniques))

    # print("-" * 50)
    return results

def run_benchmark(model: str, dataset: Any, num_questions: int = 5) -> Dict[str, float]:
    """Run benchmark for a given model and dataset, using all techniques on the same questions."""
    dataset_size = len(dataset)
    num_questions = dataset_size #min(num_questions, dataset_size) #dataset_size #
    sampled_questions = random.sample(list(dataset), num_questions)
    
    techniques = list(methods.keys())
    print(techniques)
    aggregated_results = {technique: {'correct': 0, 'total': 0} for technique in techniques}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_question, model, item, techniques) for item in sampled_questions]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            for technique in techniques:
                aggregated_results[technique]['correct'] += result[technique]['correct']
                aggregated_results[technique]['total'] += result[technique]['total']
    
    return {technique: aggregated_results[technique]['correct'] / aggregated_results[technique]['total'] if aggregated_results[technique]['total'] > 0 else 0.0 
            for technique in techniques}

import concurrent.futures
from typing import Dict

def run_all_benchmarks() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run benchmarks for all models and datasets, using all techniques on the same questions."""
    datasets = load_datasets()
    results = {model: {} for model in MODELS}

    def run_dataset_benchmark(model, dataset_name, dataset):
        print(f"Running benchmark for {model} on {dataset_name}")
        return dataset_name, run_benchmark(model, dataset)

    def run_model_benchmark(model):
        model_results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_dataset_benchmark, model, dataset_name, dataset) for dataset_name, dataset in datasets.items()]
            for future in concurrent.futures.as_completed(futures):
                dataset_name, result = future.result()
                model_results[dataset_name] = result
        return model, model_results

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_model_benchmark, model) for model in MODELS]
        for future in concurrent.futures.as_completed(futures):
            model, model_results = future.result()
            results[model] = model_results

    return results

def draw_results(results: Dict[str, Dict[str, Dict[str, float]]]):
    """Visualize the benchmark results in a single figure using bar plots."""
    models = list(results.keys())
    techniques = [ 'tree_of_thoughts',  'self_consistency']
    datasets = set()
    for model in models:
        datasets.update(results[model].keys())
    datasets = sorted(list(datasets))
    
    data = []
    for model in models:
        for dataset in datasets:
            if dataset in results[model]:
                for technique in techniques:
                    data.append({
                        'Model': model,
                        'Technique': technique,
                        'Dataset': dataset,
                        'Accuracy': results[model][dataset][technique]
                    })
    
    df = pd.DataFrame(data)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Define color palette
    colors = ['#0D408C', '#8186D8', '#BF83BA', '#FFDFD3', '#FC8D62', '#65C3A5', '#E78AC3', '#FFD966']
    palette = {technique: color for technique, color in zip(techniques, colors)}
    
    # Create the grouped bar plot
    sns.barplot(x='Dataset', y='Accuracy', hue='Technique', data=df, ax=ax, palette=palette)
    
    # Customize the plot
    ax.set_title("Performance Comparison Across Models, Techniques, and Datasets", fontsize=16)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(title='Technique', title_fontsize='12', fontsize='10', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('results/advanced_techniques_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Results visualization saved as 'advanced_techniques_benchmark_results.png'.")

def save_results_to_csv(results: Dict[str, Dict[str, float]], filename: str = f"results/advanced_techniques_benchmark_results{ROLLOUTS}_{time.time()}.csv"):
    """Save the benchmark results to a CSV file."""
    # Convert results to a DataFrame for easier saving
    data = []
    for model, datasets in results.items():
        for dataset, accuracy in datasets.items():
            data.append({
                'Model': model,
                'Dataset': dataset,
                'Accuracy': accuracy
            })
    
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"Results saved to '{filename}'.")

if __name__ == "__main__":

    get_clients()
    results = run_all_benchmarks()
    # draw_results(results)
    save_results_to_csv(results)