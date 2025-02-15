import pprint
from typing import Dict, Any, List

from final.equivalent_ans import is_equivalent_answer, is_equivalent_step
from lightllm_api.llm_api import gpt_api_caller, qwen_api_caller
# from base_model import  gpt_api_caller
# from test_api import query_api

llm_api = gpt_api_caller # 替换为你的LLM调用函数

###############################################################################
# Rollout / Equivalence-Check Counters
###############################################################################
rollout_metrics = {
    "total_rollout_calls": 0,
    "equiv_check_calls": 0,
    "rollout_prompt_intervention_count": 0,
    "rollout_direct_count": 0
}


def counted_rollout_call(do_type: int, message: List[Dict[str, str]]) -> str:
    """
    Wraps qwen_api_caller to count how many rollouts are made 
    and whether they are prompt-intervention or direct rollouts.
    """
    # Increment total rollouts
    rollout_metrics["total_rollout_calls"] += 1

    # Count the rollout type
    if do_type == 1:
        print("rollout的类型是:prompt-intervention")
        rollout_metrics["rollout_prompt_intervention_count"] += 1
    else:
        print("rollout的类型是:direct")
        rollout_metrics["rollout_direct_count"] += 1

    # Call the actual LLM API
    output = llm_api(message, 
                    #  temperature=1
                     )
    print("rollout的新节点是:",output)
    return output


def counted_equiv_check(candidate: str, ground_truth: str, check_answer=False) -> bool:
    """
    Wraps is_equivalent_answer to count how many times equivalence checks occur.
    """
    rollout_metrics["equiv_check_calls"] += 1

    if check_answer: # True代表判断结果是否正确
        return is_equivalent_answer(candidate, ground_truth)
    # False代表判断两个step意义是否一致
    return is_equivalent_step(candidate, ground_truth)

###############################################################################
# Core Functions
###############################################################################

def parse_nodes(text: str) -> List[str]:
    """
    Splits the chain-of-thought (CoT) string into a list of nodes/steps.
    """
    nodes = text.split('\n\n')
    return [node.strip() for node in nodes if node.strip()]


def get_original_metrics(response: str, ground_truth: str) -> Dict[str, Any]:
    """
    Computes the original metrics from the unmodified chain-of-thought:
    - original_token_length
    - original_accuracy
    - original_step_length
    - original_ps
    """
    # Approximate token length by counting words
    original_token_length = len(response.split())
    # Split steps
    original_steps = parse_nodes(response)
    original_step_length = len(original_steps)

    if original_steps:
        final_answer = original_steps[-1]
        flag = counted_equiv_check(final_answer, ground_truth,check_answer=True)
        print("判断原cot推理结果是否正确:",flag)
        original_ps = 1 if flag else 0
    else:
        final_answer = ""
        original_ps = 0

    metrics = {
        "original_token_length": original_token_length,
        "original_step_length": original_step_length,
        "original_accuracy": original_ps,
        "original_ps": original_ps,
    }
    return metrics


def generate_replacement_step(
    query: str,
    context_steps: List[str],
    current_step: str,
    do_type: int
) -> str:
    """
    Generates a replacement step for the current_step based on do_type logic.
    Incorporates the original query into the user content.
    """
    if do_type == 1:
        # Based on prompt intervention (skip current meaning)
        skip_message = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Continue solving the problem with mathematical expressions only, without repeating previous steps. "
                    "Provide the final answer once, ensuring it is directly connected to the preceding reasoning "
                    "and without any additional summaries or explanations. Ensure the next output node does not match "
                    f"the meaning of:\n{current_step}"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    "Current reasoning steps:\n"
                    + '\n\n'.join(context_steps)
                    + '\n\n'
                )
            }
        ]
        replacement_text = counted_rollout_call(do_type, skip_message)
    else:
        # Direct rollout with the query plus existing steps
        context_message = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Continue solving the problem with mathematical expressions only, without repeating previous steps. "
                    "Provide the final answer once, ensuring it is directly connected to the preceding reasoning "
                    "and without any additional summaries or explanations."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    "Current reasoning steps:\n"
                    + '\n\n'.join(context_steps)
                    + '\n\n'
                )
            }
        ]
        replacement_text = counted_rollout_call(do_type, context_message)

    return replacement_text


def ensure_different_step(
    query: str,
    replacement_text: str,
    original_step: str,
    context_steps: List[str],
    alter_attempts: int,
    do_type: int
) -> List[str]:
    """
    Ensures the first replacement step is different from the original step
    by re-generating up to alter_attempts times if necessary.
    Returns a list of replacement steps.
    """
    for attempt in range(1, alter_attempts + 1):
        replacement_steps = parse_nodes(replacement_text)

        # If no steps or the first new step is not equivalent to the original step, return
        flag = counted_equiv_check(replacement_steps[0], original_step)
        if not replacement_steps or not flag:
            print("新rollout的节点和原先不一致")
            return replacement_steps
        else:
            # If the new step is still too similar, try regenerating
            print("新rollout的节点和原先一致,重新生成")
            replacement_text = generate_replacement_step(query, context_steps, original_step, do_type)

            if attempt == alter_attempts:
                print("******达到最大尝试次数****")
                print(f"Reached max attempts ({alter_attempts}); replacement step is still similar.")
                return None

    return parse_nodes(replacement_text)


def evaluate_replacement_step(
    query: str,
    context_steps: List[str],
    candidate_step: str,
    ground_truth: str,
    reasoning_attempts: int
) -> float:
    """
    Given a single candidate replacement step, attempts forward passes reasoning_attempts times.
    Returns the average correctness (average Y value).
    """
    y_values = []

    for _ in range(reasoning_attempts):
        eval_context_steps = context_steps + [candidate_step]
        eval_message = [
            {
                "role": "system",
                "content": (
                "You are a helpful assistant. Continue solving the problem with mathematical expressions only, without repeating previous steps. "
                    "Provide the final answer once, ensuring it is directly connected to the preceding reasoning "
                    "and without any additional summaries or explanations."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    "Current reasoning steps:\n"
                    + '\n\n'.join(eval_context_steps)
                )
            }
        ]
        # Treat these as direct rollouts (do_type=0)
        evaluation_text = counted_rollout_call(0, eval_message)
        if not evaluation_text:
            # No generation; consider this a wrong answer
            y_values.append(0)
        else:
            eval_nodes = parse_nodes(evaluation_text)
            eval_final_answer = eval_nodes[-1] if eval_nodes else ""
            is_correct = counted_equiv_check(eval_final_answer, ground_truth,check_answer=True)
            print("新rollout的节点得到的最终答案是否正确:",is_correct)
            y_values.append(1 if is_correct else 0)

    if len(y_values) == 0:
        return 0.0
    return sum(y_values) / len(y_values)


def update_chain_if_needed(
    query: str,
    nodes: List[str],
    i: int,
    ground_truth: str,
    threshold: float,
    reasoning_attempts: int,
    do_type: int,
    alter_attempts: int
):
    """
    Attempts to intervene on a specific node (at index i) if beneficial.
    Updates the chain of nodes and returns the new chain, the new index,
    and the PN value computed for this step.
    """
    current_step = nodes[i]
    context_steps = nodes[:i]

    print("\n==== Intervening on step", i, "====")
    print("Context so far:", context_steps)
    print("Current step:", current_step)

    # Generate initial replacement text
    replacement_text = generate_replacement_step(query, context_steps, current_step, do_type)

    # Ensure the replacement text is different from the original step
    replacement_steps = ensure_different_step(
        query, replacement_text, current_step, context_steps, alter_attempts, do_type
    )
    if not replacement_steps:
        print('跳过节点,pn为None')
        print(f"Skipping PN estimation and further processing for step {i}")
        # Skip this node and move to the next one without evaluating its PN
        i += 1
        return nodes, i, None  # Returning None to indicate this step is skipped
    else:
        # Evaluate the replacement step over multiple forward passes
        candidate_step = replacement_steps[0] if replacement_steps else ""
        average_y = evaluate_replacement_step(
            query,
            context_steps,
            candidate_step,
            ground_truth,
            reasoning_attempts
        )
        pn = 1 - average_y
        print(f"PN value for step {i}: {pn}")

    # If PN is below threshold, we intervene (replace) from this step onward
    if pn < threshold:
        print(f"PN < {threshold}, updating chain from step {i} onward...")
        # Re-run a final pass with the best candidate step to get its chain
        best_replacement_text = candidate_step
        best_eval_message = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Continue solving the problem with mathematical expressions only, without repeating previous steps. "
                    "Provide the final answer once, ensuring it is directly connected to the preceding reasoning "
                    "and without any additional summaries or explanations."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    "Current reasoning steps:\n"
                    + '\n\n'.join(context_steps + [best_replacement_text])
                )
            }
        ]
        best_evaluation_text = counted_rollout_call(0, best_eval_message)
        best_eval_nodes = parse_nodes(best_evaluation_text)

        # Replace from current step onward
        nodes = context_steps + best_eval_nodes
        # Move i to the end of the newly added nodes
        i = len(nodes)
    else:
        # Otherwise, just move on
        i += 1

    return nodes, i, pn


def calculate_ps_pn(
    query: str,
    response: str,
    ground_truth: str,
    threshold: float = 0.3,
    reasoning_attempts: int = 3,
    do_type: int = 0,
    alter_attempts: int = 5
) -> Dict[str, Any]:
    """
    Main function that calculates:
    - PS(chain): Whether final answer is equivalent to ground_truth
    - PN values for each step that might be replaced
    - Token length, step length, etc.
    - Also collects counts of rollouts and equivalence checks.
    """
    # Reset metrics counters each time we call
    rollout_metrics["total_rollout_calls"] = 0
    rollout_metrics["equiv_check_calls"] = 0
    rollout_metrics["rollout_prompt_intervention_count"] = 0
    rollout_metrics["rollout_direct_count"] = 0

    # 1. Get original metrics from the chain-of-thought
    original_metrics = get_original_metrics(response, ground_truth)

    # 2. Parse the chain into nodes
    nodes = parse_nodes(response)
    pprint.pprint(nodes)

    # 3. Intervene on each node
    i = 0
    pn_values = []
    while i < len(nodes):
        nodes, i, pn = update_chain_if_needed(
            query=query,
            nodes=nodes,
            i=i,
            ground_truth=ground_truth,
            threshold=threshold,
            reasoning_attempts=reasoning_attempts,
            do_type=do_type,
            alter_attempts=alter_attempts
        )
        # Only append pn if it's not None
        if pn is not None:
            pn_values.append(pn)

    # 4. After all possible interventions, compute final metrics
    final_answer = nodes[-1] if nodes else ""
    ps = 1 if counted_equiv_check(final_answer, ground_truth,check_answer=True) else 0
    print("最终的cot是否正确(PS):",ps)
    final_chain = '\n\n'.join(nodes)

    # Approximate token length by counting words
    token_length = len(final_chain.split())
    step_length = len(nodes)

    # Assemble final results
    results = {
        "original_token_length": original_metrics["original_token_length"],
        "original_accuracy": original_metrics["original_accuracy"],
        "original_step_length": original_metrics["original_step_length"],
        "avg_PN(steps)": sum(pn_values)/len(pn_values) if pn_values else None,
        "max_PN(steps)": max(pn_values) if pn_values else None,
        "min_PN(steps)": min(pn_values) if pn_values else None,
        "PS(chain)": ps,
        "token_length": token_length,
        "accuracy": ps,  # final PS is the final accuracy
        "step_length": step_length,
        "final_chain": nodes,

        # New counters
        "total_rollout_calls": rollout_metrics["total_rollout_calls"],
        "equiv_check_calls": rollout_metrics["equiv_check_calls"],
        "rollout_prompt_intervention_count": rollout_metrics["rollout_prompt_intervention_count"],
        "rollout_direct_count": rollout_metrics["rollout_direct_count"],
    }

    return results


# -----------------------------------------------------------------------------
# Example usage:
# if __name__ == "__main__":
#     example_query = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
#     # The chain-of-thought (response) below excludes the question as its first node.
#     example_llm_response = """Alright, let's see. So, Natalia sold clips to 48 of her friends in April, and then in May, she sold half as many clips. I need to find out how many clips she sold altogether in these two months.\n\n

# First, in April, she sold 48 clips. That's straightforward. Now, in May, she sold half as many as in April. So, half of 48. Let's think, half of 40 is 20, and half of 8 is 4, so half of 48 is 24. So, in May, she sold 24 clips.\n\n

# To find the total, I need to add the clips sold in April and May together. So, 48 plus 24. Let's see, 48 plus 20 is 68, and then plus 4 more is 72. So, altogether, she sold 72 clips in April and May.\n\n

# Wait, is there another way to look at this? Maybe I can think of it as April being one part and May being half a part, so together it's one and a half parts. If one part is 48, then one and a half parts would be 48 plus 24, which again is 72. Yeah, that matches what I got before.\n\n

# I don't think I'm missing anything here. It seems pretty straightforward. She sold 48 in April, half that in May, which is 24, and together that's 72 clips.

#     """
#     ground_truth = "72"

#     results = calculate_ps_pn(
#         query=example_query,
#         response=example_llm_response,
#         ground_truth=ground_truth
#     )

#     print("\nFinal chain of steps:")
#     pprint.pprint(results["final_chain"])

#     metrics = {k: v for k, v in results.items() if k != "final_chain"}
#     print("\nMetrics:")
#     pprint.pprint(metrics)
if __name__ == "__main__":
    print("\nRunning example...")
    example_query = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    ground_truth = "72"
    example_llm_response = llm_api([{"role": "user", "content": example_query}])
    print("\nexample_llm_response:",example_llm_response)
    results = calculate_ps_pn(
        query=example_query,
        response=example_llm_response,
        ground_truth=ground_truth,
        alter_attempts=3,
        do_type=1,
    )

    print("\nFinal chain of steps:")
    pprint.pprint(results["final_chain"])

    metrics = {k: v for k, v in results.items() if k != "final_chain"}
    print("\nMetrics:")
    pprint.pprint(metrics)