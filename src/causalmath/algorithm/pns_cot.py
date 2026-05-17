import pprint
import sys
import threading
from typing import Dict, Any, List
import os
from causalmath.utils.prompts import math_prompt, common_prompt
from causalmath.algorithm.equivalence import is_equivalent_answer, is_equivalent_reasoning_re, is_equivalent_reasoning, is_equivalent_step
from causalmath.models.ollama_client import cerebras_query
from concurrent.futures import ThreadPoolExecutor

def query_api(messages, model=None, temperature=0.7):
    # Rollouts use thinking=False — without internal deliberation the model is
    # weaker and more likely to fail when a necessary step is removed, giving
    # realistic PN signal instead of PN=0 for everything.
    return cerebras_query(messages, model=model, temperature=temperature, thinking=False)

llm_api = query_api
total_prompt = math_prompt
is_commonsense = False  # set to True by run_algo_o.py for CSQA datasets
###############################################################################
# Rollout / Equivalence-Check Counters
###############################################################################
def _new_metrics():
    return {
        "total_rollout_calls": 0,
        "equiv_check_calls": 0,
        "rollout_prompt_intervention_count": 0,
        "rollout_direct_count": 0,
    }


def counted_rollout_call(do_type: int, message: List[Dict[str, str]],
                         _m: dict, _m_lock: threading.Lock) -> str:
    with _m_lock:
        _m["total_rollout_calls"] += 1
        if do_type == 1:
            _m["rollout_prompt_intervention_count"] += 1
        else:
            _m["rollout_direct_count"] += 1

    if do_type == 1:
        print("Rollout type: prompt-intervention")
    else:
        print("Rollout type: direct")

    output = llm_api(message)
    print("New rollout node:", output)
    return output


def counted_equiv_check(candidate: str, ground_truth: str,
                        _m: dict, _m_lock: threading.Lock,
                        check_answer=False) -> bool:
    with _m_lock:
        _m["equiv_check_calls"] += 1

    if check_answer:
        if is_commonsense:
            return is_equivalent_reasoning_re(candidate, ground_truth)
        return is_equivalent_answer(candidate, ground_truth)
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


def get_original_metrics(response: str, ground_truth: str,
                         _m: dict, _m_lock: threading.Lock) -> Dict[str, Any]:
    original_token_length = len(response.split())
    original_steps = parse_nodes(response)
    original_step_length = len(original_steps)

    if original_steps:
        final_answer = original_steps[-1]
        flag = counted_equiv_check(final_answer, ground_truth, _m, _m_lock, check_answer=True)
        print("Is original CoT reasoning correct:", flag)
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
    do_type: int,
    _m: dict,
    _m_lock: threading.Lock,
) -> str:
    """
    Generates a replacement step for the current_step based on do_type logic.
    Incorporates the original query into the user content.
    """
    # Bug fix: force single-step output so the replacement is a reasoning step,
    # not the full answer. Without this constraint, thinking=False models jump
    # straight to the boxed answer, making PN estimation trivial (always 0).
    single_step_instruction = (
        "\nIMPORTANT: Output ONLY ONE single reasoning step — not the full solution. "
        "Do NOT jump to the final answer. Stop after one step."
    )

    if do_type == 1:
        # Based on prompt intervention (skip current meaning)
        skip_message = [
            {
                "role": "system",
                "content": (
                    total_prompt
                    + single_step_instruction
                    + "\nEnsure this step does NOT convey the same meaning as:\n"
                    + current_step
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    "Current reasoning steps:\n"
                    + '\n\n'.join(context_steps)
                    + '\n\n'
                    "Next step (ONE step only, different from the original):"
                )
            }
        ]
        replacement_text = counted_rollout_call(do_type, skip_message, _m, _m_lock)
    else:
        context_message = [
            {
                "role": "system",
                "content": total_prompt + single_step_instruction
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    "Current reasoning steps:\n"
                    + '\n\n'.join(context_steps)
                    + '\n\n'
                    "Next step (ONE step only):"
                )
            }
        ]
        replacement_text = counted_rollout_call(do_type, context_message, _m, _m_lock)

    return replacement_text


def ensure_different_step(
    query: str,
    replacement_text: str,
    original_step: str,
    context_steps: List[str],
    alter_attempts: int,
    do_type: int,
    _m: dict,
    _m_lock: threading.Lock,
) -> List[str]:
    """
    Ensures the first replacement step is different from the original step
    by re-generating up to alter_attempts times if necessary.
    Returns a list of replacement steps.
    """
    for attempt in range(1, alter_attempts + 1):
        replacement_steps = parse_nodes(replacement_text)

        # If no steps or the first new step is not equivalent to the original step, return
        flag = counted_equiv_check(replacement_steps[0], original_step, _m, _m_lock)
        if not replacement_steps or not flag:
            print("New rollout node differs from original")
            return replacement_steps
        else:
            print("New rollout node is same as original, regenerating...")
            replacement_text = generate_replacement_step(query, context_steps, original_step, do_type, _m, _m_lock)

            if attempt == alter_attempts:
                print("****** Reached maximum attempts ****")
                print(f"Reached max attempts ({alter_attempts}); replacement step is still similar.")
                return None

    return parse_nodes(replacement_text)


def evaluate_replacement_step(
    query: str,
    context_steps: List[str],
    candidate_step: str,
    ground_truth: str,
    reasoning_attempts: int,
    _m: dict,
    _m_lock: threading.Lock,
) -> float:
    eval_context_steps = context_steps + [candidate_step]
    # Bug fix: add a strong continuation constraint so the rollout model cannot
    # re-derive the answer from the original question. Without this, even a
    # badly wrong candidate_step leads to a correct rollout (average_y≈1, PN≈0
    # for every step). The model must continue strictly from the given steps.
    continuation_instruction = (
        "\nCRITICAL: Continue the reasoning STRICTLY from the partial steps provided. "
        "Do NOT re-read or re-solve from the original question. "
        "Pick up exactly where the last step left off."
    )
    eval_message = [
        {"role": "system", "content": total_prompt + continuation_instruction},
        {
            "role": "user",
            "content": (
                f"Question (for reference only — do not restart):\n{query}\n\n"
                "Partial reasoning so far:\n"
                + '\n\n'.join(eval_context_steps)
                + "\n\nContinue strictly from the last step above:"
            )
        }
    ]

    def single_rollout(_):
        evaluation_text = counted_rollout_call(0, eval_message, _m, _m_lock)
        if not evaluation_text:
            return 0
        eval_nodes = parse_nodes(evaluation_text)
        eval_final_answer = eval_nodes[-1] if eval_nodes else ""
        is_correct = counted_equiv_check(
            eval_final_answer, ground_truth, _m, _m_lock, check_answer=True
        )
        print("Is new rollout final answer correct:", is_correct)
        return 1 if is_correct else 0

    # max_workers=2 matches OLLAMA_NUM_PARALLEL=2: sending more threads than Ollama's
    # parallel capacity just queues requests and adds overhead without GPU benefit.
    with ThreadPoolExecutor(max_workers=2) as executor:
        y_values = list(executor.map(single_rollout, range(reasoning_attempts)))

    return sum(y_values) / len(y_values) if y_values else 0.0

def update_chain_if_needed(
    query: str,
    nodes: List[str],
    i: int,
    ground_truth: str,
    threshold: float,
    reasoning_attempts: int,
    do_type: int,
    alter_attempts: int,
    _m: dict,
    _m_lock: threading.Lock,
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

    replacement_text = generate_replacement_step(query, context_steps, current_step, do_type, _m, _m_lock)

    replacement_steps = ensure_different_step(
        query, replacement_text, current_step, context_steps, alter_attempts, do_type, _m, _m_lock
    )
    if not replacement_steps:
        print('Skipping node, PN is None')
        print(f"Skipping PN estimation and further processing for step {i}")
        i += 1
        return nodes, i, None
    else:
        candidate_step = replacement_steps[0] if replacement_steps else ""
        average_y = evaluate_replacement_step(
            query,
            context_steps,
            candidate_step,
            ground_truth,
            reasoning_attempts,
            _m,
            _m_lock,
        )
        pn = 1 - average_y
        print(f"PN value for step {i}: {pn}")

    # Bug fix: paper Algorithm 1 says "Skip st" (remove it) when PN < threshold.
    # The old code replaced the ENTIRE remaining chain with a fresh rollout, causing
    # chain collapse: thinking=False model outputs just the boxed answer in one step,
    # so len(nodes) drops to 1 and the while loop exits after a single PN measurement.
    # Now we simply drop step i and let remaining original steps slide into position.
    if pn < threshold:
        print(f"PN < {threshold}, pruning step {i} (removing from chain)...")
        nodes = nodes[:i] + nodes[i+1:]
        # Do NOT increment i — the next original step is now at position i.
    else:
        # Step is necessary; keep it and advance.
        i += 1

    return nodes, i, pn


def calculate_ps_pn(
    query: str,
    response: str,
    ground_truth: str,
    threshold: float = 0.5,
    reasoning_attempts: int = 5,
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
    _m = _new_metrics()
    _m_lock = threading.Lock()

    # 1. Get original metrics from the chain-of-thought
    original_metrics = get_original_metrics(response, ground_truth, _m, _m_lock)

    # 2. Parse the chain into nodes
    nodes = parse_nodes(response)
    pprint.pprint(nodes)

    # Algorithm 1: if PS=0, chain is insufficient — Lemma 1 requires PS=1 for PN identifiability.
    # Return S_init unchanged without running PN estimation.
    if original_metrics["original_ps"] == 0:
            return {
                "original_token_length": original_metrics["original_token_length"],
                "original_accuracy": 0,
                "original_step_length": original_metrics["original_step_length"],
                "avg_PN(steps)": None,
                "max_PN(steps)": None,
                "min_PN(steps)": None,
                "PS(chain)": 0,
                "token_length": original_metrics["original_token_length"],
                "accuracy": 0,
                "step_length": original_metrics["original_step_length"],
                "final_chain": nodes,
                "pn_per_step": [],
                "total_rollout_calls": _m["total_rollout_calls"],
                "equiv_check_calls": _m["equiv_check_calls"],
                "rollout_prompt_intervention_count": 0,
                "rollout_direct_count": 0,
            }

    # 3. Intervene on each node
    # Skip the last step: it contains the final answer. Any alternative that also
    # states the correct answer gives average_y≈1 → PN≈0 → it gets pruned →
    # chain ends with no answer → revert to original (no reduction ever happens).
    # The last step is always necessary by definition, so we never prune it.

    i = 0
    pn_values = []
    while i < len(nodes) - 1:  # stop before final answer step
        nodes, i, pn = update_chain_if_needed(
            query=query,
            nodes=nodes,
            i=i,
            ground_truth=ground_truth,
            threshold=threshold,
            reasoning_attempts=reasoning_attempts,
            do_type=do_type,
            alter_attempts=alter_attempts,
            _m=_m,
            _m_lock=_m_lock,
        )
        # Only append pn if it's not None
        if pn is not None:
            pn_values.append(pn)

    # 4. After all possible interventions, compute final metrics
    final_answer = nodes[-1] if nodes else ""
    ps = 1 if counted_equiv_check(final_answer, ground_truth, _m, _m_lock, check_answer=True) else 0
    print("Is final CoT correct (PS):", ps)

    # If optimization broke a previously correct chain, revert to original
    if ps == 0 and original_metrics["original_ps"] == 1:
        print("Replacement chain incorrect — reverting to original chain.")
        nodes = parse_nodes(response)
        ps = 1

    final_chain = nodes
    token_length = len('\n\n'.join(nodes).split())
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
        "final_chain": final_chain,  # Use the string, not the list
        "pn_per_step": pn_values,

        # New counters
        "total_rollout_calls": _m["total_rollout_calls"],
        "equiv_check_calls": _m["equiv_check_calls"],
        "rollout_prompt_intervention_count": _m["rollout_prompt_intervention_count"],
        "rollout_direct_count": _m["rollout_direct_count"],
    }

    return results


def check_and_intervene(
    query: str,
    nodes: List[str],
    i: int,
    ground_truth: str,
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

    _m_legacy = _new_metrics()
    _ml_lock = threading.Lock()
    replacement_text = generate_replacement_step(query, context_steps, current_step, do_type, _m_legacy, _ml_lock)

    replacement_steps = ensure_different_step(
        query, replacement_text, current_step, context_steps, alter_attempts, do_type, _m_legacy, _ml_lock
    )
    if not replacement_steps:
        print('Skipping node, PN is None')
        print(f"Skipping PN estimation and further processing for step {i}")
        i += 1
        return nodes, i, None
    else:
        candidate_step = replacement_steps[0] if replacement_steps else ""
        average_y = evaluate_replacement_step(
            query,
            context_steps,
            candidate_step,
            ground_truth,
            reasoning_attempts,
            _m_legacy,
            _ml_lock,
        )
        pn = 1 - average_y
        print(f"PN value for step {i}: {pn}")
        i += 1

    return nodes, i, pn


def test_original_metrics(
    query: str,
    response: str,
    ground_truth: str,
    reasoning_attempts: int = 3,
    do_type: int = 0,
    alter_attempts: int = 5)-> Dict[str, Any]:
    
    _m_t = _new_metrics()
    _mt_lock = threading.Lock()
    original_metrics = get_original_metrics(response, ground_truth, _m_t, _mt_lock)

    # Parse the chain into nodes
    nodes = parse_nodes(response)
    pprint.pprint(nodes)

    # Intervene on each node
    i = 0
    pn_values = []
    while i < len(nodes):
        nodes, i, pn = check_and_intervene(
            query=query,
            nodes=nodes,
            i=i,
            ground_truth=ground_truth,
            reasoning_attempts=reasoning_attempts,
            do_type=do_type,
            alter_attempts=alter_attempts
        )
        # Only append pn if it's not None
        if pn is not None:
            pn_values.append(pn)
    
    # Assemble final results
    results = {
        "original_token_length": original_metrics["original_token_length"],
        "original_accuracy": original_metrics["original_accuracy"],
        "original_step_length": original_metrics["original_step_length"],
        "avg_PN(steps)": sum(pn_values)/len(pn_values) if pn_values else None,
        "max_PN(steps)": max(pn_values) if pn_values else None,
        "min_PN(steps)": min(pn_values) if pn_values else None,
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

# Math test

if __name__ == "__main__":
    print("\nRunning example...")
    example_query = "Evaluate $\\lceil{\\sqrt{20}}\\rceil^2$."
    ground_truth = "2"
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

# # Commonsense test

# if __name__ == "__main__":
#     print("\nRunning example...")
#     example_query = "What is the best way to begin going into trance?\nOptions:\n  A. religious experience\n  B. closed eyes\n  C. loss of control\n  D. sleep\n  E. hallucination\nPlease select the most appropriate answer and respond with the whole reasoning process together with the corresponding letter (A, B, C, D, or E)."
#     ground_truth = "Answer: B"
#     example_llm_response = llm_api([{"role": "user", "content": example_query}])
#     print("\nexample_llm_response:",example_llm_response)
#     results = calculate_ps_pn(
#         query=example_query,
#         response=example_llm_response,
#         ground_truth=ground_truth,
#         alter_attempts=3,
#         do_type=1,
#     )

#     print("\nFinal chain of steps:")
#     pprint.pprint(results["final_chain"])

#     metrics = {k: v for k, v in results.items() if k != "final_chain"}
#     print("\nMetrics:")
#     pprint.pprint(metrics)
    
    