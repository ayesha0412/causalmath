import pprint

from base_model import llm_predict_function, rollout_function
from equivalent_ans import is_equivalent_answer

def calculate_ps_pn(llm_response, ground_truth, rollout_function, llm_predict_function, threshold=0.8, max_attempts=3):
    """
    Calculate PS(chain) and PN(step) for a reasoning chain.
    """
    # split the response into nodes
    nodes = llm_response.split('### Step')
    pprint.pprint(nodes)

    # calculate PS(chain)
    final_answer = nodes[-1]
    # print("final_answer", final_answer)
    ps = 1 if is_equivalent_answer(final_answer,ground_truth) else 0

    if ps < threshold:
        return "not sufficient"
    
    # calculate PN(step)
    for i in range(len(nodes)):
        k = i + 1 # 从第一个推理步骤开始而不是从问题开始
        context = nodes[:k]  # context before the node；not contain the node[i]
        print("上下文", context)
        y_values = [] 
        successful_replacement = False

        # intervene nodes for 3 times (setting in the rollout function)
        new_node_variants = rollout_function(context)
        print("生成的三个替换节点分别是", new_node_variants) # 3 node list
        for new_node in new_node_variants:
            new_chain = context + [new_node]  
            final_answer = llm_predict_function(new_chain)[-1]  

            # evaluate Y
            if not is_equivalent_answer(final_answer,ground_truth):
                y_values.append(0)  # necessary
            else:
                y_values.append(1)  # not necessary

        pn = 1 - (sum(y_values) / len(y_values)) if y_values else 0
        temp = nodes[k]
        # if PN(step) < threshold, remove the node and re-rollout to find a necessary node
        if pn < threshold:
            for _ in range(max_attempts):
                nodes[k] = rollout_function(context)[0]  # use the first variant
                new_chain = nodes[:k + 1]  
                after_chain = llm_predict_function(new_chain)
                final_answer = after_chain[-1]
                if is_equivalent_answer(final_answer, ground_truth):
                    nodes[k+1:] = after_chain
                    successful_replacement = True
                    break

        # if after max_attempts PN is still low, retain the original node
        if not successful_replacement:
            nodes[k] = temp

    return {"PS(chain)": ps, "final_chain": nodes}


# llm_response = """Given the quadratic polynomials \\( P(x) \\) and \\( Q(x) \\) with leading coefficients 2 and -2, respectively, and both passing through the points \\((16, 54)\\) and \\((20, 53)\\), we need to find \\( P(0) + Q(0) \\).\n\n### Step 1: Express the polynomials in general form\nThe general form of a quadratic polynomial is:\n\\[ P(x) = 2x^2 + bx + c \\]\n\\[ Q(x) = -2x^2 + dx + e \\]\n\n### Step 2: Use the given points to set up equations\nSince both polynomials pass through \\((16, 54)\\) and \\((20, 53)\\), we can substitute these points into the equations for \\( P(x) \\) and \\( Q(x) \\).\n\nFor \\( P(x) \\):\n\\[ P(16) = 2(16)^2 + 16b + c = 54 \\]\n\\[ 512 + 16b + c = 54 \\]\n\\[ 16b + c = 54 - 512 \\]\n\\[ 16b + c = -458 \\quad \\text{(Equation 1)} \\]\n\n\\[ P(20) = 2(20)^2 + 20b + c = 53 \\]\n\\[ 800 + 20b + c = 53 \\]\n\\[ 20b + c = 53 - 800 \\]\n\\[ 20b + c = -747 \\quad \\text{(Equation 2)} \\]\n\nFor \\( Q(x) \\):\n\\[ Q(16) = -2(16)^2 + 16d + e = 54 \\]\n\\[ -512 + 16d + e = 54 \\]\n\\[ 16d + e = 54 + 512 \\]\n\\[ 16d + e = 566 \\quad \\text{(Equation 3)} \\]\n\n\\[ Q(20) = -2(20)^2 + 20d + e = 53 \\]\n\\[ -800 + 20d + e = 53 \\]\n\\[ 20d + e = 53 + 800 \\]\n\\[ 20d + e = 853 \\quad \\text{(Equation 4)} \\]\n\n### Step 3: Solve the system of equations for \\( P(x) \\)\nSubtract Equation 1 from Equation 2:\n\\[ (20b + c) - (16b + c) = -747 - (-458) \\]\n\\[ 4b = -289 \\]\n\\[ b = -\\frac{289}{4} \\]\n\nSubstitute \\( b = -\\frac{289}{4} \\) into Equation 1:\n\\[ 16\\left(-\\frac{289}{4}\\right) + c = -458 \\]\n\\[ -1156 + c = -458 \\]\n\\[ c = 698 \\]\n\n### Step 4: Solve the system of equations for \\( Q(x) \\)\nSubtract Equation 3 from Equation 4:\n\\[ (20d + e) - (16d + e) = 853 - 566 \\]\n\\[ 4d = 287 \\]\n\\[ d = \\frac{287}{4} \\]\n\nSubstitute \\( d = \\frac{287}{4} \\) into Equation 3:\n\\[ 16\\left(\\frac{287}{4}\\right) + e = 566 \\]\n\\[ 1148 + e = 566 \\]\n\\[ e = -582 \\]\n\n### Step 5: Find \\( P(0) \\) and \\( Q(0) \\)\n\\[ P(0) = 2(0)^2 + b(0) + c = c = 698 \\]\n\\[ Q(0) = -2(0)^2 + d(0) + e = e = -582 \\]\n\n### Step 6: Calculate \\( P(0) + Q(0) \\)\n\\[ P(0) + Q(0) = 698 + (-582) = 116 \\]\n\n### Final Answer\n\\[ \\boxed{116} \\]"""
llm_response = """1 + 2 + 5 * 3 + 4 / 2 =?
### Step 1: 1+2= 3
### Step 2: 5*3=15
### Step 3: 4/1=4
### Step 4: 4/2=2
### Step 5: 1+2+15+2=20
"""

ground_truth = "20"


results = calculate_ps_pn(
    llm_response,
    ground_truth,
    rollout_function,
    llm_predict_function
)

# print(f"PS(chain): {results['PS(chain)']}")
print(f"Final Chain: {results['final_chain']}")