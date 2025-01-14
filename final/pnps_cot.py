import pprint

from base_model import llm_predict_function, llm_predict_gpt35
from equivalent_ans import is_equivalent_answer

def calculate_ps_pn(llm_response, ground_truth, llm_predict_function, threshold=0.8, max_attempts=3):
    """
    计算推理链的 PS(chain) 和 PN(step)。
    """

    nodes = llm_response.split('### Step')
    pprint.pprint(nodes)

    # PS(chain)
    final_answer = nodes[-1] if nodes else ""
    ps = 1 if is_equivalent_answer(final_answer, ground_truth) else 0

    if ps < threshold:
        return "not sufficient"
    
    i = 1  
    while i < len(nodes):  
        context = nodes[:i]  # 当前节点前面的上下文，不包含 nodes[i]
        print("上下文", context)
        y_values = [] 
        successful_replacement = False

        # 对节点进行 3 次干预
        for m in range(3):
            print('对于第{}节点,第{}次替换'.format(i, m+1))
            after_chain = llm_predict_function(context)
            new_node = after_chain[0]
            print("生成的替换节点是", new_node) 
            final_answer = after_chain[-1]  

            # 评估 Y
            if not is_equivalent_answer(final_answer, ground_truth):
                y_values.append(0)  # 必要
            else:
                y_values.append(1)  # 不必要

        pn = 1 - (sum(y_values) / len(y_values)) if len(y_values) > 0 else 0
        print('第{}节点的pn是{}'.format(i, pn))
        temp = nodes[i]
        
        # 如果 PN(step) < threshold，则需要重新替换该节点
        if pn < threshold:
            print('节点{}的pn小于阈值,需要重新替换'.format(nodes[i]))
            for _ in range(max_attempts):
                after_chain = llm_predict_function(context)
                final_answer = after_chain[-1]  
                if is_equivalent_answer(final_answer, ground_truth):
                    nodes[i:] = after_chain
                    successful_replacement = True
                    print('重新替换后的节点是', nodes[i])
                    break

        # 如果替换失败，需要恢复原节点
        if not successful_replacement:
            nodes[i] = temp

        # 根据当前节点替换后的情况，更新 i
        if successful_replacement:
            # 如果替换成功，继续从当前节点后一个节点开始
            i = i + 1
        else:
            # 如果替换失败，直接继续遍历下一个节点
            i = i + 1

    return {"PS(chain)": ps, "final_chain": nodes}


# llm_response = """Given the quadratic polynomials \\( P(x) \\) and \\( Q(x) \\) with leading coefficients 2 and -2, respectively, and both passing through the points \\((16, 54)\\) and \\((20, 53)\\), we need to find \\( P(0) + Q(0) \\).\n\n### Step 1: Express the polynomials in general form\nThe general form of a quadratic polynomial is:\n\\[ P(x) = 2x^2 + bx + c \\]\n\\[ Q(x) = -2x^2 + dx + e \\]\n\n### Step 2: Use the given points to set up equations\nSince both polynomials pass through \\((16, 54)\\) and \\((20, 53)\\), we can substitute these points into the equations for \\( P(x) \\) and \\( Q(x) \\).\n\nFor \\( P(x) \\):\n\\[ P(16) = 2(16)^2 + 16b + c = 54 \\]\n\\[ 512 + 16b + c = 54 \\]\n\\[ 16b + c = 54 - 512 \\]\n\\[ 16b + c = -458 \\quad \\text{(Equation 1)} \\]\n\n\\[ P(20) = 2(20)^2 + 20b + c = 53 \\]\n\\[ 800 + 20b + c = 53 \\]\n\\[ 20b + c = 53 - 800 \\]\n\\[ 20b + c = -747 \\quad \\text{(Equation 2)} \\]\n\nFor \\( Q(x) \\):\n\\[ Q(16) = -2(16)^2 + 16d + e = 54 \\]\n\\[ -512 + 16d + e = 54 \\]\n\\[ 16d + e = 54 + 512 \\]\n\\[ 16d + e = 566 \\quad \\text{(Equation 3)} \\]\n\n\\[ Q(20) = -2(20)^2 + 20d + e = 53 \\]\n\\[ -800 + 20d + e = 53 \\]\n\\[ 20d + e = 53 + 800 \\]\n\\[ 20d + e = 853 \\quad \\text{(Equation 4)} \\]\n\n### Step 3: Solve the system of equations for \\( P(x) \\)\nSubtract Equation 1 from Equation 2:\n\\[ (20b + c) - (16b + c) = -747 - (-458) \\]\n\\[ 4b = -289 \\]\n\\[ b = -\\frac{289}{4} \\]\n\nSubstitute \\( b = -\\frac{289}{4} \\) into Equation 1:\n\\[ 16\\left(-\\frac{289}{4}\\right) + c = -458 \\]\n\\[ -1156 + c = -458 \\]\n\\[ c = 698 \\]\n\n### Step 4: Solve the system of equations for \\( Q(x) \\)\nSubtract Equation 3 from Equation 4:\n\\[ (20d + e) - (16d + e) = 853 - 566 \\]\n\\[ 4d = 287 \\]\n\\[ d = \\frac{287}{4} \\]\n\nSubstitute \\( d = \\frac{287}{4} \\) into Equation 3:\n\\[ 16\\left(\\frac{287}{4}\\right) + e = 566 \\]\n\\[ 1148 + e = 566 \\]\n\\[ e = -582 \\]\n\n### Step 5: Find \\( P(0) \\) and \\( Q(0) \\)\n\\[ P(0) = 2(0)^2 + b(0) + c = c = 698 \\]\n\\[ Q(0) = -2(0)^2 + d(0) + e = e = -582 \\]\n\n### Step 6: Calculate \\( P(0) + Q(0) \\)\n\\[ P(0) + Q(0) = 698 + (-582) = 116 \\]\n\n### Final Answer\n\\[ \\boxed{116} \\]"""
llm_response = """1 + 2 + 5 * 3 + 4 / 2 =?
### Step 1: 1+2=3
### Step 2: 5*3=12
### Step 3: 4/1=4
### Step 4: 4/2=2
### Step 5: 1+2+15+2=20
"""

ground_truth = "20"


results = calculate_ps_pn(
    llm_response,
    ground_truth,
    llm_predict_gpt35
)

print(f"Final Chain: {results['final_chain']}")

# 强制设置值
# prompt跳过，可能信息泄露，可能常识推理的效果

# 不同数据集/不同base model/不同intervene的方案/用不同model帮助完成推理
# qwq，推理过程的冗余
['1 + 2 + 5 * 3 + 4 / 2 =?\n',
 'Using the order of operations (PEMDAS), we need to solve the multiplication and division before addition and subtraction. ', 
 '1 + 2 + 5 * 3 + 4 / 2 = 1 + 2 + 15 + 2', 
 'Now we can simply add the numbers together to get the final answer: ', 
 '20.']