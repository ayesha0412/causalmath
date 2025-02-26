import openai
from typing import List, Dict

openai.api_key = "sk-O8VHM4wmKqyidqqYF7201937F6Ea4fC0Ac1713C99dC9A1B6"
openai.base_url = 'https://api.shubiaobiao.cn/v1/'

prompt_template = """## In-Context Learning Prompt (Sufficient and Necessary Reasoning)

### Instructions

When solving the following questions, your reasoning should:
- **Be Accurate:** Ensure your chain of thought leads to the correct answer without skipping any necessary logical steps.
- **Be Efficient:** Avoid unnecessary or redundant steps. Each step should be necessary to progress toward the solution.
- **Aim for Sufficient and Necessary Reasoning:** Only include steps that are both sufficient to reach the correct answer and necessary to avoid gaps or confusion. If a step can be removed without affecting correctness, remove it.
- **Notice the Pattern:** In the following examples, compare the original, verbose solution with the optimized solution. Learn to identify and eliminate redundant reasoning steps while preserving logical soundness.

---

### Example 1:

**Question:**  
Altitudes \\(\\overline\{{AD}}\\) and \\(\\overline{{BE}}\\) of \\(\\triangle ABC\\) intersect at \\(H\\). If \\(\\angle BAC = 54^\\circ\\) and \\(\\angle ABC = 52^\\circ\\), what is \\(\\angle AHB\\)?

**Original, Verbose CoT:**  
To find \\(\\angle AHB\\), we can leverage properties involving altitudes in triangles and the supplementary nature of certain angle pairs.

First, calculate \\(\\angle ACB\\):
\\[
\\angle ACB = 180^\\circ - \\angle BAC - \\angle ABC = 180^\\circ - 54^\\circ - 52^\\circ = 74^\\circ
\\]

In \\(\\triangle ABC\\), the altitudes intersect at the orthocenter \\(H\\), and \\(\\angle AHD\\) and \\(\\angle BHE\\) are right angles.

Consider quadrilateral \\(AHBE\\), which is a cyclic quadrilateral because opposite angles sum to \\(180^\\circ\\).

Using this property:
\\[
\\angle AHB = 180^\\circ - \\angle AEB = 180^\\circ - \\angle ACB = 180^\\circ - 74^\\circ = 106^\\circ
\\]

Final Answer:
\\[
\\boxed{{106^\\circ}}
\\]

**Optimized, Sufficient and Necessary CoT:**  
\\[
\\angle ACB = 180^\\circ - 54^\\circ - 52^\\circ = 74^\\circ
\\]

\\[
\\angle AHB = 180^\\circ - \\angle ACB = 180^\\circ - 74^\\circ = 106^\\circ
\\]

---

### Example 2:

**Question:**  
Simplify: \\(\\frac{{\\sqrt{{2.5^2 - 0.7^2}}}}{{2.7 - 2.5}}\\).

**Original, Verbose CoT:**  
Calculate the denominator:
\\[
2.7 - 2.5 = 0.2
\\]

Simplify the numerator:
\\[
2.5^2 = 6.25
\\]

\\[
0.7^2 = 0.49
\\]

\\[
6.25 - 0.49 = 5.76
\\]

Take the square root:
\\[
\\sqrt{{5.76}} = 2.4
\\]

Substitute into the expression:
\\[
\\frac{{2.4}}{{0.2}} = 12
\\]

Final Answer:
\\[
\\boxed{{12}}
\\]

**Optimized, Sufficient and Necessary CoT:**  
\\[
2.5^2 - 0.7^2 = 6.25 - 0.49 = 5.76
\\]

\\[
\\sqrt{{5.76}} = 2.4
\\]

\\[
2.7 - 2.5 = 0.2
\\]

\\[
\\frac{{2.4}}{{0.2}} = 12
\\]

### Example 3:

**Question:**
Simplify: \( \\frac{\sqrt{3.6^2 - 1.1^2}}{4.1 - 3.6} \\)

**Original, Verbose CoT:**  
To simplify the expression, we start by calculating both the numerator and the denominator.

First, calculate the denominator:
\[
4.1 - 3.6 = 0.5
\\]

Next, simplify the numerator, which is \( \sqrt{3.6^2 - 1.1^2} \\).

First, compute \( 3.6^2 \\):
\[
3.6^2 = 12.96
\\]

Now compute \( 1.1^2 \\):
\[
1.1^2 = 1.21
\\]

Subtract these values:
\[
3.6^2 - 1.1^2 = 12.96 - 1.21 = 11.75
\\]

Taking the square root of the result:
\[
\sqrt{11.75} \approx 3.43
\\]

Now substitute back into the original expression:
\[
\\frac{\sqrt{3.6^2 - 1.1^2}}{4.1 - 3.6} = \\frac{3.43}{0.5}
\\]

Finally, perform the division:
\[
\\frac{3.43}{0.5} = 6.86
\\]

Therefore, the simplified expression is \( \\boxed{6.86} \\).

**Optimized, Sufficient and Necessary CoT:**  
\[
3.6^2 = 12.96
\\]

\[
1.1^2 = 1.21
\\]

\[
12.96 - 1.21 = 11.75
\\]

\[
\sqrt{11.75} \approx 3.43
\\]

\[
4.1 - 3.6 = 0.5
\\]

\[
\\frac{3.43}{0.5} = 6.86
\\]
---

### Instructions Before Final Task

- **Notice** how the optimized solutions eliminate redundant reasoning steps while retaining the necessary calculations to reach the correct answer.
- **Learn** that your goal is to produce reasoning chains that are **sufficient and necessary**:  
  - Every step you include should **either contribute to the final answer or be necessary to ensure the logic is clear**.
  - If a step can **be skipped without compromising correctness or clarity, omit it**.
- **Output concise, accurate, and logically sound reasoning for the following problem directly, with no redundant analysis.**

"""


def query_api(messages: List[Dict[str, str]], model: str = "gpt-4o", temperature: float = 0.0):
    try:
        optimized_prompt = prompt_template
        
        messages_with_prompt = [{"role": "system", "content": optimized_prompt}] + messages
        
        response = openai.chat.completions.create(
            model=model,
            messages=messages_with_prompt,
            temperature=temperature
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return str(e)

# 示例调用
test_question ="Let $f(x)=|x-p|+|x-15|+|x-p-15|,$ where $0 < p < 15.$ Determine the minimum value taken by $f(x)$ for $x$ in the interval $p \\leq x\\leq15.$"
messages = [
    {"role": "user", "content": "Now Solve This" + test_question + "Your Simplified and Optimized Answer:"}
]

answer = query_api(messages)
print("Model's Answer:", answer)