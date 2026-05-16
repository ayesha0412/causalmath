import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random


# 加载 jsonl 文件为字典，key 为 type，value 为整行数据
def load_jsonl_by_fields(filepath, keys=["exam", "index"]):
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            try:
                combined_key = "_".join(str(item[k]) for k in keys)
                data[combined_key] = item
            except KeyError:
                continue  # 如果有某个key缺失，就跳过这行
    return data

# 文件路径（换成你自己的）
file1 = "data/COMMONSENCEQA/qwen_commonsenseqa_original_pn_np_indexed.jsonl"
file2 = "data/COMMONSENCEQA/qwen_commonsenseqa_after_pn_np_indexed.jsonl"

# 加载两个文件
data1 = load_jsonl_by_fields(file1)
data2 = load_jsonl_by_fields(file2)


def parse_key(k):
    # 处理如 'AIME_I_0' → ('AIME_I', 0)
    parts = k.split("_")
    exam = "_".join(parts[:-1])  # "AIME_I" or "AIME_II"
    index = int(parts[-1])  # 题号
    return (exam, index)


common_keys = set(data1.keys()) & set(data2.keys())

random.seed(42)
# 随机选择 30 个键
if len(common_keys) > 30:
    common_keys = random.sample(common_keys, 30)

# 对随机选择的键进行排序
common_keys = sorted(common_keys, key=parse_key)

# 提取对比数据
x_labels = []
pn1 = []
pn2 = []

for t in common_keys:
    v1 = data1[t].get("metrics", {}).get("avg_PN(steps)", None)
    v2 = data2[t].get("metrics", {}).get("avg_PN(steps)", None)
    if v1 is not None and v2 is not None:
        x_labels.append(t)
        pn1.append(v1)
        pn2.append(v2)

# 准备数据用于 seaborn
data = {
    'Question': x_labels * 2,
    'Average PN (Steps)': pn1 + pn2,
    'Stage': ['Before CoT - PNS'] * len(pn1) + ['After CoT - PNS'] * len(pn2)
}
df = pd.DataFrame(data)

# 设置 seaborn 样式
sns.set(style="whitegrid")

# 画图：双柱状图
plt.figure(figsize=(14, 6))
sns.barplot(x='Question', y='Average PN (Steps)', hue='Stage', data=df, palette=["#1f77b4", "#ff7f0e"])

# 设置图形标签和标题
plt.xticks(rotation=45, ha='right')
plt.ylabel("Average PN (Steps)", fontsize=12)
plt.title("PN Comparison on COMMONSENCEQA (Qwen - 72B, Before/After)", fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()

# 保存图形
plt.savefig("qwen_common_pn.png", dpi=300)
    