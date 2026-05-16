import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 加载 jsonl 文件为字典，key 为 index，value 为整行数据
def load_jsonl_by_fields(filepath, keys=["index"]):
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            try:
                combined_key = str(item[keys[0]])
                data[combined_key] = item
            except KeyError:
                continue  # 如果有某个 key 缺失，就跳过这行
    return data

# 文件路径（换成你自己的）
file1 = "data/COMMONSENCEQA/qwen_commonsenseqa_original_pn_np_indexed.jsonl"
file2 = "data/COMMONSENCEQA/qwen_commonsenseqa_after_pn_np_indexed.jsonl"

# 加载两个文件
data1 = load_jsonl_by_fields(file1)
data2 = load_jsonl_by_fields(file2)

common_keys = set(data1.keys()) & set(data2.keys())

# 将集合转换为列表
common_keys = list(common_keys)
# 对键按照正整数排序
common_keys = sorted(common_keys, key=lambda x: int(x))

# 筛选出第二个文件中 avg_PN(steps) 大于等于第一个文件且都非零的前 30 个键
valid_keys = []
for t in common_keys:
    v1 = data1[t].get("metrics", {}).get("avg_PN(steps)")
    v2 = data2[t].get("metrics", {}).get("avg_PN(steps)")
    if v1 is not None and v2 is not None and v1 != 0 and v2 != 0 and v2 > v1:
        valid_keys.append(t)
    if len(valid_keys) == 30:
        break

# 提取对比数据
x_labels = []
pn1 = []
pn2 = []

for t in valid_keys:
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
    