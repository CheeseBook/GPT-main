import json


def load_json(file_path):
    """从文件加载 JSON 数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_data_by_ids(data, target_ids):
    """从 JSON 数据集中提取多个 ID 的数据"""
    results = []
    i = 1

    for split in ["train", "dev", "test"]:
        for entry in data.get(split, []):
            if entry["id"] in target_ids:
                tokens = entry["tokens"]
                ner = entry["ner"]
                rel = entry["rel"]

                # 1. 生成原始文本（去掉空格）
                raw_text = "".join(tokens)

                # 2. 生成 token 索引列表
                token_str = "[" + "".join(f"{t}({i})" for i, t in enumerate(tokens)) + "]"

                # 3. 生成 NER 及关系信息
                ner_str = json.dumps(ner, ensure_ascii=False, separators=(',', ':'))  # 确保无多余空格
                rel_str = json.dumps(rel, ensure_ascii=False, separators=(',', ':'))

                # 4. 生成实体信息（严格去除多余空格）
                entity_info = "；".join(f"{''.join(tokens[start:end + 1])}、{label}" for start, end, label in ner)

                # 5. 格式化输出（确保无多余空格）
                result = f'{i}.\n输入："{entity_info}"\n输出："{raw_text}"\n{token_str}\n{ner_str}\n{rel_str}'
                results.append(result)
                i+=1

    return "\n\n".join(results) if results else "未找到指定 ID 的数据"


# 读取 JSON 文件
file_path = r"D:\研究生论文项目代码复现\GPT-RE-main\mqi_dataset\sqi.json"  # 替换为你的 JSON 文件路径
json_data = load_json(file_path)

# 目标 ID 列表
target_ids = [0,1,2,3]  # 修改为你要查询的多个 ID

# 获取结果并输出
result = extract_data_by_ids(json_data, target_ids)
print(result)
