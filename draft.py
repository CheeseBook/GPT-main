import json

# 假设 JSON 文件的路径是 data.json
file_path = 'mqi_dataset/processed_cqi_data.json'

# 加载 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 根据 id 和索引范围提取文本
def extract_text_by_id_and_indices(data, target_id, start, end):
    for item in data:
        if item["id"] == target_id:
            sentence = item["sentence"]
            return sentence[start:end + 1]
    return None

# 主程序
def main():
    data = load_json(file_path)

    # 示例提取文本
    target_id = 1
    start = 29
    end = 48
    extracted_text = extract_text_by_id_and_indices(data, target_id, start, end)
    print(f"提取的文本: {extracted_text}")

if __name__ == "__main__":
    main()
