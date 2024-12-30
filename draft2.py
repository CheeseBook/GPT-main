import json

def add_ids_to_json(file_path):
    # 读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化 ID 计数器
    current_id = 0

    # 为 train, dev, test 数据添加 id
    for split in ['train', 'dev', 'test']:
        for item in data.get(split, []):
            item['id'] = current_id
            current_id += 1

    # 将更新后的数据写回 JSON 文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 使用示例
file_path = 'mqi_dataset/cqi.json'  # 将 'data.json' 替换为你的 JSON 文件路径
add_ids_to_json(file_path)
