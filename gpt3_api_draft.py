import random
import json
from openai import OpenAI

api_key = "sk-S18T7dPfh7at8EpHFfF2265800E34897Be8d5bC265C4C79f"
base_url = "https://api.gptapi.us/v1/chat/completions"
client = OpenAI(api_key=api_key, base_url=base_url)


class SentenceAugmentor:
    def __init__(self, model="gpt-3.5-turbo", temperature=0.7, max_tokens=512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt):
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for data augmentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return completion.choices[0].message.content

    def augment_sample(self, sample):
        # Construct the prompt
        prompt = self.create_prompt(sample)
        response = self.generate(prompt)
        return self.parse_response(sample, response)

    def create_prompt(self, sample):
        tokens = " ".join(sample["tokens"])
        ner_info = json.dumps(sample["ner"])
        rel_info = json.dumps(sample["rel"])
        prompt = f"""
        给定以下文本和标注信息：
        - 文本: {tokens}
        - 实体信息: {ner_info}
        - 关系信息: {rel_info}

        请基于以下规则生成增强数据：
        1. 替换文本中的实体，但保持语义合理性。例如，替换'体温'为'身高'，并调整相关数量和单位。
        2. 确保生成的实体类型、数量和单位合理。例如，'身高'的单位应为'厘米'，而不是'摄氏度'。
        3. 更新相应的实体和关系标注信息，包括索引范围和类型。
        4. 文本中涉及到的实体都需要进行替换，并保持合理性。

        直接返回增强后的数据，格式为 JSON，包含以下字段：'tokens', 'ner', 'rel'。
        """
        return prompt

    def parse_response(self, original_sample, response):
        # Convert response JSON string back to Python dict
        new_sample = json.loads(response)
        return new_sample


def augment_dataset(dataset, percentage=20):
    augmentor = SentenceAugmentor()
    train_data = dataset["train"]
    selected_samples = random.sample(train_data, int(len(train_data) * (percentage / 100)))
    augmented_samples = []

    for sample in selected_samples:
        augmented_sample = augmentor.augment_sample(sample)
        augmented_samples.append(augmented_sample)

    train_data.extend(augmented_samples)
    return dataset


if __name__ == '__main__':
    # Load dataset
    with open("mqi_dataset/test.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Perform augmentation
    augmented_dataset = augment_dataset(dataset, percentage=100)

    # Save augmented dataset
    with open("augmented_dataset.json", "w", encoding="utf-8") as f:
        json.dump(augmented_dataset, f, ensure_ascii=False, indent=4)
