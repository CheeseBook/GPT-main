# -*- coding: utf-8 -*-
from openai import OpenAI
import os

api_key="sk-S18T7dPfh7at8EpHFfF2265800E34897Be8d5bC265C4C79f"
base_url = "https://api.gptapi.us/v1/chat/completions"
os.environ["OPENAI_API_KEY"]=api_key
os.environ["OPENAI_API_BASE"]= base_url

client = OpenAI(api_key=api_key, base_url= base_url)

class Demo(object):
    def __init__(self, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, best_of, logprobs):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.best_of = best_of
        self.logprobs = logprobs

    # prompt_list: List[str]
    def get_multiple_sample(self, prompt):
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to generate synthetic data."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            logprobs=self.logprobs
        )

        results = completion.choices[0].message.content
        probs = completion.choices[0].logprobs
        return results, probs


def run(prompt_list):
    demo = Demo(
        model="gpt-4o",  # text-davinci-002: best, text-ada-001: lowest price
        temperature=0,  # control randomness: lowring results in less random completion (0 ~ 1.0)
        max_tokens=1000,  # max number of tokens to generate (1 ~ 4,000)
        top_p=1,  # control diversity (0 ~ 1.0)
        frequency_penalty=0,  # how to penalize new tokens based on their existing frequency (0 ~ 2.0)
        presence_penalty=0,  # 这个是对于词是否已经出现过的惩罚，文档上说这个值调高可以增大谈论新topic的概率 (0 ~ 2.0)
        best_of=3,  # 这个是说从多少个里选最好的，如果这里是10，就会生成10个然后选最好的，但是这样会更贵(1 ~ 20)
        logprobs=True
    )
    results, probs = demo.get_multiple_sample(prompt_list)
    # print(results[0])
    # print(probs[0])
    print(results)


if __name__ == '__main__':
    prompt_list = f"""
    我正在创建输入输出训练对来微调我的 gpt 模型。我希望输入是几个实体名称和实体类别，输出是合成描述。类别应该是：ENTITY、NUM、UNIT。更重要的是，类别应该属于烧伤科主题：
    每个示例的编号后还应说明主题领域。格式应为以下形式：
    1. 主题领域
    输入：实体名称1、实体类别1；实体名称2、实体类别2；实体名称3、实体类别3
    输出：合成描述，实体标注
    不要在该格式周围添加任何额外的字符，因为这会导致输出解析中断。
    以下是一些有用的例子，可帮助您获得正确的输出样式。
    
    1) 烧伤科
    输入：“心率、ENTITY；112、NUM；次/分、UNIT”
    输出：”李明患者的心率是112次/分。[李(0)明(1)患(2)者(3)的(4)心(5)率(6)是(7)1(8)1(9)2(10)次(11)/(12)分(13)][(5,6,ENTITY),(8,10,NUM),(11,13,UNIT)]“
    """
    run(prompt_list)
