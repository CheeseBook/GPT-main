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
                {"role": "system", "content": "You are a helpful assistant."},
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
    prompt_list = "现你要依据标签来生成文本，每个文本包含三类标签<ENTITY><NUM><UNIT>，每个标签的冒号后面代表你要生成的文本长度，比如<ENTITY:3>代表你要生成长度为3的实体。<NUM:2>代表你要生成长度为2的数字，<UNIT:2>代表你要生成长度为2的单位。要求生成的结果与上下文的合理性，保持通顺，实体、数量、单位要合理。生成的结果替换原来标签的位置，然后你返回生成后的文本：ID:453237#体格检查:<ENTITY:2><NUM:5><UNIT:2>,<ENTITY:2><NUM:2><UNIT:3>,<ENTITY:4><NUM:2><UNIT:3>,<ENTITY:2><NUM:6><UNIT:4>,<ENTITY:2><NUM:4><UNIT:2>,一般状况:发育：正常,体型：适中,营养：营养良好,口渴：无,表情：自然,体位：自主,神志：清醒,查体：合作,周围血管：桡动脉搏动：正常,足背动脉搏动：正常,微血管充盈：毛细血管充盈良好,皮肤粘膜(烧伤创面见专科情况)：色泽：正常,皮疹：无,皮下出血：无,温度：正常,湿度：正常,弹性：弹性正常,水肿：无,浅表淋巴结：无,肿大。"
    run(prompt_list)
