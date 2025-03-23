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
        model="gpt-4",  # text-davinci-002: best, text-ada-001: lowest price
        temperature=0,  # control randomness: lowring results in less random completion (0 ~ 1.0)
        max_tokens=4000,  # max number of tokens to generate (1 ~ 4,000)
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
    # prompt_list = f"""
    # 我正在创建输入输出训练对来微调我的 GPT 模型。输入是实体名称和类别（类别为 ENTITY、NUM、UNIT），输出是合成描述及标注。
    # 关键要求：
    #
    # 每个字符必须单独编号，即使属于同一实体（例如：数字"15"需拆分为 1(3) 和 5(4)，标注为 (3,4,NUM)；单位"cm"需拆分为 c(5) 和 m(6)，标注为 (5,6,UNIT)）。
    #
    # 实体标注范围需严格覆盖字符的起止索引（如 NUM 实体"15"标注为 (3,4)）。
    #
    # 关系标注中的索引对应输入中的实体顺序（如输入中的第1个实体是 ENTITY，第2个是 NUM，第3个是 UNIT）。
    #
    # 格式应为以下形式：
    # 1.
    # 输入：实体名称1、实体类别1；实体名称2、实体类别2；实体名称3、实体类别3
    # 输出：合成描述，实体标注，关系标注
    # 关系标注解释：...
    #
    # 不要在该格式周围添加任何额外的字符，因为这会导致输出解析中断。
    # 以下是一些有用的例子，可帮助您获得正确的输出样式。
    #
    # 1)
    # 输入：“最大处、ENTITY；3*2、NUM；c㎡、UNIT”
    # 输出：”最大处大小约3*2c㎡。[最(0)大(1)处(2)大(3)小(4)约(5)3(6)*(7)2(8)c(9)㎡(10)][(0,2,ENTITY),(6,8,NUM),(9,10,UNIT)][(0,1,"ENTITY-VALUE"),(1,2,"NUM-UNIT")]“
    # 关系标注解释：(0,1,"ENTITY-VALUE")表示第0个和第1个实体之间为“实体-数量”关系，(1,2,"NUM-UNIT")表示第1个和第2个实体之间为“数量-单位”关系
    # """
    prompt_list = f"""
        请生成10个复杂输入输出训练对，输入包含至少1个实体（可跨类别重复，如多个NUM或UNIT），输出需满足：
            句子复杂：含 并列结构、介词短语、技术术语（如“在...条件下”“误差不超过”“最大耐受值为”）。
            输入输出实体一致性：输入中的每个实体必须完整出现在输出描述中，且 不允许添加任何新实体（如输入无“环境温度”，则输出不可包含该词及其数值/单位）。若输入包含多个同类实体（如两个NUM），需在描述中全部体现，但可通过 连接词（如“至”“或”）合并表达（例如“-40℃至125℃”）。
            标注严格：字符级拆分索引，实体范围精确，仅使用 ENTITY-VALUE 和 NUM-UNIT 关系。
            关系类型唯一性：每对相邻的 ENTITY → NUM → UNIT 必须生成 ENTITY-VALUE 和 NUM-UNIT 关系，禁止其他关系类型。
            场景专业：覆盖实验、工程、医学等领域的参数描述。

        格式规范（务必遵守）
        输入：实体名称1、类别1；实体名称2、类别2；实体名称3、类别3；...
        输出：多实体组合的长句，含技术性表达。
        token标注：每个字符必须单独编号，即使属于同一实体（例如：数字"15"需拆分为 1(3) 和 5(4)，标注为 (3,4,NUM)；单位"cm"需拆分为 c(5) 和 m(6)，标注为 (5,6,UNIT)）；单位"g/L"需拆分为 g(5) /(6) L(7)，标注为 (5,7,UNIT)）。
        实体标注：实体标注范围需严格覆盖字符的起止索引（如 NUM 实体"15"标注为 (3,4)）。
        关系标注：仅用 (x,y,"ENTITY-VALUE") 和 (y,z,"NUM-UNIT")，其中 x,y,z 对应输入中的实体顺序索引。
        
        生成的复杂示例演示：
        1.
        输入："下颌部、ENTITY；1x1.5、NUM；cm2、UNIT；上胸部、ENTITY；2x3、NUM；cm2、UNIT"
        输出："ID:453237#仅剩下颌部1x1.5cm2及上胸部2x3cm2，"
        token标注：[I(0)D(1):(2)4(3)5(4)3(5)2(6)3(7)7(8)#(9)仅(10)剩(11)下(12)颌(13)部(14)1(15)x(16)1(17).(18)5(19)c(20)m(21)2(22)及(23)上(24)胸(25)部(26)2(27)x(28)3(29)c(30)m(31)2(32)，(33)]
        实体标注：[[12,14,"ENTITY"],[15,19,"NUM"],[20,22,"UNIT"],[24,26,"ENTITY"],[27,29,"NUM"],[30,32,"UNIT"]]
        关系标注：[[1,2,"NUM-UNIT"],[4,5,"NUM-UNIT"],[0,1,"ENTITY-VALUE"],[3,4,"ENTITY-VALUE"]]
        
        2.
        输入："血氧饱和度I度、ENTITY；89、NUM；%、UNIT"
        输出："ID:453306#昨晚凌晨血氧饱和度I度下降至89%，"
        token标注：[I(0)D(1):(2)4(3)5(4)3(5)3(6)0(7)6(8)#(9)昨(10)晚(11)凌(12)晨(13)血(14)氧(15)饱(16)和(17)度(18)I(19)度(20)下(21)降(22)至(23)8(24)9(25)%(26)，(27)]
        实体标注：[[14,20,"ENTITY"],[24,25,"NUM"],[26,26,"UNIT"]]
        关系标注：[[1,2,"NUM-UNIT"],[0,1,"ENTITY-VALUE"]]
        
        3.
        输入："白细胞、ENTITY；155.00、NUM；/ul、UNIT"
        输出："ID:454202#2016-2-26临检:尿常规分析检验报告：白细胞155.00/ul↑。"
        token标注：[I(0)D(1):(2)4(3)5(4)4(5)2(6)0(7)2(8)#(9)2(10)0(11)1(12)6(13)-(14)2(15)-(16)2(17)6(18)临(19)检(20):(21)尿(22)常(23)规(24)分(25)析(26)检(27)验(28)报(29)告(30)：(31)白(32)细(33)胞(34)1(35)5(36)5(37).(38)0(39)0(40)/(41)u(42)l(43)↑(44)。(45)]
        实体标注：[[32,34,"ENTITY"],[35,40,"NUM"],[41,43,"UNIT"]]
        关系标注：[[1,2,"NUM-UNIT"],[0,1,"ENTITY-VALUE"]]
        """
    run(prompt_list)
