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
        输出：
            合成描述：多实体组合的长句，含技术性表达。
            实体标注：每个字符必须单独编号，即使属于同一实体（例如：数字"15"需拆分为 1(3) 和 5(4)，标注为 (3,4,NUM)；单位"cm"需拆分为 c(5) 和 m(6)，标注为 (5,6,UNIT)）；单位"g/L"需拆分为 g(5) /(6) L(7)，标注为 (5,7,UNIT)）。
            关系标注：仅用 (x,y,"ENTITY-VALUE") 和 (y,z,"NUM-UNIT")，其中 x,y,z 对应输入中的实体顺序索引。
        
        生成的复杂示例演示：
        1.
        输入："下颌部、ENTITY；1x1.5、NUM；cm2、UNIT；上胸部、ENTITY；2x3、NUM；cm2、UNIT"
        输出："ID:453237#仅剩下颌部1x1.5cm2及上胸部2x3cm2，"
        [I(0)D(1):(2)4(3)5(4)3(5)2(6)3(7)7(8)#(9)仅(10)剩(11)下(12)颌(13)部(14)1(15)x(16)1(17).(18)5(19)c(20)m(21)2(22)及(23)上(24)胸(25)部(26)2(27)x(28)3(29)c(30)m(31)2(32)，(33)]
        [[12,14,"ENTITY"],[15,19,"NUM"],[20,22,"UNIT"],[24,26,"ENTITY"],[27,29,"NUM"],[30,32,"UNIT"]]
        [[1,2,"NUM-UNIT"],[4,5,"NUM-UNIT"],[0,1,"ENTITY-VALUE"],[3,4,"ENTITY-VALUE"]]
        
        2.
        输入："血氧饱和度I度、ENTITY；89、NUM；%、UNIT"
        输出："ID:453306#昨晚凌晨血氧饱和度I度下降至89%，"
        [I(0)D(1):(2)4(3)5(4)3(5)3(6)0(7)6(8)#(9)昨(10)晚(11)凌(12)晨(13)血(14)氧(15)饱(16)和(17)度(18)I(19)度(20)下(21)降(22)至(23)8(24)9(25)%(26)，(27)]
        [[14,20,"ENTITY"],[24,25,"NUM"],[26,26,"UNIT"]]
        [[1,2,"NUM-UNIT"],[0,1,"ENTITY-VALUE"]]
        
        3.
        输入："白细胞、ENTITY；155.00、NUM；/ul、UNIT"
        输出："ID:454202#2016-2-26临检:尿常规分析检验报告：白细胞155.00/ul↑。"
        [I(0)D(1):(2)4(3)5(4)4(5)2(6)0(7)2(8)#(9)2(10)0(11)1(12)6(13)-(14)2(15)-(16)2(17)6(18)临(19)检(20):(21)尿(22)常(23)规(24)分(25)析(26)检(27)验(28)报(29)告(30)：(31)白(32)细(33)胞(34)1(35)5(36)5(37).(38)0(39)0(40)/(41)u(42)l(43)↑(44)。(45)]
        [[32,34,"ENTITY"],[35,40,"NUM"],[41,43,"UNIT"]]
        [[1,2,"NUM-UNIT"],[0,1,"ENTITY-VALUE"]]
        
        4.
        输入："“B”型红细胞悬液、ENTITY；2、NUM；单位、UNIT；血浆、ENTITY；200、NUM；ml、UNIT"
        输出："ID:454278#患者昨日术后予输“B”型红细胞悬液2单位及血浆200ml，"
        [I(0)D(1):(2)4(3)5(4)4(5)2(6)7(7)8(8)#(9)患(10)者(11)昨(12)日(13)术(14)后(15)予(16)输(17)“(18)B(19)”(20)型(21)红(22)细(23)胞(24)悬(25)液(26)2(27)单(28)位(29)及(30)血(31)浆(32)2(33)0(34)0(35)m(36)l(37)，(38)]
        [[18,26,"ENTITY"],[27,27,"NUM"],[28,29,"UNIT"],[31,32,"ENTITY"],[33,35,"NUM"],[36,37,"UNIT"]]
        [[1,2,"NUM-UNIT"],[4,5,"NUM-UNIT"],[0,1,"ENTITY-VALUE"],[3,4,"ENTITY-VALUE"]]
        
        5.
        输入："无创血压、ENTITY；211-105、NUM；mmHg、UNIT"
        输出："ID:454278#下肢无创血压约211-105mmHg，"
        [I(0)D(1):(2)4(3)5(4)4(5)2(6)7(7)8(8)#(9)下(10)肢(11)无(12)创(13)血(14)压(15)约(16)2(17)1(18)1(19)-(20)1(21)0(22)5(23)m(24)m(25)H(26)g(27)，(28)]
        [[12,15,"ENTITY"],[17,23,"NUM"],[24,27,"UNIT"]]
        [[1,2,"NUM-UNIT"],[0,1,"ENTITY-VALUE"]]
        
        6.
        输入："1*1、NUM；平方厘米、UNIT；活组织、ENTITY"
        输出："ID:454749#下方各取1*1平方厘米活组织，"
        [I(0)D(1):(2)4(3)5(4)4(5)7(6)4(7)9(8)#(9)下(10)方(11)各(12)取(13)1(14)*(15)1(16)平(17)方(18)厘(19)米(20)活(21)组(22)织(23)，(24)]
        [[14,16,"NUM"],[17,20,"UNIT"],[21,23,"ENTITY"]]
        [[0,1,"NUM-UNIT"],[2,0,"ENTITY-VALUE"]]
        
        7.
        输入："苄星青霉素、ENTITY；240万、NUM；单位、UNIT"
        输出："ID:455018#按其医嘱予以苄星青霉素240万单位每周肌注治疗，"
        [I(0)D(1):(2)4(3)5(4)5(5)0(6)1(7)8(8)#(9)按(10)其(11)医(12)嘱(13)予(14)以(15)苄(16)星(17)青(18)霉(19)素(20)2(21)4(22)0(23)万(24)单(25)位(26)每(27)周(28)肌(29)注(30)治(31)疗(32)，(33)]
        [[16,20,"ENTITY"],[21,24,"NUM"],[25,26,"UNIT"]]
        [[1,2,"NUM-UNIT"],[0,1,"ENTITY-VALUE"]]
        
        8.
        输入："凝血酶原时间、ENTITY；11.30、NUM；秒、UNIT"
        输出："ID:456401#临检:出凝血全套A（PT/TT/APTT/FIB/D二聚体/ATII检验报告：凝血酶原时间11.30秒，"
        [I(0)D(1):(2)4(3)5(4)6(5)4(6)0(7)1(8)#(9)临(10)检(11):(12)出(13)凝(14)血(15)全(16)套(17)A(18)（(19)P(20)T(21)/(22)T(23)T(24)/(25)A(26)P(27)T(28)T(29)/(30)F(31)I(32)B(33)/(34)D(35)二(36)聚(37)体(38)/(39)A(40)T(41)I(42)I(43)检(44)验(45)报(46)告(47)：(48)凝(49)血(50)酶(51)原(52)时(53)间(54)1(55)1(56).(57)3(58)0(59)秒(60)，(61)]
        [[49,54,"ENTITY"],[55,59,"NUM"],[60,60,"UNIT"]]
        [[1,2,"NUM-UNIT"],[0,1,"ENTITY-VALUE"]]
        
        9.
        输入："0.9%氯化钠、ENTITY；1000、NUM；毫升、UNIT"
        输出："ID:452599#0.9%氯化钠1000毫升）后，"
        [I(0)D(1):(2)4(3)5(4)2(5)5(6)9(7)9(8)#(9)0(10).(11)9(12)%(13)氯(14)化(15)钠(16)1(17)0(18)0(19)0(20)毫(21)升(22)）(23)后(24)，(25)]
        [[10,16,"ENTITY"],[17,20,"NUM"],[21,22,"UNIT"]]
        [[1,2,"NUM-UNIT"],[0,1,"ENTITY-VALUE"]]
        
        """
    run(prompt_list)
