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
        model="deepseek-r1",  # text-davinci-002: best, text-ada-001: lowest price
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
        请生成5个复杂输入输出训练对，输入包含 至少2个实体（可跨类别重复，如多个NUM或UNIT），输出需满足：
            句子复杂：含 并列结构、介词短语、技术术语（如“在...条件下”“误差不超过”“最大耐受值为”）。
            标注严格：字符级拆分索引，实体范围精确，仅使用 ENTITY-VALUE 和 NUM-UNIT 关系。
            场景专业：覆盖实验、工程、医学等领域的参数描述。

        格式规范（务必遵守）
        输入：实体名称1、类别1；实体名称2、类别2；实体名称3、类别3；...
        输出：
            合成描述：多实体组合的长句，含技术性表达。
            实体标注：每个字符必须单独编号，即使属于同一实体（例如：数字"15"需拆分为 1(3) 和 5(4)，标注为 (3,4,NUM)；单位"cm"需拆分为 c(5) 和 m(6)，标注为 (5,6,UNIT)）；单位"g/L"需拆分为 g(5) /(6) L(7)，标注为 (5,7,UNIT)）。
            关系标注：仅用 (x,y,"ENTITY-VALUE") 和 (y,z,"NUM-UNIT")，其中 x,y,z 对应输入中的实体顺序索引。
        
        生成的复杂示例演示：
        1. 
        输入：”最大耐受值、ENTITY；50、NUM；mg、UNIT；每日摄入量、ENTITY；2、NUM；次、UNIT“
        输出：”根据临床试验结果，最大耐受值为50mg，每日摄入量不得超过2次。
        [根(0)据(1)临(2)床(3)试(4)验(5)结(6)果(7)，(8)最(9)大(10)耐(11)受(12)值(13)为(14)5(15)0(16)m(17)g(18)，(19)每(20)日(21)摄(22)入(23)量(24)不(25)得(26)超(27)过(28)2(29)次(30)。(31)]
        [(9,13,ENTITY),(15,16,NUM),(17,18,UNIT),(20,24,ENTITY),(29,29,NUM),(30,30,UNIT)]
        [(0,1,"ENTITY-VALUE"),(1,2,"NUM-UNIT"),(3,4,"ENTITY-VALUE"),(4,5,"NUM-UNIT")]“
        关系标注解释：
        (0,1,"ENTITY-VALUE")：第0个实体（最大耐受值）与第1个实体（50）关联。
        (1,2,"NUM-UNIT")：第1个实体（50）与第2个实体（mg）关联。
        
        2.
        输入：”转速、ENTITY；1200、NUM；rpm、UNIT；工作温度、ENTITY；-10、NUM；℃、UNIT；40、NUM；℃、UNIT“
        输出：”设备在空载状态下，转速需稳定在1200rpm范围内，工作温度应介于-10℃至40℃之间。
        [设(0)备(1)在(2)空(3)载(4)状(5)态(6)下(7)，(8)转(9)速(10)需(11)稳(12)定(13)在(14)1(15)2(16)0(17)0(18)r(19)p(20)m(21)范(22)围(23)内(24)，(25)工(26)作(27)温(28)度(29)应(30)介(31)于(32)-(33)1(34)0(35)℃(36)至(37)4(38)0(39)℃(40)之(41)间(42)。(43)]
        [(9,10,ENTITY),(15,18,NUM),(19,21,UNIT),(26,29,ENTITY),(34,35,NUM),(36,36,UNIT),(38,39,NUM),(40,40,UNIT)]
        [(0,1,"ENTITY-VALUE"),(1,2,"NUM-UNIT"),(3,4,"ENTITY-VALUE"),(4,5,"NUM-UNIT"),(3,6,"ENTITY-VALUE"),(6,7,"NUM-UNIT")]“
        """

    """
    1. 会生成一些之外的数量信息，比如环境温度
    输入：”输出电压、ENTITY；3.3、NUM；V、UNIT；结温、ENTITY；-40、NUM；℃、UNIT；125、NUM；℃、UNIT“  
    输出：  
    合成描述：当环境温度低于0℃时，输出电压需调整为3.3V，结温必须在-40℃至125℃之间以保证稳定性。
    
    2. 会生成一些例外的关系："NUM-VALUE"
    3. 标注还是会有一些不正确：
    输入：”载波频率、ENTITY；2.4、NUM；GHz、UNIT；峰值电流、ENTITY；1.5、NUM；A、UNIT；纹波系数、ENTITY；3%、NUM；“  
    输出：  
    合成描述：射频模块参数要求：载波频率2.4GHz±50MHz，峰值电流1.5A时纹波系数需＜3%，工作温度-20℃~85℃。  
    实体标注：  
    [射(0)频(1)模(2)块(3)参(4)数(5)要(6)求(7)：(8)载(9)波(10)频(11)率(12)2(13).(14)4(15)G(16)H(17)z(18)±(19)5(20)0(21)M(22)H(23)z(24)，(25)峰(26)值(27)电(28)流(29)1(30).(31)5(32)A(33)时(34)纹(35)波(36)系(37)数(38)需(39)＜(40)3(41)%(42)，(43)工(44)作(45)温(46)度(47)-(48)2(49)0(50)℃(51)~(52)8(53)5(54)℃(55)。(56)]  
    [(9,12,ENTITY),(13,15,NUM),(16,18,UNIT),(26,29,ENTITY),(30,32,NUM),(33,33,UNIT),(35,38,ENTITY),(41,41,NUM)]  
    关系标注：  
    [(0,1,"ENTITY-VALUE"),(1,2,"NUM-UNIT"),(3,4,"ENTITY-VALUE"),(4,5,"NUM-UNIT"),(6,7,"ENTITY-VALUE")]
    
    输入：”半数抑制浓度、ENTITY；IC50、ENTITY；12.8、NUM；μM、UNIT；置信区间、ENTITY；95%、NUM；“  
    输出：  
    合成描述：经毒理学评估，化合物A的半数抑制浓度IC50为12.8μM（置信区间95%），符合欧盟REACH法规限值。  
    实体标注：  
    [经(0)毒(1)理(2)学(3)评(4)估(5)，(6)化(7)合(8)物(9)A(10)的(11)半(12)数(13)抑(14)制(15)浓(16)度(17)I(18)C(19)5(20)0(21)为(22)1(23)2(24).(25)8(26)μ(27)M(28)（(29)置(30)信(31)区(32)间(33)9(34)5(35)%(36)）(37)，(38)符(39)合(40)欧(41)盟(42)R(43)E(44)A(45)C(46)H(47)法(48)规(49)限(50)值(51)。(52)]  
    [(12,17,ENTITY),(18,21,ENTITY),(23,26,NUM),(27,28,UNIT),(30,33,ENTITY),(34,35,NUM)]  
    关系标注：  
    [(0,2,"ENTITY-VALUE"),(2,3,"NUM-UNIT"),(4,5,"ENTITY-VALUE")]  
    """
    run(prompt_list)
