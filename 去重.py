import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 安装必要库（首次运行需要执行）
# pip install sentence-transformers numpy scikit-learn

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载中文优化模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def chinese_deduplicate(sentences, threshold=0.8):
    """
    中文语句去重函数
    :param sentences: 待去重的中文语句列表
    :param threshold: 相似度阈值（0.7-0.9之间）
    :return: 去重后的语句列表
    """
    # 生成语义嵌入向量
    embeddings = model.encode(sentences, convert_to_tensor=True).cpu().numpy()

    # 保留唯一索引的列表
    unique_indices = [0]

    # 遍历所有语句
    for i in range(1, len(embeddings)):
        current_embedding = embeddings[i].reshape(1, -1)

        # 计算与已保留语句的相似度
        similarities = cosine_similarity(current_embedding,
                                         embeddings[unique_indices])

        # 如果所有相似度都低于阈值，则保留该语句
        if np.all(similarities < threshold):
            unique_indices.append(i)

    # 返回去重后的中文语句
    return [sentences[idx] for idx in unique_indices]


samples = [
    "辅助检查:入院时血糖测定：4.7mmol/L。",
    "临检:出凝血全套A（PT/TT/APTT/FIB/D二聚体/ATII检验报告：凝血酶原时间10.60秒，",
    "活化部分凝血活酶时间26.1秒，",
    "红细胞4.77*10^12每升",
    "血常规分析检验报告：白细胞8.25*10^9每升，"
]
# 执行去重
cleaned = chinese_deduplicate(samples, threshold=0.5)

print("原始数据数量:", len(samples))
print("去重后数量:", len(cleaned))
print("保留结果:")
for s in cleaned:
    print(f"- {s}")

