#!/usr/bin/env python3
"""
第三课：RAG 实战 —— 企业知识库问答机器人
使用 DeepSeek API + TF-IDF 向量检索，从零搭建一个文档问答系统

使用方法：
1. pip install openai numpy scikit-learn
2. 把下面的 API_KEY 换成你的 DeepSeek Key
3. python rag_knowledge_base.py
"""

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ==================== 配置区 ====================

API_KEY = "sk-your-deepseek-api-key"  # ← 改成你的 DeepSeek API Key
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """你是一个企业知识库助手。你的任务是根据提供的参考资料回答用户问题。

规则：
1. 只根据参考资料中的内容回答，不要编造
2. 如果参考资料中没有相关信息，明确说「知识库中未找到相关内容」
3. 回答时注明信息来源，如「根据《员工手册》...」
4. 回答简洁清晰，用中文
"""


# ==================== 知识库文档 ====================
# 真实场景下替换成你的 PDF/Word/HTML 导出的文本

KNOWLEDGE_DOCS = [
    {
        "title": "员工手册-请假流程",
        "content": """请假流程：员工请假需先在OA系统提交请假申请，选择请假类型（事假、病假、年假、调休），
填写请假时间和事由，提交后由直属领导审批。如果请假超过3天（含3天），还需HR部门复核。
病假需提供医院证明。年假每年根据工龄计算，入职满1年享有5天年假，满3年享有10天。
未经审批擅自离岗按旷工处理，旷工3天以上可解除劳动合同。"""
    },
    {
        "title": "IT管理制度-密码规范",
        "content": """服务器密码规则：所有服务器密码长度不少于8位，必须包含数字和字母组合。
密码每90天强制更换一次，不能与最近5次使用过的密码重复。
数据库密码需额外包含特殊字符，长度不少于12位。
禁止将密码写在代码中或配置文件明文存储，必须使用密钥管理系统。
新员工首次登录需在24小时内修改初始密码。"""
    },
    {
        "title": "IT管理制度-VPN使用",
        "content": """VPN使用规范：员工远程办公需使用公司VPN接入内网。VPN账号由IT部门统一分配，
申请流程为：在OA系统提交VPN申请→部门主管审批→IT部门开通。
VPN密码与域账号密码一致，每90天同步更换。VPN连接后禁止共享给他人使用。
如发现账号异常登录，IT部门有权临时冻结账号。VPN客户端下载地址在IT门户首页。"""
    },
    {
        "title": "报销制度-差旅报销",
        "content": """差旅报销流程：出差前需在OA系统提交出差申请，注明出差事由、目的地、预计费用。
差旅费用包括交通费、住宿费、餐饮补贴。交通费按职级标准：普通员工火车硬卧/高铁二等座，
经理级别高铁一等座/飞机经济舱。住宿费上限：一线城市500元/晚，其他城市400元/晚。
餐饮补贴为100元/天。出差结束后5个工作日内提交报销单，附发票原件。
超过预算部分需特别说明，由部门负责人和财务双重审批。"""
    },
    {
        "title": "新员工入职指南",
        "content": """新员工入职流程：入职第一天到HR部门报到，提交身份证、学历证明、银行卡复印件。
领取工牌、门禁卡后到IT部门领取办公设备（笔记本、显示器）。IT会分配域账号、邮箱、OA系统账号。
首次登录需修改初始密码。入职第一周需完成线上安全培训考试，分数不低于80分。
入职满1个月后参加转正答辩，通过后转为正式员工。试用期工资为转正后的80%。"""
    },
    {
        "title": "会议室预约制度",
        "content": """会议室预约规则：公司共有5个会议室，A会议室（6人）、B会议室（10人）、
C会议室（20人）、D会议室（30人，带投影）、E会议室（50人，带视频会议）。
预约方式：在OA系统→行政办公→会议室预约，选择时间段和会议室。
同一时间段先到先得。预约后未使用且未提前30分钟取消的，记一次违规。
累计3次违规暂停预约权限1周。外部来访人员需在前台登记后方可进入会议室。"""
    },
]


# ==================== 检索引擎 ====================

class KnowledgeBase:
    """简易向量检索引擎——用 TF-IDF 实现的 RAG 检索层

    生产环境建议替换为 Embedding 方案（见文章第八节），
    或使用专业向量数据库如 Chroma、FAISS、Milvus。
    """

    def __init__(self, docs):
        self.docs = docs
        self.chunks = []  # 切片后的文档块
        self._build_chunks()
        self._build_index()

    def _build_chunks(self):
        """把每篇文档按句子切块，每 2 句一块"""
        for doc in self.docs:
            sentences = doc["content"].replace("\n", "。").split("。")
            sentences = [s.strip() for s in sentences if s.strip()]
            for i in range(0, len(sentences), 2):
                text = "。".join(sentences[i:i + 2])
                if text:
                    self.chunks.append({"title": doc["title"], "text": text})
        print(f"✅ 文档切片完成：{len(self.docs)} 篇文档 → {len(self.chunks)} 个知识块")

    def _build_index(self):
        """用 TF-IDF 把文本转成向量矩阵"""
        corpus = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        print(f"✅ 索引构建完成：矩阵形状 {self.tfidf_matrix.shape}")

    def search(self, query, top_k=3):
        """检索：给定一个问题，返回最相关的 top_k 个文档块"""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"title": self.chunks[i]["title"], "text": self.chunks[i]["text"], "score": float(scores[i])}
            for i in top_indices
        ]


# ==================== RAG 核心流程 ====================

def rag_answer(kb, user_question, top_k=3):
    """RAG = 检索（R）+ 增强（A）+ 生成（G）

    1. 检索：从知识库中找到最相关的文档块
    2. 增强：把文档块拼成参考资料，和问题一起发给 AI
    3. 生成：AI 基于参考资料生成回答
    """
    # ① 检索
    results = kb.search(user_question, top_k=top_k)

    # ② 拼接参考资料
    context = ""
    for i, r in enumerate(results, 1):
        context += f"\n【参考资料{i}】来源：{r['title']}\n{r['text']}\n"

    # ③ 发给 AI 生成回答
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"参考资料：{context}\n\n用户问题：{user_question}"},
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,  # 低温度 = 更老实，不乱编
    )
    return response.choices[0].message.content


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 初始化知识库
    kb = KnowledgeBase(KNOWLEDGE_DOCS)

    print()
    print("=" * 60)
    print("🤖 企业知识库机器人启动！输入 quit 退出")
    print("=" * 60)

    # 交互式问答
    while True:
        user_input = input("\n👤 你：").strip()
        if user_input.lower() in ("quit", "exit", "退出"):
            print("👋 再见！")
            break
        if not user_input:
            continue

        # 展示检索结果
        results = kb.search(user_input, top_k=3)
        print("\n🔍 检索到的相关文档：")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['title']}")

        # RAG 回答
        print("\n🤖 思考中...\n")
        answer = rag_answer(kb, user_input)
        print(f"🤖 答：{answer}")
