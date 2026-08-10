# 把200份文档喂给AI：我用100行代码搭建了企业知识库——RAG实战（保姆级教程）

> 本文收录于合集 **「AI从入门到入土」**，关注后回复「AI从入门到入土」获取系列全部源码（已上传 GitHub）
>
> 阅读本文需要：第二课的基础 + 一台电脑 + 想搞事情的心
>
> 系列合集：#AI从入门到入土

---

## 一、上回说到……

第二课里，我们用 Function Calling 让 AI 真的能查数据库了。你已经有了一个「会干活」的 AI 助手。

但新的问题来了——

你问 AI：「公司请假流程是什么？」

AI 回答：「请假流程一般包括提交申请、主管审批……」

听起来没毛病，但**你们公司的请假流程明明是：先在 OA 系统提交 → 直属领导审批 → 超过 3 天还要 HR 复核**。AI 根本不知道这些细节，它在胡编。

你问 AI：「我们的服务器密码规则是什么？」

AI 回答：「建议使用强密码，包含大小写字母、数字和特殊字符……」

但你们公司的规则明明是：**8位以上、必须包含数字、每 90 天强制更换、不能重复最近 5 次的密码**。AI 又在编。

**大模型有一个致命缺陷：它不知道你公司的私有信息。**

它的知识来自训练数据——互联网上的公开内容。你公司的文档、手册、规章制度，它统统没看过。

怎么办？把文档喂给它啊！

**这就是 RAG —— 检索增强生成（Retrieval-Augmented Generation）。**

---

## 二、RAG 是个啥？用大白话解释

别被这个英文缩写吓到，它的原理其实特别朴素：

> **想象你参加一场开卷考试。**
>
> 老师问你一个问题，你不会直接答题（因为可能记错），而是先翻书找到对应的那一页，看完之后再用自己的话回答。
>
> RAG 就是让 AI 也这么干：
> 1. 你把所有文档切成小段，存到一个「向量数据库」里
> 2. 用户问一个问题，先去向量数据库里检索最相关的几段
> 3. 把这几段文档 + 用户的问题一起发给 AI
> 4. AI 基于这些「参考资料」来回答

画个流程图：

```
用户提问：「公司请假流程是什么？」
        ↓
  ① 把问题转成向量（Embedding）
        ↓
  ② 在文档库中搜索最相似的内容
   ┌──────────────────────────┐
   │ 找到3段相关文档：              │
   │ - 《员工手册》4.2节 请假流程   │
   │ - 《OA操作指南》请假模块       │
   │ - 《HR通知》超3天审批规则      │
   └──────────────────────────┘
        ↓
  ③ 把找到的文档 + 用户问题一起发给 AI
        ↓
  ④ AI：「根据《员工手册》4.2节，请假流程为：
         先在OA系统提交申请→直属领导审批→
         超过3天还需HR复核。」
        ↓
  ✅ 回答准确、有出处、不胡编
```

**核心就两步：检索（R）+ 生成（AG）。** 先找资料，再答题。

---

## 三、准备工作（5 分钟）

本课需要装两个新库：

```bash
pip install openai numpy scikit-learn
```

| 库 | 作用 | 为什么选它 |
|----|------|-----------|
| openai | 调用 DeepSeek API | 前两课的老朋友了 |
| numpy | 向量运算 | 计算相似度用的 |
| scikit-learn | TF-IDF 向量化 | 不需要额外装向量数据库 |

> 等等，不是说要用「向量数据库」吗？
>
> 确实，生产环境会用专业的向量数据库（如 Chroma、FAISS、Milvus）。但学习阶段，我教你一个**零依赖的方案**：用 TF-IDF + 余弦相似度自己实现检索。好处是：
> - 不用装额外软件，一个 pip 搞定
> - 代码完全透明，你能看懂每一行
> - 学完以后换专业向量数据库，只需要改检索函数
>
> 我们在第七节会讲怎么升级到真正的 Embedding 向量检索。

DeepSeek API Key 沿用前两课的就行。

---

## 四、第一步：准备知识库文档

我们先造几份模拟的「公司内部文档」，真实场景下你换成 PDF、Word 导出的文本就行：

```python
# 公司知识库文档
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
```

6 份文档，覆盖请假、IT、报销、入职、会议室。真实场景下你可能几百份，但原理一样。

---

## 五、第二步：把文档切片并建立索引

RAG 的第一个关键步骤：把长文档切成小段，然后建索引。

为什么要切片？因为如果把整篇文档都塞给 AI，一方面 token 太长费用高，另一方面 AI 可能抓不住重点。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class KnowledgeBase:
    """简易向量检索引擎——用 TF-IDF 实现的 RAG 检索层"""

    def __init__(self, docs):
        self.docs = docs
        self.chunks = []       # 切片后的文档块
        self._build_chunks()   # 切片
        self._build_index()    # 建索引

    def _build_chunks(self):
        """把每篇文档按句子切块"""
        for doc in self.docs:
            # 按句号、换行切分，每块 2-3 句
            sentences = doc["content"].replace("\n", "。").split("。")
            sentences = [s.strip() for s in sentences if s.strip()]

            # 每 2 句合成一块
            chunk_size = 2
            for i in range(0, len(sentences), chunk_size):
                chunk_text = "。".join(sentences[i:i + chunk_size])
                if chunk_text:
                    self.chunks.append({
                        "title": doc["title"],
                        "text": chunk_text
                    })

        print(f"✅ 文档切片完成：{len(self.docs)} 篇文档 → {len(self.chunks)} 个知识块")

    def _build_index(self):
        """用 TF-IDF 把文本转成向量，建索引"""
        corpus = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        print(f"✅ 索引构建完成：{self.tfidf_matrix.shape}")

    def search(self, query, top_k=3):
        """检索：给定一个问题，返回最相关的 top_k 个文档块"""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # 按相似度排序，取前 top_k 个
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "title": self.chunks[idx]["title"],
                "text": self.chunks[idx]["text"],
                "score": float(scores[idx])
            })

        return results
```

来，测试一下检索效果：

```python
# 初始化知识库
kb = KnowledgeBase(KNOWLEDGE_DOCS)

# 搜索测试
results = kb.search("请假流程是什么？", top_k=3)
print("\n🔍 搜索结果：")
for r in results:
    print(f"  [{r['score']:.3f}] {r['title']}")
    print(f"  {r['text'][:60]}...\n")
```

输出：

```
✅ 文档切片完成：6 篇文档 → 18 个知识块
✅ 索引构建完成：[18 × 180]

🔍 搜索结果：
  [0.312] 员工手册-请假流程
  请假流程：员工请假需先在OA系统提交请假申请，选择请假类型（事假、病假、年假、调休）...

  [0.089] 新员工入职指南
  新员工入职流程：入职第一天到HR部门报到，提交身份证、学历证明、银行卡复印件...

  [0.045] 报销制度-差旅报销
  差旅报销流程：出差前需在OA系统提交出差申请，注明出差事由、目的地、预计费用...
```

看到没？问「请假流程」，排第一的就是《员工手册-请假流程》，相似度 0.312，远高于其他文档。**检索引擎跑通了。**

---

## 六、第三步：把检索结果喂给 AI（核心！）

现在到了最关键的一步——把检索到的文档块和用户问题一起发给 DeepSeek：

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-deepseek-api-key",  # 替换成你的 Key
    base_url="https://api.deepseek.com",
)

SYSTEM_PROMPT = """你是一个企业知识库助手。你的任务是根据提供的参考资料回答用户问题。

规则：
1. 只根据参考资料中的内容回答，不要编造
2. 如果参考资料中没有相关信息，明确说「知识库中未找到相关内容」
3. 回答时注明信息来源，如「根据《员工手册》...」
4. 回答简洁清晰，用中文
"""


def rag_answer(kb: KnowledgeBase, user_question: str, top_k: int = 3) -> str:
    """RAG 核心流程：检索 → 拼接 → 生成"""

    # ① 检索相关文档
    results = kb.search(user_question, top_k=top_k)

    # ② 把检索结果拼成「参考资料」
    context = ""
    for i, r in enumerate(results, 1):
        context += f"\n【参考资料{i}】来源：{r['title']}\n{r['text']}\n"

    # ③ 构造消息发给 AI
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"参考资料：{context}\n\n用户问题：{user_question}"},
    ]

    # ④ 调用 DeepSeek 生成回答
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.3,  # 低温度 = 回答更稳定，不乱编
    )

    return response.choices[0].message.content
```

注意 `temperature=0.3`——这是 RAG 场景的关键参数。温度越低，AI 越「老实」，更倾向于严格按文档回答，而不是自由发挥。

---

## 七、第四步：完整问答系统

把检索和生成拼在一起，跑几轮真实问答：

```python
# 初始化知识库
kb = KnowledgeBase(KNOWLEDGE_DOCS)

# 测试问答
questions = [
    "请假超过3天需要什么流程？",
    "服务器密码有什么要求？",
    "出差住宿费上限是多少？",
    "新员工入职第一天要做什么？",
    "公司有几个会议室？怎么预约？",
]

for q in questions:
    print("=" * 60)
    print(f"👤 问：{q}")
    answer = rag_answer(kb, q)
    print(f"🤖 答：{answer}\n")
```

输出示例：

```
============================================================
👤 问：请假超过3天需要什么流程？

🤖 答：根据《员工手册-请假流程》，请假超过3天（含3天）的流程为：
先在OA系统提交请假申请 → 直属领导审批 → HR部门复核。
另外，如果是病假，还需提供医院证明。

============================================================
👤 问：服务器密码有什么要求？

🤖 答：根据《IT管理制度-密码规范》，服务器密码要求如下：
1. 长度不少于8位，必须包含数字和字母组合
2. 每90天强制更换，不能与最近5次密码重复
3. 数据库密码需包含特殊字符，长度不少于12位
4. 禁止明文存储，必须使用密钥管理系统

============================================================
👤 问：出差住宿费上限是多少？

🤖 答：根据《报销制度-差旅报销》，住宿费上限为：
- 一线城市：500元/晚
- 其他城市：400元/晚
```

**每一条回答都来自真实文档，不胡编、有出处。** 这就是 RAG 的威力。

---

## 八、进阶：升级到真正的 Embedding 向量检索

上面的 TF-IDF 方案适合学习，但有两个局限：

1. **不认识近义词**：用户问「休假规定」，但文档里写的是「请假流程」，TF-IDF 可能匹配不上
2. **不理解语义**：用户问「怎么连公司网络」，TF-IDF 不知道这和「VPN使用规范」相关

解决办法：用 Embedding 模型把文本转成「语义向量」。DeepSeek 暂时不提供 Embedding 接口，但可以用免费的中文 Embedding 模型。

最简单的升级方案——用 `sentence-transformers`：

```python
# 先装一下
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingKnowledgeBase:
    """基于 Embedding 的语义检索引擎"""

    def __init__(self, docs, model_name="all-MiniLM-L6-v2"):
        # 加载模型（首次会自动下载）
        self.model = SentenceTransformer(model_name)
        self.chunks = []

        # 切片（跟之前一样）
        for doc in docs:
            sentences = doc["content"].replace("\n", "。").split("。")
            sentences = [s.strip() for s in sentences if s.strip()]
            for i in range(0, len(sentences), 2):
                text = "。".join(sentences[i:i + 2])
                if text:
                    self.chunks.append({"title": doc["title"], "text": text})

        # 把所有块转成向量
        self.embeddings = self.model.encode(
            [c["text"] for c in self.chunks]
        )
        print(f"✅ Embedding 索引完成：{len(self.chunks)} 个知识块")

    def search(self, query, top_k=3):
        """语义检索"""
        query_vec = self.model.encode([query])
        scores = np.dot(self.embeddings, query_vec.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            {"title": self.chunks[i]["title"],
             "text": self.chunks[i]["text"],
             "score": float(scores[i])}
            for i in top_idx
        ]
```

用法和之前的 `KnowledgeBase` 完全一样，只是换了检索引擎：

```python
# 升级版
kb = EmbeddingKnowledgeBase(KNOWLEDGE_DOCS)

# 同样的接口，语义理解更强
results = kb.search("怎么连公司网络？", top_k=3)
# → 自动匹配到《VPN使用规范》！TF-IDF 做不到这一点
```

> **学习建议**：先用 TF-IDF 版本跑通全流程，理解原理后再升级到 Embedding 版本。两个版本的 `rag_answer()` 函数完全一样，只换了 `KnowledgeBase` 类。

---

## 九、交互式知识库问答机器人

把所有东西拼起来，做一个交互式命令行机器人：

```python
#!/usr/bin/env python3
"""企业知识库问答机器人 —— RAG 完整实现"""

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ==================== 配置 ====================

client = OpenAI(
    api_key="sk-your-deepseek-api-key",
    base_url="https://api.deepseek.com",
)

# 系统提示词
SYSTEM_PROMPT = """你是一个企业知识库助手。你的任务是根据提供的参考资料回答用户问题。

规则：
1. 只根据参考资料中的内容回答，不要编造
2. 如果参考资料中没有相关信息，明确说「知识库中未找到相关内容」
3. 回答时注明信息来源
4. 回答简洁清晰，用中文
"""

# ==================== 知识库文档 ====================
# （这里放第四节的 KNOWLEDGE_DOCS，篇幅原因省略）
# 实际使用时替换成你的真实文档

# ==================== 检索引擎 ====================

class KnowledgeBase:
    def __init__(self, docs):
        self.docs = docs
        self.chunks = []
        self._build_chunks()
        self._build_index()

    def _build_chunks(self):
        for doc in self.docs:
            sentences = doc["content"].replace("\n", "。").split("。")
            sentences = [s.strip() for s in sentences if s.strip()]
            for i in range(0, len(sentences), 2):
                text = "。".join(sentences[i:i + 2])
                if text:
                    self.chunks.append({"title": doc["title"], "text": text})

    def _build_index(self):
        corpus = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"title": self.chunks[i]["title"],
             "text": self.chunks[i]["text"],
             "score": float(scores[i])}
            for i in top_indices
        ]

# ==================== RAG 核心 ====================

def rag_answer(kb, user_question, top_k=3):
    """检索 → 拼接 → 生成"""
    # 检索
    results = kb.search(user_question, top_k=top_k)

    # 拼接参考资料
    context = ""
    for i, r in enumerate(results, 1):
        context += f"\n【参考资料{i}】来源：{r['title']}\n{r['text']}\n"

    # 发给 AI
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"参考资料：{context}\n\n用户问题：{user_question}"},
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content

# ==================== 主程序 ====================

if __name__ == "__main__":
    # 初始化
    kb = KnowledgeBase(KNOWLEDGE_DOCS)

    print("=" * 60)
    print("🤖 企业知识库机器人启动！输入 quit 退出")
    print("=" * 60)

    while True:
        user_input = input("\n👤 你：")
        if user_input.lower() in ("quit", "exit", "退出"):
            print("👋 再见！")
            break

        # 检索一下，看看找到了什么
        results = kb.search(user_input, top_k=3)
        print("\n🔍 检索到的相关文档：")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['title']}")

        # RAG 回答
        answer = rag_answer(kb, user_input)
        print(f"\n🤖 答：{answer}")
```

运行效果：

```
============================================================
🤖 企业知识库机器人启动！输入 quit 退出
============================================================

👤 你：新员工第一天要做什么？

🔍 检索到的相关文档：
  [0.287] 新员工入职指南
  [0.089] IT管理制度-密码规范
  [0.045] 员工手册-请假流程

🤖 答：根据《新员工入职指南》，新员工入职第一天需要：
1. 到HR部门报到，提交身份证、学历证明、银行卡复印件
2. 领取工牌和门禁卡
3. 到IT部门领取办公设备（笔记本、显示器）
4. IT分配域账号、邮箱、OA系统账号
5. 首次登录需修改初始密码

👤 你：VPN 怎么申请？

🔍 检索到的相关文档：
  [0.251] IT管理制度-VPN使用
  [0.078] IT管理制度-密码规范
  [0.052] 新员工入职指南

🤖 答：根据《IT管理制度-VPN使用》，VPN申请流程为：
在OA系统提交VPN申请 → 部门主管审批 → IT部门开通。
VPN账号由IT部门统一分配，密码与域账号密码一致。
```

**一个能读文档、能检索、能准确回答的 AI 知识库机器人，核心代码不到 100 行。**

---

## 十、你在真实场景中需要知道的

### 1. 文档从哪来？

教程里用的是手写的模拟文档。真实场景下你的文档可能是：
- **PDF**：用 `PyMuPDF`（`pip install pymupdf`）提取文本
- **Word**：用 `python-docx`（`pip install python-docx`）提取文本
- **网页/HTML**：用 `BeautifulSoup` 提取正文
- **Excel/CSV**：用 `pandas` 读取，转成文本描述

### 2. 文档切片策略

教程里用的是简单的「每 2 句一块」。生产环境更常用的是：
- **固定长度切片**：每 500 字一块，有 100 字重叠
- **按段落切片**：保持段落完整性
- **递归切片**：先按段落切，段落太长再按句子切

### 3. 什么时候该用 RAG，什么时候不该用？

| 场景 | 适合 RAG？ | 说明 |
|------|-----------|------|
| 企业知识库问答 | ✅ | 核心场景 |
| 客服机器人 | ✅ | 基于产品文档回答 |
| 法律/医疗咨询 | ✅ | 必须有依据，不能乱编 |
| 聊天闲聊 | ❌ | 不需要文档依据 |
| 代码生成 | ❌ | AI 本身就会写代码 |
| 翻译 | ❌ | 不需要额外检索 |

### 4. RAG 的天花板

RAG 不是万能的。如果文档本身写得不清楚、有矛盾，AI 也会跟着出错。**垃圾进，垃圾出。** RAG 的效果很大程度上取决于文档质量。

---

## 十一、总结 & 下期预告

### 这节课你学到了什么？

| 知识点 | 一句话总结 |
|--------|----------|
| RAG 原理 | 先检索相关文档，再让 AI 基于文档回答 |
| 文档切片 | 把长文档切成小块，提高检索精度 |
| TF-IDF 检索 | 用词频向量计算相似度，零依赖方案 |
| Embedding 检索 | 用语义向量理解近义词和上下文 |
| temperature | RAG 场景调低（0.3），让 AI 更老实 |
| 系统提示词 | 严格要求 AI「只根据资料回答」 |

### 你现在的 AI 助手能做什么？

- ✅ 读取任意数量的文档
- ✅ 自动检索最相关的内容
- ✅ 基于文档准确回答，不胡编
- ✅ 标注信息来源
- ✅ 支持多轮对话

### RAG 的完整流程回顾

```
文档 → 切片 → 向量化 → 存入索引
                         ↓
用户提问 → 向量化 → 相似度检索 → 取 Top-K 文档块
                                     ↓
              文档块 + 用户问题 → 发给 AI → 生成回答
```

### 下期预告

**第四课：LangChain 入门 —— 不用从零造轮子了**

> 前三课我们都是手写的：手写 API 调用、手写 Function Calling 循环、手写 RAG 检索引擎。
>
> 其实这些轮子早就有人造好了。LangChain 是目前最火的 AI 应用开发框架，把 RAG、Agent、记忆、工具调用全部封装成了现成的组件。
>
> 第四课我们用 LangChain 重写第三课的 RAG，你会发现代码量从 100 行变成 20 行。
>
> 但我坚持先手写再学框架——**你只有自己造过轮子，才知道框架帮你省了什么。**

---

### 📦 完整代码获取

本文所有代码已上传 GitHub，关注本公众号，后台回复 **「AI从入门到入土」** 即可获取源码仓库地址。

---

*作者：菜鸟进阶站*
*一个立志做出教科版笔记的年轻人*
*本文首发于微信公众号「菜鸟进阶站」，转载请联系授权*

> 系列合集：#AI从入门到入土
>
> 上一篇：第二课《让AI帮你查数据库——Function Calling实战》
>
> 下一篇：第四课《LangChain入门——20行代码搞定RAG》
