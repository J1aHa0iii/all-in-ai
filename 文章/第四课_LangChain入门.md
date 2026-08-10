# 20行代码重构100行RAG：LangChain入门实战（保姆级教程）

> 本文收录于合集 **「AI从入门到入土」**，关注后回复「AI从入门到入土」获取系列全部源码（已上传 GitHub）
>
> 阅读本文需要：第三课的基础 + 一台电脑 + 被自己写的代码折磨过的灵魂
>
> 系列合集：#AI从入门到入土

---

## 一、上回说到……

第三课我们搭了一个企业知识库，核心流程是这样的：

```python
# 第三课的手写 RAG，大概 100 行核心代码
class KnowledgeBase:       # 30 行 —— 切片 + TF-IDF 索引
    ...

def rag_answer(kb, q):     # 20 行 —— 检索 + 拼接 + 发给 AI
    ...

# 主程序                     # 50 行 —— 初始化 + 交互循环
while True:
    q = input()
    results = kb.search(q)
    answer = rag_answer(kb, q)
```

**确实能跑，但你有没有发现一个问题？**

每次想加点新功能，比如「记住对话历史」、或者「把多个文档源（PDF + 网页 + 数据库）混在一起检索」，代码就开始失控了。100 行变成 300 行，300 行变成 800 行。

**怎么避免重复造轮子？**

有大佬早就把这些通用逻辑封装成了框架——**LangChain**。

今天这篇文章我不跟你讲 LangChain 的底层原理（网上多的是），我只带你看一件事：

> **同样一个 RAG 知识库，手写用了 100 行，LangChain 只用 20 行。这 20 行里藏了哪些「被你忽略的工程细节」？**

---

## 二、LangChain 是个啥？一句大白话

LangChain = **一堆现成的积木，帮你拼 AI 应用**。

| 积木类型 | 举例 | 你自己写要多少行？ |
|---------|------|------------------|
| 模型调用 | `ChatOpenAI` | 5行（但不用手写 OpenAI 协议） |
| 文档加载 | `TextLoader`、`PyPDFLoader` | 每种格式你要写 20-50 行 |
| 文档切片 | `RecursiveCharacterTextSplitter` | 30行+ |
| 向量存储 | `FAISS`、`Chroma` | 100行+ |
| 检索链 | `RetrievalQA` | **这才是重头戏** |
| 对话记忆 | `ConversationBufferMemory` | 单独写又是 50 行 |

它不是什么黑科技——它就是把你前三课手写的东西标准化了，让你不用每次都从零开始。

**先手写、再学框架，这才是最佳学习路径。** 你现在已经有前三课的底子了，LangChain 的每一行代码你都能猜到它背后在干什么。

---

## 三、准备工作

老三样，一个新朋友：

```bash
# 新朋友 —— LangChain 全家桶
pip install langchain langchain-community langchain-openai

# 老朋友 —— DeepSeek 对接（LangChain 内置自己的 OpenAI client）
pip install openai

# FAISS 向量库（Facebook 开源的，工业级向量检索）
pip install faiss-cpu

# 文档处理
pip install pymupdf
```

> **为什么装这么多？** LangChain 是模块化设计的——核心逻辑在 `langchain`，社区贡献的 loader/splitter/vectorstore 在 `langchain-community`，各模型厂商的适配在 `langchain-openai`。分开装是为了不一次装太多你不需要的东西。

---

## 四、第一行：加载文档

第三课我们是手写了一个 Python 字典装文档。LangChain 把各种格式的文档读取都封装好了：

```python
from langchain_community.document_loaders import TextLoader

# 一行读取 markdown 文件
loader = TextLoader("知识库/员工手册.md", encoding="utf-8")
docs = loader.load()

print(f"加载了 {len(docs)} 篇文档")
print(docs[0].page_content[:200])  # 看看第一段内容
```

如果你的文档是 PDF：

```python
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("知识库/产品手册.pdf")
docs = loader.load()
```

**LangChain 支持的格式**：TXT、Markdown、PDF、Word、CSV、JSON、HTML、Notion、飞书文档……你只要换一个 Loader，代码其他部分完全不用动。

---

## 五、第二行：智能切片

第三课的切片很简单：每 2 句一块。但真实文档的段落长短不一，一刀切会出问题。

LangChain 的 `RecursiveCharacterTextSplitter` 的做法更聪明——先按段落切，段落太长再按句子切，句子还太长再按字符切：

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每块最多 500 字
    chunk_overlap=50,      # 块之间重叠 50 字（防止一句话被腰斩）
    separators=["\n\n", "\n", "。", "，", " ", ""],  # 逐级回退的切割符
)

chunks = text_splitter.split_documents(docs)
print(f"切成了 {len(chunks)} 个知识块")

# 看一眼第一个块
print(f"\n第一块内容（前200字）：\n{chunks[0].page_content[:200]}")
```

`chunk_overlap=50` 这个参数很关键——它让相邻的两块有 50 个字重叠，防止关键信息正好卡在切割边界上被丢掉。

**你手写的切片 vs LangChain 的切片：**

| 对比维度 | 手写（第三课） | LangChain |
|---------|-------------|-----------|
| 策略 | 固定 2 句一块 | 递归多层回退 |
| 重叠 | ❌ 没有 | ✅ chunk_overlap |
| 多格式支持 | ❌ 只支持纯文本 | ✅ 自动适配 Markdown/PDF/HTML |
| 元数据保留 | ❌ 手动 | ✅ 自动保留文档来源 |

---

## 六、第三行：向量化存储

第三课用的是 TF-IDF + 自己写相似度计算。LangChain 直接对接了工业级向量数据库：

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. 选一个 Embedding 模型（这里用 DeepSeek 兼容 OpenAI 接口的）
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 但 DeepSeek 没有 Embedding 接口……
    base_url="https://api.deepseek.com",
    api_key="sk-your-deepseek-api-key",
)
```

等等……**DeepSeek 目前不提供 Embedding API**。

这就是 LangChain 的一个好处了：**轻松换模型**。换成智谱的免费 Embedding：

```python
from langchain_community.embeddings import ZhipuAIEmbeddings

# 智谱 Embedding —— 注册就送额度，每天免费 100 万 tokens
embeddings = ZhipuAIEmbeddings(
    model="embedding-2",
    api_key="你的智谱API Key",  # 注册地址：open.bigmodel.cn
)

# 如果嫌麻烦想用本地模型（完全不要 API Key）👇
# 也可以用 sentence-transformers（第三课提过的方案）
```

或者用 HuggingFace 的免费方案，完全不需要任何 API Key：

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# 免费本地 Embedding，第一次会自动下载模型（约 100MB）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",  # 中文 Embedding，体积小效果好
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

选好 Embedding 模型后，建向量库只需要一行：

```python
# 2. 自动向量化所有知识块，存入 FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

print(f"✅ 向量库构建完成，共 {len(chunks)} 条知识")
```

**一行代码 = 你第三课写的 50 行 TF-IDF 索引代码。**

---

## 七、第四行：组装 RAG 链

这是 LangChain 最大的杀手锏——用「链」把各个组件串起来：

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. 选择模型（DeepSeek）
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="sk-your-deepseek-api-key",
    temperature=0.3,  # RAG 场景保持低温度
)

# 2. 一条链搞定：检索 + 生成
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",          # 最简模式：检索到的文档直接拼进 prompt
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 3}   # 取 Top-3 最相关文档块
    ),
    return_source_documents=True,  # 返回引用来源
)
```

**LangChain 的 4 种 chain_type：**

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `stuff` | 把所有检索结果塞进一个 prompt | 文档较少（< 4K tokens） |
| `map_reduce` | 每篇文档单独问答，再汇总 | 文档很多 |
| `refine` | 一篇篇叠代优化答案 | 需要从多篇文档中综合 |
| `map_rerank` | 每篇打分，取最高分的回答 | 答案很可能只在一篇文档里 |

教程用 `stuff` 就够了——我们的知识库不大，一次能全塞进去。

---

## 八、效果演示

```python
# 测试几个问题
questions = [
    "请假超过3天需要什么流程？",
    "服务器密码有什么要求？",
    "出差住宿费上限是多少？",
]

for q in questions:
    result = qa_chain.invoke({"query": q})
    print(f"👤 问：{q}")
    print(f"🤖 答：{result['result']}")
    
    # 看看引用了哪些文档
    print("📎 参考来源：")
    for doc in result["source_documents"]:
        print(f"  - {doc.metadata.get('source', '未知来源')}")
    print("-" * 60)
```

输出示例：

```
👤 问：请假超过3天需要什么流程？
🤖 答：根据《员工手册-请假流程》，请假超过3天的流程为：
先在OA系统提交申请 → 直属领导审批 → HR部门复核。

📎 参考来源：
  - 知识库/员工手册.md
------------------------------------------------------------
```

**对比第三课，多了什么？** —— `source_documents`，自动告诉你答案来自哪篇文档。你手写的时候这个功能至少再加 15 行代码。

---

## 九、完整代码（20 行核心！）

```python
#!/usr/bin/env python3
"""用 LangChain 重构第三课的 RAG —— 20 行核心代码"""
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# ===== 大前提：文档必须上传到 知识库/ 目录下 =====
# 用法：把 .md、.txt 文件丢到 知识库/，然后跑这个脚本

DEEPSEEK_KEY = "sk-your-deepseek-api-key"
KNOWLEDGE_DIR = "知识库"

# 1. 加载文档（3行）
documents = []
for f in os.listdir(KNOWLEDGE_DIR):
    if f.endswith((".md", ".txt")):
        loader = TextLoader(os.path.join(KNOWLEDGE_DIR, f), encoding="utf-8")
        documents.extend(loader.load())

print(f"📂 加载了 {len(documents)} 篇文档")

# 2. 切片（3行）
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"✂️ 切成了 {len(chunks)} 个知识块")

# 3. 向量化存储（2行）
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = FAISS.from_documents(chunks, embeddings)
print("🧠 向量库构建完成")

# 4. 组装 RAG 链（5行）
llm = ChatOpenAI(
    model="deepseek-chat", base_url="https://api.deepseek.com",
    api_key=DEEPSEEK_KEY, temperature=0.3,
)
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
)

# 5. 交互式问答（7行）
print("\n" + "=" * 60)
print("🤖 RAG 知识库机器人启动！输入 quit 退出")
print("=" * 60)

while True:
    q = input("\n👤 你：")
    if q.lower() in ("quit", "exit", "退出"):
        break
    result = qa.invoke({"query": q})
    print(f"\n🤖 答：{result['result']}")
    sources = set(d.metadata.get("source", "未知") for d in result["source_documents"])
    print(f"📎 来源：{', '.join(sources)}")
```

**核心代码不到 20 行。** 第三课你写了 100 行做的事，LangChain 帮你省了 80%。

---

## 十、手写 vs LangChain：框架到底帮你省了什么？

| 你手写干的活 | LangChain 代为处理的 | 省了多少行 |
|------------|-------------------|----------|
| 文件读取 + 编码处理 | `TextLoader` / `PyMuPDFLoader` | ~20 行 |
| 递归切片 + 重叠 | `RecursiveCharacterTextSplitter` | ~30 行 |
| TF-IDF 向量化 + 相似度计算 | `FAISS` + `HuggingFaceEmbeddings` | ~50 行 |
| 拼接 prompt + 调用 API | `RetrievalQA` 链 | ~20 行 |
| 来源追溯 | `return_source_documents=True` | ~15 行 |
| 多格式文档混用 | 换一个 Loader 就行 | 无法估量 |
| 多轮对话记忆 | 加一个 `Memory` 组件 （第五课讲） | ~50 行 |

**LangChain 不是魔法——它只是把你手写的代码标准化了。**

这就是为什么我先让你手写三课再学框架。当你看到 `FAISS.from_documents()` 这一行的时候，你知道它背后在做：分词 → 向量化 → 建倒排索引 → 相似度归一化——你已经懂了，就不会被框架「黑盒」吓到。

---

## 十一、进阶：给知识库加「记忆」

第三课的知识库每次都是「一锤子买卖」——你问一句它答一句，完全不记得你上一句问了什么。

LangChain 加对话记忆只需要几行代码：

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,  # 👈 就是多这一行
)

# 现在可以多轮对话了
qa.invoke({"question": "请假流程是什么？"})
qa.invoke({"question": "你刚才说的超过3天那个，要在OA系统的哪个入口提交？"})
# → AI 记得上下文！知道「那个」指的是请假超过3天的流程
```

这就引出了下一课的主题——**Agent 智能体**。记忆 + 工具调用 + 自主决策，让 AI 从「助手」升级成「员工」。

---

## 十二、总结 & 下期预告

### 这节课的核心收获

| 知识点 | 一句话 |
|--------|--------|
| LangChain 是什么 | 把 RAG、Agent、Memory 标准化的积木框架 |
| Document Loader | 一行代码读取 MD/PDF/Word/网页…… |
| Recursive splitter | 多层回退切片，比手写更聪明 |
| FAISS | 工业级向量检索，替代手写 TF-IDF |
| RetrievalQA | 一条链搞定检索+生成，省 50 行代码 |
| Memory | 给 AI 加记忆，支持多轮对话 |

### 学习路径里程碑

```
手写时代（第一~三课）
├── 第一课：调通 API，知道 AI 怎么「说话」
├── 第二课：Function Calling，知道 AI 怎么「动手」
└── 第三课：手写 RAG，知道 AI 怎么「查资料」
         ↓
框架时代（第四课起）
├── 第四课 ✅ LangChain 重构 RAG，知道框架怎么「省钱」
├── 第五课 ⏳ Agent 智能体，让 AI 自己决定「先做什么」
└── 第六课 ⏳ 完整 AI 应用实战
```

### 下期预告

**第五课：Agent 智能体实战——让 AI 自己决定该查数据库还是该翻文档**

> 前三课的 AI 都只会做「你指定的一件事」：要么写 SQL、要么查数据库、要么翻文档。
>
> 但真实场景是这样的：你问「上个月销量最高的产品有没有对应知识库里的退货政策？」——AI 需要先把问题拆成两段：① 查数据库找出销量最高产品 ② 翻知识库查该产品的退货政策 ③ 把两个结果串起来回答。
>
> 这就是 **Agent**——给 AI 一个工具箱，让它自己决定用哪个、什么时候用、怎么组合。
>
> 第五课，我们用手写 Agent（先造轮子）再帮你过渡到 LangChain Agent，从原理到实战，一次性搞懂。

---

### 📦 完整代码获取

本文所有代码已上传 GitHub，关注本公众号，后台回复 **「AI从入门到入土」** 即可获取源码仓库地址。

---

*作者：菜鸟进阶站*
*一个立志做出教科版笔记的年轻人*
*本文首发于微信公众号「菜鸟进阶站」，转载请联系授权*

> 系列合集：#AI从入门到入土
>
> 上一篇：第三课《把200份文档喂给AI——RAG实战》
>
> 下一篇：第五课《Agent智能体实战——让AI自己决定该干什么》
