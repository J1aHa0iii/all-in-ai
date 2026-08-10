# 让AI自己决定先干嘛：Agent智能体实战（保姆级教程）

> 本文收录于合集 **「AI从入门到入土」**，关注后回复「AI从入门到入土」获取系列全部源码（已上传 GitHub）
>
> 阅读本文需要：前四课的基础 + 一台电脑 + 不想再给AI当保姆的心
>
> 系列合集：#AI从入门到入土

---

## 一、上回说到……

第四课我们用 LangChain 搭建了 RAG 知识库，AI 终于能读文档了。

但现在你的 AI 工具箱里有三样武器：

| 武器 | 来源 | 能干嘛 |
|------|------|--------|
| 写 SQL | 第一课 | AI 生成 SQL 语句 |
| 查数据库 | 第二课 | Function Calling 执行查询 |
| 翻文档 | 第三、四课 | RAG 检索知识库 |

**问题来了——每次用哪个武器，都是你（人类）在帮它选。**

你问「上个月销售额最高的产品退货政策是什么？」——这个问题的完整回答需要两步：① 查数据库找产品名 → ② 翻知识库找该产品的退货政策。但前几课的 AI 只能做其中一步。

**Agent 就是让 AI 自己选武器、自己决定先用哪个后用哪个。**

---

## 二、Agent 是个啥？用大白话解释

想象你是一个老板，手下有三个员工：

- **小王**（查数据库）—— 只会跑 SQL
- **小张**（翻文档）—— 只会读文件
- **小李**（算数学）—— 只会加减乘除

现在你问：「上个月销量最高的产品名字是什么？它的退货政策呢？」

以前你得这样指挥：先叫小王查数据库 → 拿到产品名 → 再叫小张去翻文档。

**Agent 就是把这套「思考 → 选工具 → 执行 → 再看结果 → 再选工具」的逻辑写进了代码里，让 AI 自己当老板。**

标准术语叫 **ReAct 模式**（Reasoning + Acting）：

```
Thought：我需要知道上个月销量最高的产品是谁
Action：调用「查数据库」工具
Observation：返回「无线蓝牙耳机」
Thought：现在我知道了产品名，需要查它的退货政策
Action：调用「翻文档」工具
Observation：7天无理由退货，需要保留原包装
Final Answer：无线蓝牙耳机退货政策是……
```

**这套逻辑听起来玄乎，但核心代码只有 30 行。** 这节课我带你从零写一个。

---

## 三、准备工作

老三样，加一个新工具——SQLite 内存数据库：

```bash
# 老朋友
pip install openai

# 新朋友 —— 向量检索（第四课用过）
pip install langchain langchain-community faiss-cpu sentence-transformers

# SQLite 是 Python 内置的，不用装
```

---

## 四、第一步：定义工具

工具就是「AI 能调用的函数」。每个工具需要三个信息：**名字**、**描述**、**参数**。

```python
# ==================== 工具定义 ====================

import sqlite3
import json

# 工具1：查数据库
def query_sales(sql):
    """执行 SQL 查询，返回结果"""
    conn = sqlite3.connect(":memory:")  # 内存数据库，每次运行都是新的
    conn.execute("CREATE TABLE sales (product TEXT, amount REAL, month TEXT)")
    
    # 模拟数据
    data = [
        ("蓝牙耳机", 58000, "2026-07"),
        ("无线键盘", 42000, "2026-07"),
        ("鼠标垫", 12000, "2026-07"),
        ("蓝牙耳机", 35000, "2026-06"),
        ("手机支架", 8000, "2026-07"),
    ]
    conn.executemany("INSERT INTO sales VALUES (?, ?, ?)", data)
    
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return json.dumps(rows, ensure_ascii=False)
    except Exception as e:
        conn.close()
        return f"SQL 错误：{e}"


# 工具2：查知识库
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os

# 构建知识库（跟第四课一样）
def build_vectorstore():
    docs = []
    knowledge = {
        "退货政策": "蓝牙耳机：7天无理由退货，需保留原包装。无线键盘：15天质量问题换新。鼠标垫：拆封后不退换。手机支架：30天无理由退货。",
        "保修政策": "蓝牙耳机：1年质保。无线键盘：2年质保。鼠标垫：无质保。手机支架：6个月质保。",
        "发货说明": "下单后24小时内发货。江浙沪皖次日达，其他地区2-3天。全场满99元包邮。",
    }
    
    # 写入临时文件
    os.makedirs("/tmp/agent_kb", exist_ok=True)
    for name, content in knowledge.items():
        with open(f"/tmp/agent_kb/{name}.txt", "w", encoding="utf-8") as f:
            f.write(content)
    
    for f in os.listdir("/tmp/agent_kb"):
        loader = TextLoader(f"/tmp/agent_kb/{f}", encoding="utf-8")
        docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(chunks, embeddings)


def search_knowledge(query):
    """搜索知识库，返回相关文档内容"""
    results = vectorstore.similarity_search(query, k=2)
    return "\n".join([r.page_content for r in results])


# 初始化向量库
vectorstore = build_vectorstore()
```

> **注意**：上面的代码里，知识库文档写了三份模拟数据。你实际使用时把这个字典换成从文件读取就行——第四课教过的 `TextLoader` 直接读。

---

## 五、第二步：工具注册表

AI 不是直接调函数，而是通过**注册表**告诉它有哪些工具可用：

```python
# ==================== 工具注册表 ====================

TOOLS = [
    {
        "name": "query_sales",
        "description": "查询销售数据库。输入标准的 SQL SELECT 语句，返回查询结果。"
                       "示例：SELECT product, SUM(amount) as total FROM sales WHERE month='2026-07' GROUP BY product ORDER BY total DESC",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "要执行的 SQL 查询语句"}
            },
            "required": ["sql"],
        },
    },
    {
        "name": "search_knowledge",
        "description": "搜索公司知识库。输入一个查询关键词，返回相关的政策、流程、说明文档。"
                       "适用于：退货政策、保修政策、发货说明、联系方式等。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词，例如「退货政策」「蓝牙耳机保修」"}
            },
            "required": ["query"],
        },
    },
]

# 工具名称 → 函数的映射
TOOL_FUNCTIONS = {
    "query_sales": query_sales,
    "search_knowledge": search_knowledge,
}
```

这个注册表有两个作用：
1. **把 `TOOLS` 发给 AI**，告诉它「我有哪些工具、怎么调用」
2. **`TOOL_FUNCTIONS` 是给 Python 用的**，AI 说「调 query_sales」，代码就去查这个字典执行对应的函数

---

## 六、第三步：核心——Agent 循环（30行！）

这就是整个 Agent 的灵魂：

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-deepseek-api-key",
    base_url="https://api.deepseek.com",
)

SYSTEM_PROMPT = """你是一个智能助手。你可以使用工具来回答用户问题。

工作流程：
1. 分析用户问题，判断需要哪些信息
2. 如果需要数据 → 调用 query_sales 工具查询数据库
3. 如果需要政策/规定 → 调用 search_knowledge 工具搜知识库
4. 如果工具结果不够 → 可以再次调用工具
5. 信息齐全后 → 综合所有结果，用自然语言回答

注意：一次只调用一个工具。收到工具返回结果后，决定下一步。"""

def run_agent(user_question, max_steps=5):
    """Agent 核心循环 —— 思考 → 行动 → 观察 → 再思考"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]
    
    step = 0
    while step < max_steps:
        step += 1
        print(f"\n{'='*50}")
        print(f"🔄 第 {step} 步")
        
        # ① 让 AI 思考下一步做什么
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",  # 👈 AI 自己决定：用工具还是直接回答
        )
        
        msg = response.choices[0].message
        
        # ② 如果 AI 觉得信息够了，直接回答
        if msg.content and not msg.tool_calls:
            print(f"💬 AI 决定直接回答")
            return msg.content
        
        # ③ 如果 AI 想调工具，执行它
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            print(f"🔧 调用工具：{tool_name}")
            print(f"   参数：{tool_args}")
            
            # 执行工具函数
            func = TOOL_FUNCTIONS[tool_name]
            result = func(**tool_args)
            
            print(f"📊 工具返回：{result[:100]}...")
            
            # ④ 把工具执行结果追加到对话中
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_call.function.arguments,
                    }
                }]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
    
    return "达到最大步数限制，请重新提问。"


# ==================== 测试 ====================

if __name__ == "__main__":
    questions = [
        "蓝牙耳机的退货政策是什么？",
        "上个月销量最高的产品是哪个？它的保修政策是什么？",
        "发货要多久？满多少包邮？",
    ]
    
    for q in questions:
        print(f"\n{'#'*60}")
        print(f"👤 用户：{q}")
        answer = run_agent(q)
        print(f"\n🤖 最终回答：{answer}")
```

**整个 Agent 循环的核心就是这 30 行 `run_agent` 函数。** 它做的事：

```
用户提问 → AI 分析：需要查数据？
         → 是 → 调工具 → 拿到结果 → 再分析：还需要查文档？
         → 是 → 调工具 → 拿到结果 → 信息够了，直接回答
         → 否 → 直接回答
```

跟第二课的 Function Calling 有什么区别？**第二课的 tool_choice 是你指定的，Agent 的 tool_choice="auto" 是 AI 自己选的。**

---

## 七、效果演示

```
👤 用户：蓝牙耳机的退货政策是什么？

🔄 第 1 步
🔧 调用工具：search_knowledge
   参数：{'query': '蓝牙耳机退货政策'}
📊 工具返回：蓝牙耳机：7天无理由退货，需保留原包装。无线键盘：15天质量问题换新...

💬 AI 直接回答

🤖 最终回答：根据知识库中的退货政策，蓝牙耳机支持7天无理由退货，需要保留原包装。
```

```
👤 用户：上个月销量最高的产品是哪个？它的保修政策是什么？

🔄 第 1 步
🔧 调用工具：query_sales
   参数：{'sql': "SELECT product, SUM(amount) as total FROM sales WHERE month='2026-07' GROUP BY product ORDER BY total DESC LIMIT 1"}
📊 工具返回：[["蓝牙耳机", 58000.0]]

🔄 第 2 步
🔧 调用工具：search_knowledge
   参数：{'query': '蓝牙耳机保修政策'}
📊 工具返回：蓝牙耳机：1年质保...

💬 AI 直接回答

🤖 最终回答：上个月（2026年7月）销量最高的产品是蓝牙耳机，销售额为58000元。
根据保修政策，蓝牙耳机享有1年质保服务。
```

**注意第二个例子——AI 先查数据库、拿到产品名后自动又去查保修政策。** 这就是 Agent 的「自主决策」。以前你要手动走两遍，现在它自己搞定了。

---

## 八、进阶：LangChain 写 Agent 只需要几行

手写 Agent 帮你理解了原理，但生产环境中你不想每次手写循环。LangChain 已经封装好了：

```python
from langchain_openai import ChatOpenAI
from langchain.agents import tool, initialize_agent, AgentType

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="sk-your-deepseek-api-key",
    temperature=0,
)

# LangChain 的 @tool 装饰器自动把函数包装成 Agent 工具
@tool
def query_sales_tool(sql: str) -> str:
    """查询销售数据库。输入 SQL SELECT 语句，返回查询结果。"""
    return query_sales(sql)  # 调用前面写好的函数

@tool  
def search_knowledge_tool(query: str) -> str:
    """搜索公司知识库。输入关键词，返回相关政策文档。"""
    return search_knowledge(query)

tools = [query_sales_tool, search_knowledge_tool]

# 👇 三行搞定 Agent！
agent = initialize_agent(
    tools, llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ReAct 模式
    verbose=True,  # 打印思考过程
)

# 使用
result = agent.run("上个月销量最高的产品是哪个？它的保修政策是什么？")
print(result)
```

**LangChain Agent 的三种模式：**

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `ZERO_SHOT_REACT_DESCRIPTION` | 根据工具描述自己决策 | 简单的多工具场景 |
| `CONVERSATIONAL_REACT_DESCRIPTION` | 带记忆的 ReAct | 多轮对话 + 多工具 |
| `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION` | 多参数工具 | 工具参数复杂时 |

教程用 `ZERO_SHOT_REACT_DESCRIPTION` 就足够了。

---

## 九、完整代码

把所有东西拼起来——手写版 + LangChain 版，一个文件两个模式：

```python
#!/usr/bin/env python3
"""Agent 智能体实战 —— 手写版 + LangChain 版，改 Key 就能跑"""

import json, sqlite3, os, sys
from openai import OpenAI

DEEPSEEK_KEY = "sk-your-deepseek-api-key"
client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")

# ==================== 工具函数 ====================

def query_sales(sql: str) -> str:
    """执行 SQL，模拟销售数据库"""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sales (product TEXT, amount REAL, month TEXT)")
    data = [
        ("蓝牙耳机", 58000, "2026-07"), ("无线键盘", 42000, "2026-07"),
        ("鼠标垫", 12000, "2026-07"), ("蓝牙耳机", 35000, "2026-06"),
        ("手机支架", 8000, "2026-07"), ("无线键盘", 38000, "2026-06"),
    ]
    conn.executemany("INSERT INTO sales VALUES (?, ?, ?)", data)
    try:
        rows = conn.execute(sql).fetchall()
        conn.close()
        return json.dumps(rows, ensure_ascii=False)
    except Exception as e:
        conn.close()
        return f"SQL错误：{e}"


def build_kb():
    """构建内存知识库"""
    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    knowledge = {
        "退货政策": "蓝牙耳机：7天无理由退货需保留原包装。无线键盘：15天质量问题换新。鼠标垫：拆封后不退换。手机支架：30天无理由退货。",
        "保修政策": "蓝牙耳机：1年质保。无线键盘：2年质保。鼠标垫：无质保。手机支架：6个月质保。",
        "发货说明": "下单后24小时内发货。江浙沪皖次日达，其他地区2-3天。全场满99元包邮。",
    }
    os.makedirs("/tmp/agent_kb", exist_ok=True)
    for name, content in knowledge.items():
        with open(f"/tmp/agent_kb/{name}.txt", "w", encoding="utf-8") as f:
            f.write(content)

    docs = []
    for f in os.listdir("/tmp/agent_kb"):
        docs.extend(TextLoader(f"/tmp/agent_kb/{f}", encoding="utf-8").load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(chunks, embeddings)

vectorstore = build_kb()

def search_knowledge(query: str) -> str:
    """搜索知识库"""
    results = vectorstore.similarity_search(query, k=2)
    return "\n".join(r.page_content for r in results)


# ==================== 工具注册表 ====================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_sales",
            "description": "查询销售数据库。输入标准SQL SELECT语句。示例：SELECT product, SUM(amount) FROM sales WHERE month='2026-07' GROUP BY product ORDER BY SUM(amount) DESC",
            "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索公司知识库。输入关键词查询退货/保修/发货等政策。示例：蓝牙耳机退货政策",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
]

TOOL_FUNCTIONS = {"query_sales": query_sales, "search_knowledge": search_knowledge}

# ==================== Agent 循环 ====================

SYSTEM_PROMPT = """你是智能助手。你可以使用工具回答用户问题。
工作流程：分析问题→调用工具获取信息→综合结果回答。
一次只调用一个工具。信息齐全后直接回答，不要多余调用。"""

def run_agent(question: str, max_steps: int = 5) -> str:
    """手写 Agent 核心循环"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        print(f"\n{'─'*40}\n🔄 第 {step} 步")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # 直接回答
        if msg.content and not msg.tool_calls:
            print("💬 AI 直接回答")
            return msg.content

        # 调用工具
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            print(f"🔧 调用：{name} {args}")
            result = TOOL_FUNCTIONS[name](**args)
            print(f"📊 结果：{result[:80]}...")

            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": name, "arguments": tc.function.arguments}}]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "达到最大步数。"

# ==================== 交互模式 ====================

if __name__ == "__main__":
    use_langchain = "--langchain" in sys.argv
    
    if use_langchain:
        # LangChain 模式
        from langchain_openai import ChatOpenAI
        from langchain.agents import tool, initialize_agent, AgentType
        
        @tool
        def query_sales_tool(sql: str) -> str:
            """查询销售数据库。输入SQL SELECT语句返回结果。"""
            return query_sales(sql)
        
        @tool
        def search_knowledge_tool(query: str) -> str:
            """搜索公司知识库。输入关键词返回相关政策。"""
            return search_knowledge(query)
        
        llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com", api_key=DEEPSEEK_KEY, temperature=0)
        agent = initialize_agent(
            [query_sales_tool, search_knowledge_tool], llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
        )
        
        print("=" * 50)
        print("   🤖 Agent 智能体 —— LangChain 版")
        print("   工具：查数据库 / 搜知识库 / 直接回答")
        print("=" * 50)
        
        while True:
            try:
                q = input("\n👤 你：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "退出", "q"):
                break
            if not q:
                continue
            answer = agent.run(q)
            print(f"\n🤖 答：{answer}")
    else:
        # 手写 Agent 模式（默认）
        print("=" * 50)
        print("   🤖 Agent 智能体 —— 手写版")
        print("   工具：查数据库 / 搜知识库 / 直接回答")
        print("   加上 --langchain 参数可切换到 LangChain 版")
        print("=" * 50)
        
        while True:
            try:
                q = input("\n👤 你：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "退出", "q"):
                break
            if not q:
                continue
            answer = run_agent(q)
            print(f"\n🤖 答：{answer}")
```

**跑法**：

```bash
# 手写版（默认，不依赖 LangChain）
python 第五课_agent.py

# LangChain 版（加一个参数）
pip install langchain langchain-community langchain-openai faiss-cpu sentence-transformers
python 第五课_agent.py --langchain
```

---

## 十、总结 & 下期预告

### Agent vs 前四课的 AI

| 对比维度 | 第一~三课 | 第四课 LangChain | 第五课 Agent |
|---------|----------|-----------------|-------------|
| 能做几件事 | 一件 | 一件（但代码少了） | **多件** |
| 谁决定用哪个工具 | 你写死了 | 你写死了 | **AI 自己** |
| 需要几步完成 | 一步 | 一步 | **多步自动串行** |
| 典型场景 | 「帮我写个SQL」 | 「查一下」 | **「上个月销冠是啥？它的退货政策呢？」** |

### 这节课你学到了

| 知识点 | 一句话 |
|--------|--------|
| Agent 原理 | 思考→行动→观察→再思考 的循环 |
| ReAct 模式 | Reasoning + Acting |
| tool_choice="auto" | 让 AI 自己决定调工具还是直接回答 |
| 工具注册表 | 名字+描述+参数，告诉 AI 你能干嘛 |
| 多步推理 | AI 先查 A、拿结果、再查 B、最后回答 |
| LangChain Agent | 三行代码替代手写循环 |

### 下期预告

**第六课：上线！把你的 AI 助手变成一个 Web 应用**

> 五节课了，你还在黑框框里跟 AI 聊天。
>
> 第六课我们用 **Gradio** 把 SQL 助手、知识库、Agent 都变成带界面的 Web 应用，一个链接就能分享给同事用。
>
> 不需要学前端——Gradio 三行代码出一个网页，部署到公网只要一条命令。

---

### 📦 完整代码获取

本文所有代码已上传 GitHub，关注本公众号，后台回复 **「AI从入门到入土」** 即可获取源码仓库地址。

---

*作者：菜鸟进阶站*
*一个立志做出教科版笔记的年轻人*
*本文首发于微信公众号「菜鸟进阶站」，转载请联系授权*

> 系列合集：#AI从入门到入土
>
> 上一篇：第四课《LangChain入门——20行代码重构RAG》
>
> 下一篇：第六课《上线！把你的AI助手变成一个Web应用》
