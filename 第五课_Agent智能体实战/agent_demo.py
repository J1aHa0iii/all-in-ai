#!/usr/bin/env python3
"""Agent 智能体实战 —— 手写版 + LangChain 版，改 Key 就能跑

使用：
  python agent_demo.py               # 手写 Agent（不依赖 LangChain）
  python agent_demo.py --langchain   # LangChain Agent（需要先装 langchain）

依赖：
  pip install openai
  pip install langchain langchain-community langchain-openai faiss-cpu sentence-transformers  # LangChain 模式需要
"""

import json
import sqlite3
import os
import sys

# ==================== 配置 ====================

DEEPSEEK_KEY = "sk-your-deepseek-api-key"  # 替换成你的 Key

from openai import OpenAI
client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")

# ==================== 工具函数 ====================

def query_sales(sql: str) -> str:
    """执行 SQL 查询，模拟销售数据库"""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE sales ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  product TEXT NOT NULL,"
        "  amount REAL NOT NULL,"
        "  month TEXT NOT NULL"
        ")"
    )
    data = [
        ("蓝牙耳机", 58000, "2026-07"),
        ("无线键盘", 42000, "2026-07"),
        ("鼠标垫", 12000, "2026-07"),
        ("手机支架", 8000, "2026-07"),
        ("蓝牙耳机", 35000, "2026-06"),
        ("无线键盘", 38000, "2026-06"),
        ("鼠标垫", 9000, "2026-06"),
        ("手机支架", 15000, "2026-06"),
    ]
    conn.executemany("INSERT INTO sales (product, amount, month) VALUES (?, ?, ?)", data)

    try:
        rows = conn.execute(sql).fetchall()
        conn.close()
        return json.dumps(rows, ensure_ascii=False)
    except Exception as e:
        conn.close()
        return f"SQL错误：{e}"


def build_knowledge_base():
    """构建内存向量知识库（首次运行下载 Embedding 模型约100MB）"""
    print("🧠 正在构建知识库（首次运行需下载模型）……")

    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    knowledge = {
        "退货政策": (
            "蓝牙耳机：7天无理由退货，需保留原包装。"
            "无线键盘：15天质量问题换新。"
            "鼠标垫：拆封后不予退换。"
            "手机支架：30天无理由退货。"
        ),
        "保修政策": (
            "蓝牙耳机：1年质保。"
            "无线键盘：2年质保。"
            "鼠标垫：无质保。"
            "手机支架：6个月质保。"
        ),
        "发货说明": (
            "下单后24小时内发货。"
            "江浙沪皖次日达，其他地区2-3天送达。"
            "全场满99元包邮，不满99元运费8元。"
        ),
    }

    kb_dir = "/tmp/agent_kb"
    os.makedirs(kb_dir, exist_ok=True)
    for name, content in knowledge.items():
        filepath = os.path.join(kb_dir, f"{name}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    docs = []
    for f in sorted(os.listdir(kb_dir)):
        if f.endswith(".txt"):
            docs.extend(TextLoader(os.path.join(kb_dir, f), encoding="utf-8").load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"✅ 知识库构建完成：{len(chunks)} 个知识块\n")
    return vectorstore


# 全局初始化（手写版和 LangChain 版都用到）
print("=" * 50)
print("   🤖 Agent 智能体 —— 给 AI 一个工具箱")
print("=" * 50)
vectorstore = build_knowledge_base()


def search_knowledge(query: str) -> str:
    """搜索公司知识库，返回相关政策文档"""
    results = vectorstore.similarity_search(query, k=2)
    return "\n".join(r.page_content for r in results)


# ==================== 工具注册表 ====================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_sales",
            "description": (
                "查询销售数据库。输入标准SQL SELECT语句，返回查询结果（JSON列表）。"
                "数据库表名：sales，列：product(产品名), amount(销售额), month(月份，格式YYYY-MM)。"
                "示例SQL：SELECT product, SUM(amount) as total FROM sales "
                "WHERE month='2026-07' GROUP BY product ORDER BY total DESC"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "要执行的SQL SELECT查询语句",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "搜索公司内部知识库，返回相关文档内容。"
                "包含：退货政策、保修政策、发货说明等。"
                "输入关键词或问题，返回文档片段。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，例如「蓝牙耳机退货」「保修政策」",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "query_sales": query_sales,
    "search_knowledge": search_knowledge,
}

# ==================== Agent 循环（手写版） ====================

SYSTEM_PROMPT = """你是一个智能助手，可以使用工具来回答用户问题。

工作流程：
1. 分析用户问题，判断需要哪些信息
2. 需要数据统计 → 调用 query_sales 工具
3. 需要政策规定 → 调用 search_knowledge 工具
4. 收到工具结果后，判断是否还需要其他信息
5. 信息齐全后，综合所有结果用自然语言回答

规则：
- 一次只调用一个工具
- 工具参数必须是合法的JSON
- SQL查询只允许SELECT语句
- 如果问题涉及同一个实体的多个方面（比如查销量+查政策），先查数据再查政策
- 用中文回答，简洁清晰"""


def run_agent(question: str, max_steps: int = 5) -> str:
    """手写 Agent 核心循环 —— ReAct 模式"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        print(f"\n{'─' * 40}")
        print(f"🔄 第 {step} 步：思考中……")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # AI 决定直接回答
        if msg.content and not msg.tool_calls:
            print("💬 AI 决定直接回答（信息已足够）")
            return msg.content

        # AI 决定调用工具
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            print(f"🔧 调用工具：{name}")
            print(f"   参数：{json.dumps(args, ensure_ascii=False)}")

            result = TOOL_FUNCTIONS[name](**args)
            preview = result[:100] + "……" if len(result) > 100 else result
            print(f"📊 返回：{preview}")

            # 追加工具调用和结果到对话历史
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [{
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": tc.function.arguments,
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "⚠️ 达到最大步数限制，请重新提问。"


# ==================== LangChain Agent 版 ====================

def run_langchain_agent():
    """使用 LangChain 的 Agent 框架"""
    from langchain_openai import ChatOpenAI
    from langchain.agents import tool, initialize_agent, AgentType

    @tool
    def query_sales_tool(sql: str) -> str:
        """查询销售数据库。输入SQL SELECT语句返回JSON格式的查询结果。数据库包含product, amount, month列。"""
        return query_sales(sql)

    @tool
    def search_knowledge_tool(query: str) -> str:
        """搜索公司知识库。输入关键词返回退货政策、保修政策、发货说明等相关文档。"""
        return search_knowledge(query)

    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=DEEPSEEK_KEY,
        temperature=0,
    )

    agent = initialize_agent(
        [query_sales_tool, search_knowledge_tool],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    print("\n" + "=" * 50)
    print("   🤖 Agent 智能体 —— LangChain 版")
    print("   工具：查数据库 | 搜知识库 | 直接回答")
    print("   输入问题开始，输入 quit 退出")
    print("=" * 50)

    while True:
        try:
            q = input("\n👤 你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break
        if q.lower() in ("quit", "exit", "退出", "q"):
            print("👋 再见！")
            break
        if not q:
            continue
        answer = agent.run(q)
        print(f"\n🤖 答：{answer}")
        print("-" * 50)


# ==================== 主程序 ====================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--langchain":
        run_langchain_agent()
    else:
        print("\n工具：查数据库 | 搜知识库 | 直接回答")
        print("输入问题开始，输入 quit 退出")
        print("加上 --langchain 参数可切换到 LangChain 版")
        print("=" * 50)

        while True:
            try:
                q = input("\n👤 你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                break
            if q.lower() in ("quit", "exit", "退出", "q"):
                print("👋 再见！")
                break
            if not q:
                continue
            answer = run_agent(q)
            print(f"\n🤖 答：{answer}")
            print("-" * 50)
