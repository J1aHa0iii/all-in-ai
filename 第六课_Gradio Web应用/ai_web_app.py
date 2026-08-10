"""
第六课：Gradio Web应用 —— 把AI助手变成网页产品
用法：
  python ai_web_app.py                # 本地运行 (http://127.0.0.1:7860)
  python ai_web_app.py --share        # 生成公网分享链接
  python ai_web_app.py --port 8080    # 自定义端口

依赖安装：
  pip install gradio openai langchain-community faiss-cpu sentence-transformers
"""

import os
import sys
import json
import sqlite3
import tempfile
from openai import OpenAI

# ==================== 配置 ====================

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "你的Key写这里")
DEEPSEEK_BASE = "https://api.deepseek.com"


def get_client(api_key=None):
    """获取 OpenAI 客户端（支持网页传入 Key）"""
    key = api_key or DEEPSEEK_KEY
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE)


# ==================== 知识库初始化（启动时运行一次）====================

print("🔧 初始化知识库...")
KB_DIR = tempfile.mkdtemp()
docs_data = {
    "退货政策.txt": (
        "蓝牙耳机：7天无理由退货，需保留原包装和配件。"
        "无线键盘：15天内出现质量问题可换新，退货需扣除20%折旧费。"
        "鼠标垫：拆封后不支持退货。"
        "手机支架：30天无理由退货。"
    ),
    "保修政策.txt": (
        "蓝牙耳机：自购买之日起1年质保，非人为损坏免费维修。"
        "无线键盘：2年质保，第一年免费，第二年收取材料费。"
        "手机支架：6个月质保。"
        "鼠标垫：无质保服务。"
    ),
    "发货说明.txt": (
        "下单后24小时内发货（节假日顺延）。"
        "配送时效：江浙沪皖次日达，华北华南2-3天，西部偏远地区3-5天。"
        "全场满99元包邮，不满99元收取6元运费。"
        "默认发中通快递，可加5元升级顺丰。"
    ),
    "公司简介.txt": (
        "菜鸟科技有限公司成立于2020年，总部位于深圳。"
        "主营业务：智能硬件研发与销售，包括蓝牙耳机、智能手表、无线键盘等。"
        "2025年营收突破5亿元，员工规模500+人。"
        "核心价值观：用户至上、技术驱动、诚信为本。"
    ),
    "常见问题.txt": (
        "Q: 如何修改订单地址？A: 发货前可在订单详情页修改，发货后联系客服。"
        "Q: 可以开发票吗？A: 支持电子发票和纸质发票，下单时勾选即可。"
        "Q: 收到商品有质量问题怎么办？A: 签收24小时内拍照联系客服，核实后免费换新。"
        "Q: 支持哪些支付方式？A: 微信支付、支付宝、银行卡。"
    ),
}

for name, content in docs_data.items():
    with open(os.path.join(KB_DIR, name), "w", encoding="utf-8") as f:
        f.write(content)

# 构建向量索引
try:
    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    all_docs = []
    for fname in os.listdir(KB_DIR):
        all_docs.extend(TextLoader(
            os.path.join(KB_DIR, fname), encoding="utf-8"
        ).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"✅ 知识库初始化完成：{len(chunks)} 个文本块，{len(docs_data)} 份文档")
except ImportError as e:
    print(f"⚠️ 知识库初始化失败（可能是依赖未安装）：{e}")
    print("  运行: pip install langchain-community faiss-cpu sentence-transformers")
    vectorstore = None


# ==================== 辅助函数 ====================

def build_sqlite_demo():
    """构建 SQLite 内存示例数据库"""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product TEXT NOT NULL,
            category TEXT,
            amount REAL NOT NULL,
            quantity INTEGER NOT NULL,
            month TEXT NOT NULL,
            region TEXT
        )
    """)
    data = [
        ("蓝牙耳机", "数码", 15000, 50, "2026-06", "华南"),
        ("蓝牙耳机", "数码", 18000, 60, "2026-07", "华南"),
        ("蓝牙耳机", "数码", 22000, 73, "2026-08", "华南"),
        ("无线键盘", "数码", 12000, 40, "2026-06", "华东"),
        ("无线键盘", "数码", 13500, 45, "2026-07", "华东"),
        ("无线键盘", "数码", 11000, 37, "2026-08", "华东"),
        ("鼠标垫", "配件", 3000, 200, "2026-06", "华北"),
        ("鼠标垫", "配件", 3500, 233, "2026-07", "华北"),
        ("鼠标垫", "配件", 2800, 187, "2026-08", "华北"),
        ("手机支架", "配件", 8000, 160, "2026-06", "西南"),
        ("手机支架", "配件", 9500, 190, "2026-07", "西南"),
        ("手机支架", "配件", 10200, 204, "2026-08", "西南"),
        ("智能手表", "数码", 25000, 25, "2026-07", "华南"),
        ("智能手表", "数码", 30000, 30, "2026-08", "华南"),
        ("充电宝", "配件", 6000, 120, "2026-07", "华东"),
        ("充电宝", "配件", 7200, 144, "2026-08", "华东"),
    ]
    conn.executemany("INSERT INTO sales(product,category,amount,quantity,month,region) VALUES(?,?,?,?,?,?)", data)
    conn.commit()
    return conn


def format_table_result(cursor, rows):
    """把查询结果格式化为 Markdown 表格"""
    if not rows:
        return "*(无结果)*"
    cols = [d[0] for d in cursor.description] if cursor.description else []
    md = "| " + " | ".join(cols) + " |\n"
    md += "| " + " | ".join(["---"] * len(cols)) + " |\n"
    for row in rows:
        md += "| " + " | ".join(str(c) for c in row) + " |\n"
    return md


# ==================== 三大核心函数 ====================

def sql_assistant(message, history, api_key=""):
    """Tab 1: SQL 助手 —— 自然语言 → SQL → 数据库查询"""
    if not message.strip():
        return "请输入你的问题。"

    client = get_client(api_key)

    # 数据库表结构描述
    schema = """
表名: sales
字段:
  - id: 自增主键
  - product: 商品名称（蓝牙耳机、无线键盘、鼠标垫、手机支架、智能手表、充电宝）
  - category: 分类（数码、配件）
  - amount: 销售金额（元）
  - quantity: 销售数量
  - month: 月份（格式2026-06/07/08）
  - region: 区域（华南、华东、华北、西南）

示例查询: SELECT product, SUM(amount) as total FROM sales WHERE month='2026-07' GROUP BY product ORDER BY total DESC
"""

    # Step 1: AI 生成 SQL
    sql_prompt = f"""你是一个SQL专家。根据以下表结构和用户问题，只输出一条SELECT查询语句。
不要任何解释、不要markdown代码块标记、不要分号结尾以外的多余字符。

表结构：
{schema}

用户问题：{message}

SQL:"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": sql_prompt}],
            temperature=0,
        )
        sql = response.choices[0].message.content.strip()
        # 清理可能的 markdown 标记
        sql = sql.replace("```sql", "").replace("```", "").strip().rstrip(";")

        # Step 2: 执行 SQL
        conn = build_sqlite_demo()
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            result_table = format_table_result(cursor, rows)
            conn.close()
        except Exception as e:
            conn.close()
            return f"**❌ SQL 执行错误**\n\n生成的SQL：\n```sql\n{sql}\n```\n\n错误信息：{e}"

        return f"**📝 生成的SQL：**\n```sql\n{sql}\n```\n\n**📊 查询结果：**\n\n{result_table}"

    except Exception as e:
        return f"❌ API 调用失败：{e}\n\n请检查 API Key 是否正确。"


def kb_assistant(message, history, api_key=""):
    """Tab 2: 知识库问答 —— RAG 检索增强生成"""
    if not message.strip():
        return "请输入你的问题。"

    if vectorstore is None:
        return "⚠️ 知识库未初始化。请确保已安装 langchain-community, faiss-cpu, sentence-transformers。"

    client = get_client(api_key)

    try:
        # Step 1: 检索相关文档
        docs = vectorstore.similarity_search(message, k=3)
        context = "\n---\n".join(
            f"[来源：{d.metadata.get('source', '未知')}]\n{d.page_content}"
            for d in docs
        )

        # Step 2: 喂给 AI
        prompt = f"""你是一个企业AI助手。请根据以下公司资料回答用户问题。

规则：
1. 如果资料中有答案，直接引用并回答
2. 如果资料中只有部分信息，诚实说明哪些有、哪些不确定
3. 如果资料中完全没有相关信息，说"知识库中没有找到相关信息，建议联系人工客服"
4. 回答要简洁，尽量用分点列表

公司资料：
{context}

用户问题：{message}

回答："""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        answer = response.choices[0].message.content

        # 拼接来源
        sources = set()
        for d in docs:
            src = os.path.basename(d.metadata.get("source", ""))
            sources.add(src.replace(".txt", ""))

        return f"{answer}\n\n---\n📎 *参考来源：{', '.join(sources)}*"

    except Exception as e:
        return f"❌ 处理出错：{e}"


def agent_assistant(message, history, api_key=""):
    """Tab 3: Agent 智能体 —— AI 自主决策用什么工具"""
    if not message.strip():
        return "请输入你的问题。"

    if vectorstore is None:
        return "⚠️ Agent 依赖知识库，请先确保知识库已初始化。"

    client = get_client(api_key)

    # 工具定义
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search_policy",
                "description": "搜索公司政策知识库。可查询退货政策、保修政策、发货说明、公司信息、常见问题等。输入关键词或问题。",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "搜索关键词"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "执行数学计算。输入数学表达式（如 '158*37*0.85' 或 '1000+200'）。仅用于纯数学运算。",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string", "description": "数学表达式"}},
                    "required": ["expression"],
                },
            },
        },
    ]

    TOOL_FNS = {
        "search_policy": lambda query: "\n".join(
            d.page_content for d in vectorstore.similarity_search(query, k=2)
        ),
        "calculate": lambda expression: str(eval(expression)),
    }

    SYSTEM_PROMPT = """你是企业智能助手。你拥有以下工具：

1. search_policy: 搜索公司知识库（退货/保修/发货/公司信息/FAQ等）
2. calculate: 执行数学计算

工作规则：
- 用户问政策类问题 → 调用 search_policy
- 用户问计算问题 → 调用 calculate
- 用户可能一个问题需要多个工具 → 分步调用
- 信息齐全后，综合所有结果给出完整回答
- 一次只调用一个工具
- 回答时自然流畅，把工具结果融入回复中"""

    # 构建消息（包含历史）
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-5:]:
        if h[0]:
            messages.append({"role": "user", "content": h[0]})
        if h[1]:
            messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    try:
        tool_log = []  # 记录调用了哪些工具

        for step in range(5):
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            # 直接回答了
            if msg.content and not msg.tool_calls:
                # 有工具调用记录时加上说明
                if tool_log:
                    tools_used = ", ".join(set(tool_log))
                    return f"{msg.content}\n\n---\n🔧 *使用了工具：{tools_used}*"
                return msg.content

            # 调用工具
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = TOOL_FNS[name](**args)
                tool_log.append(name)
                
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
                    "content": str(result),
                })

        return "抱歉，处理超时。请换个更具体的问题试试。"

    except Exception as e:
        return f"❌ 处理出错：{e}\n\n请检查 API Key 是否正确，以及网络是否通畅。"


# ==================== Gradio 界面 ====================

def build_ui():
    try:
        import gradio as gr
    except ImportError:
        print("❌ 请先安装 Gradio：pip install gradio")
        sys.exit(1)

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="gray",
    )

    with gr.Blocks(
        title="🤖 AI 智能助手 | 菜鸟进阶站",
        theme=theme,
        css="""
        .main-header { text-align: center; margin-bottom: 10px; }
        .footer { text-align: center; color: #999; font-size: 12px; margin-top: 20px; }
        """,
    ) as app:

        # 标题
        gr.Markdown(
            """
            <div class="main-header">
            <h1>🤖 AI 智能助手</h1>
            <p>SQL助手 · 知识库问答 · Agent智能体 —— 来自「菜鸟进阶站」</p>
            </div>
            """,
            elem_classes=["main-header"],
        )

        # 设置面板（折叠）
        with gr.Accordion("⚙️ 设置", open=False):
            api_key_input = gr.Textbox(
                label="DeepSeek API Key",
                type="password",
                placeholder="sk-...",
                value=DEEPSEEK_KEY if DEEPSEEK_KEY != "你的Key写这里" else "",
            )
            gr.Markdown(
                """
                > 💡 在这里输入你的 [DeepSeek API Key](https://platform.deepseek.com/api_keys)。
                > 如果不填，程序会尝试读取环境变量 `DEEPSEEK_API_KEY`。
                > Key 只在本页面使用，不会上传到任何服务器。
                """
            )

        # 三标签页
        with gr.Tabs():
            # Tab 1: SQL 助手
            with gr.TabItem("📊 SQL 助手"):
                gr.Markdown(
                    """
                    **用自然语言查询数据库。** AI 自动生成 SQL 并返回结果。

                    💬 试试这些：
                    - *上个月卖得最好的产品是什么？*
                    - *2026年7月各区域的销售总额*
                    - *数码类产品8月的平均单价是多少？*
                    """
                )
                gr.ChatInterface(
                    fn=lambda msg, hist: sql_assistant(
                        msg, hist, api_key_input.value
                    ),
                    chatbot=gr.Chatbot(height=450, render_markdown=True),
                    textbox=gr.Textbox(
                        placeholder="输入你的问题，比如：上个月卖了多少蓝牙耳机？",
                        scale=7,
                    ),
                    title="",
                )

            # Tab 2: 知识库问答
            with gr.TabItem("📚 知识库问答"):
                gr.Markdown(
                    """
                    **基于公司知识库回答问题。** 文档检索 + AI 生成答案。

                    💬 试试这些：
                    - *蓝牙耳机怎么退货？*
                    - *公司有多少员工？*
                    - *发货一般要几天？*
                    """
                )
                gr.ChatInterface(
                    fn=lambda msg, hist: kb_assistant(
                        msg, hist, api_key_input.value
                    ),
                    chatbot=gr.Chatbot(height=450, render_markdown=True),
                    textbox=gr.Textbox(
                        placeholder="输入你的问题，比如：无线键盘保修多久？",
                        scale=7,
                    ),
                    title="",
                )

            # Tab 3: Agent 智能体
            with gr.TabItem("🤖 Agent 智能体"):
                gr.Markdown(
                    """
                    **AI 自主决策用什么工具。** 该查知识库还是直接回答？不用你选。

                    💬 试试这些（一个消息问多个问题）：
                    - *蓝牙耳机退货政策是什么？顺便帮我算 158×37×0.85*
                    - *公司有多少人？充电宝7月卖了多少？*
                    """
                )
                gr.ChatInterface(
                    fn=lambda msg, hist: agent_assistant(
                        msg, hist, api_key_input.value
                    ),
                    chatbot=gr.Chatbot(height=450, render_markdown=True),
                    textbox=gr.Textbox(
                        placeholder="随便问，AI 自己判断该用什么工具...",
                        scale=7,
                    ),
                    title="",
                )

        # 页脚
        gr.Markdown(
            """
            <div class="footer">
            <p>「AI从入门到入土」系列 · 菜鸟进阶站出品</p>
            <p>关注公众号回复「AI从入门到入土」获取全部源码</p>
            </div>
            """,
            elem_classes=["footer"],
        )

    return app


# ==================== 启动 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI 智能助手 Web 应用")
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    parser.add_argument("--port", type=int, default=7860, help="监听端口（默认7860）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    args = parser.parse_args()

    app = build_ui()

    print("=" * 55)
    print("  🤖 AI 智能助手 —— 菜鸟进阶站出品")
    print("=" * 55)
    print(f"  本地访问: http://127.0.0.1:{args.port}")
    if args.share:
        print("  公网分享: 启动后会在下方显示链接（72小时有效）")
    print()
    print("  三个标签页：")
    print("    📊 SQL助手   —— 自然语言查数据库")
    print("    📚 知识库问答 —— RAG 检索增强回答")
    print("    🤖 Agent智能体 —— AI自主决策用工具")
    print()
    print("  按 Ctrl+C 停止服务")
    print("=" * 55)

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
