# 上线！30行代码把你的AI助手变成Web应用——Gradio实战

> 本文收录于合集 **「AI从入门到入土」**，关注后回复「AI从入门到入土」获取系列全部源码（已上传 GitHub）
>
> 阅读本文需要：前五课的基础 + 一颗「不想再让同事用命令行」的心
>
> 系列合集：#AI从入门到入土

---

五节课了，你的 AI 助手一直跑在黑框框里。

同事想用？「你先把 Python 装上，再 pip install openai，然后配个环境变量，最后在终端输入 python ai_assistant.py……」

**人家转身就走。**

这节课，我们用 **Gradio** 给它穿一件「网页外衣」——一个链接发过去，浏览器打开就能用。不需要学 HTML/CSS/JS，不需要搞前后端分离，Python 一把梭。

读完你能做出来的东西：

![最终效果：一个带标签页的Web应用，左边SQL助手、中间知识库、右边Agent，输入问题点发送，AI在网页上直接回复]

---

## 一、前五课的尴尬

复习一下我们都做了什么：

| 课程 | 能力 | 但现在的问题是…… |
|------|------|------------------|
| 第一课 | DeepSeek 生成 SQL | 黑框框里打字，黑框框里看结果 |
| 第二课 | Function Calling 查数据库 | 同上 |
| 第三课 | RAG 知识库问答 | 同上 |
| 第四课 | LangChain 重构 | 同上，代码少了但界面没有 |
| 第五课 | Agent 智能体 | 同上，多工具切换但还是要敲命令 |

**东西越做越强，但给别人看的样子还是像在写代码。**

Gradio 解决的问题就一个：**三行代码，把 Python 函数变成一个网页界面。**

---

## 二、Gradio 大白话

### 2.1 它是个啥

Gradio 是一个 Python 库，专门给机器学习模型做 Web 演示界面。

你写一个函数：

```python
def greet(name):
    return f"你好，{name}！"
```

然后用 Gradio 包装：

```python
import gradio as gr
gr.Interface(fn=greet, inputs="text", outputs="text").launch()
```

跑起来——浏览器自动打开一个网页，上面一个输入框、一个输出框，你输入名字点按钮，它就显示「你好，张三！」。

**你就写了三行代码，它自动生成了一个完整的 Web 服务**——输入框、按钮、输出区、样式、响应式布局，啥都给你搞好了。

### 2.2 你不需要懂的东西

| 你不需要 | Gradio 帮你干了 |
|---------|----------------|
| HTML / CSS | 自动生成网页布局和样式 |
| JavaScript / fetch | 自动处理前后端通信 |
| Flask / FastAPI | 内置 Web 服务器 |
| 前端框架（React/Vue） | 不需要，纯 Python |

### 2.3 但我们这节课不止做这个

光是 `gr.Interface` 太简单了。我们要做一个**有标签页的、能切换三种模式的、带聊天记录的完整应用**。

三栏设计：

```
┌──────────────────────────────────────────────────┐
│  🤖 AI 智能助手  [SQL助手] [知识库] [智能体]  │
├──────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌──────────────────┐  │
│  │                     │  │                  │  │
│  │   聊天记录区         │  │   配置面板        │  │
│  │   （多轮对话、       │  │   API Key        │  │
│  │    历史记录          │  │   模式开关        │  │
│  │    支持Markdown）    │  │   清空按钮        │  │
│  │                     │  │                  │  │
│  └─────────────────────┘  └──────────────────┘  │
│  ┌────────────────────────────────────────────┐  │
│  │  输入框                              [发送] │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

三个标签页对应前五课的三种核心能力，用户在页面上点一下就切换，不用改代码。

---

## 三、准备工作

### 3.1 安装

```bash
pip install gradio openai
```

没错，就两个库。Gradio 自带 Web 服务器，不需要额外装 Flask 之类的东西。

### 3.2 项目结构

```
第六课_Gradio Web应用/
├── ai_web_app.py          ← 主程序（一个文件搞定）
└── README.md
```

对，就一个文件。这就是 Gradio 的威力——不需要 templates/、不需要 static/、不需要 config/，一个 `.py` 文件就是一个完整的 Web 应用。

---

## 四、核心代码拆解

别急着复制整段代码跑，先理解每一块在干嘛。

### 4.1 导入 & 配置

```python
import gradio as gr
from openai import OpenAI
import sqlite3
import os

# DeepSeek API 配置
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "你的Key写这里")
client = OpenAI(
    api_key=DEEPSEEK_KEY,
    base_url="https://api.deepseek.com",
)
```

### 4.2 SQL 助手函数

这就是第一课的升级版——接收自然语言，返回 SQL + 查询结果：

```python
def sql_assistant(question, history):
    """AI SQL 助手：自然语言 → SQL → 数据库查询"""
    # 先让AI生成SQL
    sql_prompt = f"用户问题：{question}\n请只输出SQL语句，不要解释："
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": sql_prompt}],
    )
    sql = response.choices[0].message.content.strip().lstrip("```sql").rstrip("```").strip()

    # 执行SQL（用 SQLite 内存数据库演示）
    try:
        conn = sqlite3.connect(":memory:")
        # 建示例表
        conn.execute("""CREATE TABLE sales (
            product TEXT, amount REAL, month TEXT
        )""")
        conn.executemany("INSERT INTO sales VALUES(?,?,?)", [
            ("蓝牙耳机", 15000, "2026-07"),
            ("无线键盘", 12000, "2026-07"),
            ("鼠标垫", 3000, "2026-08"),
        ])
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description] if cursor.description else []
        conn.close()

        result = f"**生成的SQL：**\n```sql\n{sql}\n```\n\n**查询结果：**\n"
        if rows:
            result += "| " + " | ".join(cols) + " |\n"
            result += "| " + " | ".join(["---"] * len(cols)) + " |\n"
            for row in rows:
                result += "| " + " | ".join(str(c) for c in row) + " |\n"
        else:
            result += "（无结果）"
        return result
    except Exception as e:
        return f"**生成的SQL：**\n```sql\n{sql}\n```\n\n❌ 执行出错：{e}"
```

关键点：返回的内容用了 Markdown 格式（表格、代码块），Gradio 会自动渲染。

### 4.3 知识库问答函数

第三课+第四课的 RAG 能力，包装成 Gradio 可调用的函数：

```python
# 初始化知识库（启动时运行一次）
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import tempfile

# 创建示例文档
KB_DIR = tempfile.mkdtemp()
docs_data = {
    "退货政策.txt": "蓝牙耳机：7天无理由退货需保留原包装。无线键盘：15天质量问题换新。",
    "保修政策.txt": "蓝牙耳机：1年质保。无线键盘：2年质保。手机支架：6个月质保。",
    "发货说明.txt": "下单后24小时内发货。江浙沪次日达，其他地区2-3天。满99包邮。",
}
for name, content in docs_data.items():
    with open(os.path.join(KB_DIR, name), "w", encoding="utf-8") as f:
        f.write(content)

# 建向量索引
all_docs = []
for f in os.listdir(KB_DIR):
    all_docs.extend(TextLoader(os.path.join(KB_DIR, f), encoding="utf-8").load())
chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30).split_documents(all_docs)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.from_documents(chunks, embeddings)


def kb_assistant(question, history):
    """RAG 知识库问答"""
    # 检索相关文档
    docs = vectorstore.similarity_search(question, k=2)
    context = "\n".join(d.page_content for d in docs)

    # 喂给 AI 回答问题
    prompt = f"""根据以下资料回答问题。如果资料中没有相关信息，就说"知识库中没有相关内容"。

资料：
{context}

问题：{question}"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content

    return f"**📚 检索到的资料：**\n> {context[:200]}...\n\n**🤖 AI 回答：**\n{answer}"
```

### 4.4 Agent 智能体函数

第五课的核心，包装成 Gradio 可调用：

```python
# Agent 工具定义（复用第五课的逻辑）
import json

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索公司知识库（退货/保修/发货政策）",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算，输入数学表达式",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

TOOL_FNS = {
    "search_knowledge": lambda query: vectorstore.similarity_search(query, k=2)[0].page_content,
    "calculate": lambda expression: str(eval(expression)),
}

SYSTEM_PROMPT = """你是智能助手。你可以：
1. search_knowledge - 查公司政策
2. calculate - 算数学
一次调用一个工具，信息全了再回答。"""


def agent_assistant(question, history):
    """Agent 智能体：AI 自主决定用什么工具"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 把历史也加进去（多轮对话上下文）
    for h in history[-5:]:  # 最近5轮
        messages.append({"role": "user", "content": h[0]})
        if h[1]:
            messages.append({"role": "assistant", "content": h[1]})

    messages.append({"role": "user", "content": question})

    # Agent 循环
    for _ in range(5):
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if msg.content and not msg.tool_calls:
            # 整合：如果有工具结果，加上来源说明
            has_tool = any(m["role"] == "tool" for m in messages)
            if has_tool:
                return f"{msg.content}\n\n---\n*（以上信息来自知识库检索，仅供参考）*"
            return msg.content

        if msg.tool_calls:
            tc = msg.tool_calls[0]
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            result = TOOL_FNS[name](**args)

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [{
                    "id": tc.id, "type": "function",
                    "function": {"name": name, "arguments": tc.function.arguments},
                }],
            })
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    return "抱歉，处理超时，请换个问法试试。"
```

### 4.5 搭建 Gradio 界面

重点来了——三栏布局+标签页切换：

```python
def build_ui():
    """构建 Gradio Web 界面"""
    with gr.Blocks(title="🤖 AI 智能助手", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🤖 AI 智能助手 —— 菜鸟进阶站出品
        ### 从 SQL 助手到 Agent 智能体，五节课的能力全在这一个页面里
        """)

        with gr.Tabs():
            # ====== Tab 1: SQL 助手 ======
            with gr.TabItem("📊 SQL 助手"):
                gr.Markdown("用自然语言提问，AI 自动生成 SQL 并查询数据库。示例：*上个月卖得最好的产品是什么？*")
                sql_chat = gr.ChatInterface(
                    fn=sql_assistant,
                    chatbot=gr.Chatbot(height=400, render_markdown=True),
                    textbox=gr.Textbox(placeholder="输入你的问题，比如：查询7月各产品的销售总额...", scale=7),
                )

            # ====== Tab 2: 知识库 ======
            with gr.TabItem("📚 知识库问答"):
                gr.Markdown("基于公司知识库回答问题。示例：*蓝牙耳机的退货政策是什么？*")
                kb_chat = gr.ChatInterface(
                    fn=kb_assistant,
                    chatbot=gr.Chatbot(height=400, render_markdown=True),
                    textbox=gr.Textbox(placeholder="输入你的问题，比如：退货政策是什么？", scale=7),
                )

            # ====== Tab 3: Agent 智能体 ======
            with gr.TabItem("🤖 Agent 智能体"):
                gr.Markdown("AI 自主决策：该查知识库还是直接回答。示例：*蓝牙耳机退货要几天？顺便帮我算 158*37*0.85*")
                agent_chat = gr.ChatInterface(
                    fn=agent_assistant,
                    chatbot=gr.Chatbot(height=400, render_markdown=True),
                    textbox=gr.Textbox(placeholder="随便问，AI 自己判断该用什么工具...", scale=7),
                )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",   # 允许局域网访问
        server_port=7860,         # 默认端口
        share=False,              # 改成 True 就生成公网链接
    )
```

---

## 五、启动！

### 5.1 本地运行

```bash
# 1. 装依赖
pip install gradio openai langchain-community faiss-cpu sentence-transformers

# 2. 设置 API Key
# Windows:
set DEEPSEEK_API_KEY=sk-你的Key
# Mac/Linux:
export DEEPSEEK_API_KEY=sk-你的Key

# 3. 启动
python ai_web_app.py
```

看到这个输出就说明成功了：

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live  ← 如果你设置了 share=True
```

浏览器会自动打开 `http://127.0.0.1:7860`，你就能看到一个漂亮的 Web 界面。

### 5.2 分享给同事

如果想让别人（不在你电脑前）也能用，把 `share=True`：

```python
app.launch(share=True)
```

Gradio 会自动生成一个公网链接（`https://xxxxx.gradio.live`），有效期 72 小时。把链接甩给同事，他们浏览器打开就能用——不需要装 Python、不需要配环境、不需要命令行。

> ⚠️ **安全提醒**：share=True 会把你的应用暴露到公网。不要在生产环境用这个方式，它只是个临时分享方案。

### 5.3 效果演示

三个标签页的效果：

**📊 SQL 助手**：
```
用户：上个月卖了多少蓝牙耳机？
AI：
  生成的SQL：
  SELECT SUM(amount) FROM sales WHERE product='蓝牙耳机' AND month='2026-07'
  
  查询结果：
  | SUM(amount) |
  |-------------|
  | 15000       |
→ 上个月蓝牙耳机卖了15000元。
```

**📚 知识库问答**：
```
用户：蓝牙耳机能退货吗？
AI：
  检索到的资料：蓝牙耳机：7天无理由退货需保留原包装...
  
  AI回答：可以退货，但需要在7天内申请，且必须保留原包装。
```

**🤖 Agent 智能体**：
```
用户：蓝牙耳机退货要几天？顺便帮我算一下 158*37*0.85
AI：
  → [自动调用 search_knowledge]
  → [自动调用 calculate]
  
  蓝牙耳机支持7天无理由退货，需要保留原包装。
  158 × 37 × 0.85 = 4969.1
  （AI 自动判断了先用哪个工具、后查什么，你一条消息两个问题全搞定）
```

---

## 六、进阶：加一个「设置」面板

把 API Key 的配置也放到网页上，用户不用改代码：

```python
def build_ui_with_settings():
    """带设置面板的版本"""
    with gr.Blocks(title="🤖 AI 智能助手", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("# 🤖 AI 智能助手")

        # API Key 设置区
        with gr.Accordion("⚙️ 设置", open=False):
            api_key = gr.Textbox(
                label="DeepSeek API Key",
                type="password",
                placeholder="sk-...",
                value=os.getenv("DEEPSEEK_API_KEY", ""),
            )
            gr.Markdown("> 不填就用环境变量 `DEEPSEEK_API_KEY`。Key 只在本地保存，不会上传。")

        with gr.Tabs():
            with gr.TabItem("📊 SQL 助手"):
                gr.ChatInterface(
                    fn=lambda msg, hist: sql_assistant_with_key(msg, hist, api_key.value),
                    chatbot=gr.Chatbot(height=400, render_markdown=True),
                )
            # ... 其他标签页同理

    return app
```

这样用户打开网页就能输入自己的 API Key，不用碰代码。

---

## 七、Gradio 常用技巧

### 7.1 美化主题

```python
# 内置主题
gr.themes.Soft()       # 圆角、柔和
gr.themes.Monochrome() # 黑白简约
gr.themes.Glass()      # 毛玻璃效果

# 自定义颜色
gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
)
```

### 7.2 阻止用户乱点

```python
# 提交按钮 — 处理时禁用，防止重复点击
btn = gr.Button("发送")
btn.click(fn=handle, inputs=..., outputs=...)

# 或
gr.ChatInterface(
    fn=handle,
    submit_btn="🚀 发送",
    stop_btn="⏹ 停止",
    retry_btn="🔄 重试",
)
```

### 7.3 文件上传

```python
def process_file(file):
    # file 是一个临时文件路径
    with open(file.name, "r", encoding="utf-8") as f:
        content = f.read()
    return f"文件内容：{content[:500]}..."

gr.Interface(
    fn=process_file,
    inputs=gr.File(label="上传文档"),
    outputs="text",
).launch()
```

### 7.4 多轮对话记忆

`gr.ChatInterface` 默认就支持多轮对话，`history` 参数自动维护。如果你想手动控制：

```python
def chat_fn(message, history):
    # history 是一个 list of [user_msg, bot_msg]
    full_context = "\n".join([f"用户：{h[0]}\n助手：{h[1]}" for h in history[-5:]])
    # ... 用 full_context 做上下文
    return "AI的回答"
```

---

## 八、完整代码

（完整代码见配套源码 `ai_web_app.py`，包含：SQL 助手 + 知识库问答 + Agent 智能体 + 设置面板，200 行搞定）

核心数据流：

```
浏览器                                    Python
  │                                         │
  │  用户输入 "退货政策"                      │
  │ ──────────────────────────────────────→ │
  │        (Gradio 自动 HTTP POST)            │
  │                                         │ 调用 kb_assistant()
  │                                         │   → LangChain 检索
  │                                         │   → DeepSeek API 生成答案
  │                                         │
  │  Markdown 格式的答案                     │
  │ ←────────────────────────────────────── │
  │     (Gradio 自动渲染)                     │
```

不需要写一行 HTML，不需要配置路由，不需要处理 JSON 序列化——Gradio 全帮你干了。

---

## 九、部署到公网（免费）

临时分享用 `share=True` 就够。但如果要长期用，有两个免费方案：

### 方案 A：CloudStudio（推荐）

直接把代码文件夹部署到 CloudStudio 沙箱，自动生成一个永久链接。

### 方案 B：HuggingFace Spaces

1. 去 huggingface.co 注册，创建一个 Space
2. 选择 Gradio SDK
3. 把 `ai_web_app.py` 改名为 `app.py`，上传
4. 自动部署，得到一个 `https://你的用户名-空间名.hf.space` 的永久链接

---

## 十、总结 & 下期预告

### 这节课你学到了

| 知识点 | 一句话 |
|--------|--------|
| Gradio 是什么 | Python 写 Web 界面，三行代码出网页 |
| gr.ChatInterface | 自带聊天记录的开箱即用聊天组件 |
| gr.Tabs | 多标签页布局，一个页面集成多种功能 |
| gr.Markdown | 在 Gradio 里渲染 Markdown 表格和代码块 |
| share=True | 一键生成公网分享链接 |
| GrChatbot(render_markdown=True) | 支持表格、代码高亮、列表等富文本 |

### 和前五课的关系

```
第一~三课：手写核心逻辑（理解原理）
第四~五课：用框架精简代码（学会工具）
第六课：   装进网页给人用（做出产品）  ← 你在这
```

从第一课的一行 API 调用，到第六课的一个完整 Web 应用——**六节课，从调通大模型到上线产品。**

### 下期预告

**第七课：多模态实战——让AI看图说话**

> 文字玩够了，该让 AI 看图片了。下节课用 DeepSeek 的多模态能力（或其他视觉模型），做一个「上传截图自动生成分析报告」的工具。
>
> 图片识别 + AI 推理 + Gradio 前端 = 一个真正的生产力工具。

---

### 📦 完整代码获取

本文所有代码已上传 GitHub，关注本公众号，后台回复 **「AI从入门到入土」** 即可获取源码仓库地址。

---

*作者：菜鸟进阶站*
*一个立志做出教科版笔记的年轻人*
*本文首发于微信公众号「菜鸟进阶站」，转载请联系授权*

> 系列合集：#AI从入门到入土
>
> 上一篇：第五课《Agent智能体实战——让AI自己决定先干嘛》
>
> 下一篇：第七课《多模态实战——让AI看图说话》
