# 10块钱入门大模型第二课：让AI帮你查数据库——Function Calling实战（保姆级教程）

> 阅读本文需要：第一课的基础 + 一台电脑 + 想进阶的心
> 
> 系列合集：#AI从入门到入土

---

## 一、上回说到……

第一课里，我们 5 行代码调通了 DeepSeek，写出了一个能帮我们「写 SQL」的 AI 助手。

但你有没有觉得哪里不够爽？

你跟 AI 说：「帮我查一下上个月销售额最高的 10 个产品。」

AI 给你回了一段 SQL：

```sql
SELECT product_name, SUM(amount) as total
FROM sales
WHERE sale_date >= '2026-07-01' AND sale_date < '2026-08-01'
GROUP BY product_name
ORDER BY total DESC
LIMIT 10;
```

然后呢？**你还得自己复制粘贴到数据库客户端去跑。**

能不能让对话变成这样——

> 你：帮我查一下上个月销售额最高的 10 个产品
> 
> AI：好的，正在查询…… 查到了！上个月销售 Top 10 如下：
> 1. iPhone 16 —— ¥2,380,000
> 2. MacBook Pro —— ¥1,950,000
> ……

**这就是今天的主题：Function Calling —— 给 AI 装上「手」，让它真的能「干活」。**

---

## 二、Function Calling 是个啥？用大白话解释

先别被术语吓到。我用一个比喻：

> **第一课的 AI 像一个只会说话的顾问**——你问它怎么写 SQL，它告诉你了，但没法帮你执行。
> 
> **Function Calling 就是给这个顾问配了一个「工具箱」**——你告诉它：「工具箱里有一个叫 `query_database` 的工具，你把 SQL 丢进去就能拿到结果。」
> 
> 然后它就会自动判断：用户问的问题需要用哪个工具？参数怎么填？什么时候该去查数据库？

画个流程图你就明白了：

```
用户：上月销售额 Top 10 产品？
        ↓
    AI 分析意图
        ↓
  「这需要查数据库！」
        ↓
  AI 自动调用 query_database(sql="SELECT ...")
        ↓
  数据库返回结果 → AI 理解结果
        ↓
  AI 用自然语言回答：「查到了，Top 1 是 iPhone 16...」
```

**整个过程你不用写一行判断逻辑，AI 自己决定什么时候该调哪个函数。**

> DeepSeek 的 API 完全兼容 OpenAI 的 Function Calling 规范，所以学会了这一招，以后切换到 GPT-4、Claude 也一样用。

---

## 三、准备工作（3 分钟）

本课只需要两个东西：

```bash
pip install openai
```

SQLite —— Python 自带，不需要额外安装。

> 为什么用 SQLite？因为它是 Python 内置的，零配置、零安装。你学完以后换成 MySQL、Oracle、PostgreSQL 只需要改一行连接代码。

DeepSeek API Key 沿用第一课的就行，确保余额充足（10 块钱够玩一个月了）。

---

## 四、第一步：创建一个示例数据库

我们先造一个销售数据库，让 AI 有东西可查：

```python
import sqlite3

# 创建数据库和表
conn = sqlite3.connect("company.db")
cursor = conn.cursor()

# 销售表
cursor.execute("""
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    amount REAL NOT NULL,
    quantity INTEGER NOT NULL,
    sale_date TEXT NOT NULL,
    region TEXT NOT NULL,
    salesperson TEXT NOT NULL
)
""")

# 插入 20 条示例数据
sample_data = [
    ("iPhone 16", "手机", 7999, 120, "2026-07-01", "华东", "张三"),
    ("MacBook Pro", "电脑", 14999, 45, "2026-07-02", "华东", "李四"),
    ("iPad Air", "平板", 4999, 80, "2026-07-03", "华南", "王五"),
    ("AirPods Pro", "耳机", 1899, 200, "2026-07-05", "华北", "张三"),
    ("Apple Watch", "穿戴", 3199, 65, "2026-07-06", "华东", "赵六"),
    ("iPhone 16", "手机", 7999, 150, "2026-07-08", "华南", "李四"),
    ("MacBook Pro", "电脑", 14999, 38, "2026-07-10", "华北", "王五"),
    ("iPad Air", "平板", 4999, 55, "2026-07-12", "华东", "张三"),
    ("AirPods Pro", "耳机", 1899, 175, "2026-07-14", "华南", "赵六"),
    ("Apple Watch", "穿戴", 3199, 50, "2026-07-15", "华北", "李四"),
    ("iPhone 16", "手机", 7999, 100, "2026-07-16", "华东", "王五"),
    ("MacBook Pro", "电脑", 14999, 42, "2026-07-18", "华南", "张三"),
    ("iPad Air", "平板", 4999, 70, "2026-07-20", "华北", "赵六"),
    ("AirPods Pro", "耳机", 1899, 190, "2026-07-22", "华东", "李四"),
    ("Apple Watch", "穿戴", 3199, 55, "2026-07-24", "华南", "王五"),
    ("iPhone 16", "手机", 7999, 130, "2026-07-25", "华北", "张三"),
    ("MacBook Pro", "电脑", 14999, 35, "2026-07-27", "华东", "赵六"),
    ("iPad Air", "平板", 4999, 60, "2026-07-28", "华南", "李四"),
    ("AirPods Pro", "耳机", 1899, 210, "2026-07-29", "华北", "王五"),
    ("Apple Watch", "穿戴", 3199, 45, "2026-07-30", "华东", "张三"),
]

cursor.executemany(
    "INSERT INTO sales (product_name, category, amount, quantity, sale_date, region, salesperson) VALUES (?, ?, ?, ?, ?, ?, ?)",
    sample_data
)

conn.commit()
print("✅ 数据库创建成功！共", cursor.rowcount, "条销售记录")
conn.close()
```

跑一下，我们的公司销售数据库就有了。20 条记录，5 个产品，4 个地区，6 个销售。

---

## 五、第二步：定义 AI 能用的「工具」

Function Calling 的核心就是给 AI 注册一个或多个函数，告诉它：「这些函数你可以用，这是它们的用途、参数说明。」

我们先定义三个工具：

```python
import sqlite3
import json

# ==================== 工具箱 ====================

DB_PATH = "company.db"


def query_database(sql: str) -> str:
    """
    执行 SQL 查询并返回结果。
    这是 AI 最主要的工具 —— 你给它一段 SQL，它帮你跑，然后把结果返回给 AI。
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)

        # 获取列名
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()

        conn.close()

        if not rows:
            return "查询结果为空"

        # 格式化返回结果
        result = f"共 {len(rows)} 条记录\n"
        result += " | ".join(columns) + "\n"
        result += "-" * 50 + "\n"
        for row in rows[:20]:  # 最多返回20条，避免太长
            result += " | ".join(str(v) for v in row) + "\n"

        if len(rows) > 20:
            result += f"... (还有 {len(rows) - 20} 条记录未显示)"

        return result

    except Exception as e:
        return f"SQL 执行出错：{str(e)}"


def get_table_info(table_name: str = "") -> str:
    """
    获取数据库中所有表的结构信息，让 AI 知道有哪些表、哪些字段。
    AI 不知道你的数据库长什么样，所以需要这个函数告诉它。
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        if table_name:
            cursor.execute(f"PRAGMA table_info({table_name})")
        else:
            # 先获取所有表名
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = cursor.fetchall()

            if not tables:
                conn.close()
                return "数据库中没有任何表"

            all_info = []
            for (tname,) in tables:
                cursor.execute(f"PRAGMA table_info({tname})")
                cols = cursor.fetchall()
                col_info = [f"    {c[1]} ({c[2]})" for c in cols]
                all_info.append(f"表名：{tname}\n" + "\n".join(col_info))

            conn.close()
            return "\n\n".join(all_info)

        conn.close()
    except Exception as e:
        return f"获取表结构出错：{str(e)}"


def analyze_data(analysis_request: str) -> str:
    """
    对查询结果进行分析和总结。
    有时候 AI 拿到原始数据后还想做进一步分析时用。
    """
    # 这个函数实际上是让 AI 拿到数据后自己总结
    # 在 Function Calling 流程中，AI 会先调 query_database 拿数据，然后自己分析
    return f"收到分析请求：{analysis_request}。请基于已查询的数据进行分析。"
```

---

## 六、第三步：给 DeepSeek「注册」这些工具

重点来了！我们需要用特定的 JSON 格式告诉 DeepSeek 我们有哪些工具可用：

```python
# 工具定义（OpenAI Function Calling 标准格式）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "执行SQL查询语句，返回查询结果。当用户想查询数据库中的数据时使用此函数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "要执行的SQL查询语句",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_info",
            "description": "获取数据库中的表结构和字段信息。当需要了解数据库中有哪些表、每个表有哪些字段时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "要查看的表名（可选，不填则返回所有表结构）",
                    }
                },
                "required": [],
            },
        },
    },
]

# 函数名 → 实际 Python 函数的映射
AVAILABLE_FUNCTIONS = {
    "query_database": query_database,
    "get_table_info": get_table_info,
}
```

这里的 JSON 有几个要点：

- `name`：函数名，AI 会输出这个名字来「调用」它
- `description`：**这个最重要！** AI 靠它来判断什么时候该用哪个函数。写清楚、写具体
- `parameters`：告诉 AI 这个函数需要什么参数，每个参数是什么意思

---

## 七、第四步：实现 Function Calling 循环

这是整个第二课最核心的部分 —— 一个完整的对话循环：

```python
from openai import OpenAI
import json

client = OpenAI(
    api_key="sk-your-deepseek-api-key",  # 替换成你的 Key
    base_url="https://api.deepseek.com",
)

# 系统提示词 —— 告诉 AI 它的角色和能力
SYSTEM_PROMPT = """你是一个数据分析助手，连接着一个公司销售数据库。

数据库中有这些表：
- sales（销售记录表）：product_name(产品名), category(品类), amount(单价), quantity(数量), sale_date(日期), region(地区), salesperson(销售员)

你可以使用以下工具：
1. get_table_info：查看表结构
2. query_database：执行 SQL 查询

规则：
- 查询前先用 get_table_info 了解表结构
- 金额计算用 amount * quantity
- 回答要简洁清晰，用中文
"""


def run_agent(user_message: str):
    """
    核心循环：处理一条用户消息，返回 AI 的回复。
    可能会经过多轮「AI 要调工具 → 调工具返回结果 → AI 理解结果继续...」
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    max_rounds = 10  # 安全阀：最多调 10 次工具
    round_count = 0

    while round_count < max_rounds:
        round_count += 1

        # 调用 DeepSeek
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",  # 让 AI 自己决定要不要调工具
        )

        choice = response.choices[0]
        msg = choice.message

        # ===== 情况1：AI 想调工具 =====
        if msg.tool_calls:
            # 把 AI 的「工具调用请求」加入对话历史
            messages.append(msg)

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"\n🔧 AI 调用工具：{func_name}")
                print(f"   参数：{json.dumps(func_args, ensure_ascii=False)}")

                # 执行对应的 Python 函数
                func = AVAILABLE_FUNCTIONS.get(func_name)
                if func:
                    result = func(**func_args)
                else:
                    result = f"未知工具：{func_name}"

                print(f"   结果：{result[:100]}...")

                # 把工具返回结果加入对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # 继续循环，让 AI 基于工具返回的结果继续思考
            continue

        # ===== 情况2：AI 直接回复（不调工具） =====
        return msg.content

    return "⚠️ 达到最大对话轮次，请换个方式提问。"


# ==================== 测试 ====================

if __name__ == "__main__":
    # 先让 AI 了解数据库结构
    print("=" * 60)
    print("🤖 AI 数据库助手启动！输入 quit 退出")
    print("=" * 60)

    while True:
        user_input = input("\n👤 你：")
        if user_input.lower() in ("quit", "exit", "退出"):
            print("👋 再见！")
            break

        print("\n🤖 AI 思考中...")
        answer = run_agent(user_input)
        print(f"\n🤖 AI：{answer}")
```

---

## 八、跑起来试试效果

让我们来几轮真实对话，看看 AI 的表现：

```
👤 你：数据库里有哪些表？

🔧 AI 调用工具：get_table_info
   参数：{}
   结果：表名：sales
    id (INTEGER)
    product_name (TEXT)
    ...

🤖 AI：数据库中有一个 sales 表，包含以下字段：
- id：序号
- product_name：产品名称
- category：产品品类
- amount：单价
- quantity：数量
- sale_date：销售日期
- region：销售地区
- salesperson：销售员
```

```
👤 你：帮我查一下上个月销售额最高的3个产品

🔧 AI 调用工具：query_database
   参数：{"sql": "SELECT product_name, SUM(amount * quantity) as total_sales FROM sales WHERE sale_date >= '2026-07-01' AND sale_date < '2026-08-01' GROUP BY product_name ORDER BY total_sales DESC LIMIT 3"}
   结果：3 条记录
   product_name | total_sales
   iPhone 16 | 3999500.0
   AirPods Pro | 1461000.0
   MacBook Pro | 1295000.0

🤖 AI：上个月销售额 Top 3 如下：
1. 🥇 iPhone 16 —— ¥3,999,500
2. 🥈 AirPods Pro —— ¥1,461,000
3. 🥉 MacBook Pro —— ¥1,295,000
```

```
👤 你：华东地区的销售情况怎么样？

🔧 AI 调用工具：query_database
   参数：{"sql": "SELECT product_name, SUM(amount * quantity) as total, COUNT(*) as order_count FROM sales WHERE region = '华东' AND sale_date >= '2026-07-01' GROUP BY product_name ORDER BY total DESC"}
   结果：...

🤖 AI：华东地区 7 月份销售概况：
- 总订单数：XX 笔
- 热销产品依次为...
```

**完全不用写任何判断逻辑，AI 自己理解了用户意图，自己写 SQL，自己调用函数，自己整理结果。**

---

## 九、进阶扩展：加一个「数据导出」工具

学到这里，你肯定已经想到了——能不能再加个工具，让 AI 把查询结果导出成 CSV？

```python
def export_to_csv(sql: str, filename: str) -> str:
    """将 SQL 查询结果导出为 CSV 文件"""
    import csv
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(sql)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    
    conn.close()
    return f"✅ 已导出 {len(rows)} 条数据到 {filename}"
```

然后把它也注册到 `TOOLS` 和 `AVAILABLE_FUNCTIONS` 里，AI 就自动会用了：

```
👤 你：把华东地区的销售数据导出成 CSV

🤖 AI：好的，已导出到 east_sales.csv，共 15 条记录。
```

---

## 十、总结 & 下期预告

### 这节课你学到了什么？

| 知识点 | 一句话总结 |
|--------|----------|
| Function Calling | 给 AI 注册函数，让它能「动手」干活 |
| 工具定义 | 用 JSON 描述函数名、用途、参数 |
| 对话循环 | AI 调工具 → 返回结果 → 继续思考 → 最终回复 |
| tool_choice="auto" | AI 自己判断要不要调工具 |

### 你现在的 AI 助手能做什么？

- ✅ 理解自然语言查询需求
- ✅ 自己写 SQL
- ✅ 自己执行 SQL 拿结果
- ✅ 用自然语言总结结果
- ✅ 支持多轮对话、追问

### 下期预告

**第三课：RAG 实战 —— 让 AI 读你的文档，变身企业知识库专家**

> 你有没有遇到过这种情况：公司有一堆技术文档、操作手册，新人天天来问你问题？
> 
> 第三课我们就要做一个「公司内部 AI 问答机器人」——把你几十份 PDF、Word 文档喂给 AI，让它自动回答同事的各种问题。
> 
> 涉及技术：文本向量化、相似度检索、LangChain 入门。

---

> **本文源码已上传 GitHub，关注公众号「菜鸟进阶站」回复「AI从入门到入土」获取系列全部源码。**
> 
> 系列合集：#AI从入门到入土
> 
> 下一篇：第三课《RAG实战：把200份文档喂给AI，打造企业知识库》
