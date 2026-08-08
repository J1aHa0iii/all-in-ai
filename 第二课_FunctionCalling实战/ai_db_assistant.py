"""
AI 数据库助手 —— Function Calling 实战
第二课完整代码。先运行 init_db.py 创建数据库，再运行本文件。

使用方法：
    python ai_db_assistant.py

对话示例：
    你：数据库里有哪些表？
    你：上个月销售额最高的3个产品是什么？
    你：华东地区销售排名？
    你：张三卖了多少货？
    你：退出
"""
import sqlite3
import json
import os
from openai import OpenAI

# ==================== 配置 ====================

DEEPSEEK_API_KEY = "sk-your-deepseek-api-key"  # ← 改成你的 Key
DB_PATH = "company.db"

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

# ==================== 工具函数 ====================

def query_database(sql: str) -> str:
    """执行 SQL 查询并返回格式化结果"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)

        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        else:
            conn.commit()
            conn.close()
            return "✅ SQL 执行成功"

        conn.close()

        if not rows:
            return "查询结果为空"

        result = f"共 {len(rows)} 条记录\n"
        result += " | ".join(columns) + "\n"
        result += "-" * 50 + "\n"
        for row in rows[:20]:
            result += " | ".join(str(v) for v in row) + "\n"

        if len(rows) > 20:
            result += f"... (还有 {len(rows) - 20} 条记录)"

        return result

    except Exception as e:
        return f"SQL 执行出错：{str(e)}"


def get_table_info(table_name: str = "") -> str:
    """获取数据库表结构"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        if table_name:
            cursor.execute(f"PRAGMA table_info({table_name})")
            cols = cursor.fetchall()
            conn.close()
            if not cols:
                return f"表 {table_name} 不存在"
            return "\n".join([f"  {c[1]} ({c[2]})" for c in cols])

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = cursor.fetchall()

        if not tables:
            conn.close()
            return "数据库中没有表"

        all_info = []
        for (tname,) in tables:
            cursor.execute(f"PRAGMA table_info({tname})")
            cols = cursor.fetchall()
            col_info = [f"    {c[1]} ({c[2]})" for c in cols]
            all_info.append(f"表名：{tname}\n" + "\n".join(col_info))

        conn.close()
        return "\n\n".join(all_info)

    except Exception as e:
        return f"获取表结构出错：{str(e)}"


# ==================== 工具注册 ====================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "执行SQL查询语句并返回结果。当用户想查询数据库中的数据时使用。注意：金额需用 amount * quantity 计算。",
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
            "description": "获取数据库表结构。当需要了解表有哪些字段、字段类型时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "表名（可选，不填返回所有表）",
                    }
                },
                "required": [],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "query_database": query_database,
    "get_table_info": get_table_info,
}

SYSTEM_PROMPT = """你是一个数据分析助手，连接着公司销售数据库。

数据库 sales 表字段：product_name(产品名), category(品类), amount(单价), quantity(数量), sale_date(日期YYYY-MM-DD), region(地区), salesperson(销售员)。

规则：
- 查询前先了解表结构
- 销售额 = amount * quantity
- 回答简洁清晰，用中文
- 如果 SQL 出错，分析原因并重试"""


# ==================== 核心循环 ====================

def chat(user_message: str, messages: list) -> str:
    """处理一条用户消息，返回 AI 回复"""
    messages.append({"role": "user", "content": user_message})

    for _ in range(10):  # 最多10轮工具调用
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # AI 想调工具
        if msg.tool_calls:
            messages.append(msg)

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                print(f"\n  🔧 {name}({json.dumps(args, ensure_ascii=False)})")

                func = AVAILABLE_FUNCTIONS.get(name)
                result = func(**args) if func else f"未知工具：{name}"
                print(f"  📋 {result[:120]}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue

        # AI 直接回复
        messages.append({"role": "assistant", "content": msg.content})
        return msg.content

    return "⚠️ 达到最大轮次。"


# ==================== 启动 ====================

if __name__ == "__main__":
    if "your-deepseek" in DEEPSEEK_API_KEY:
        print("⚠️ 请先在代码中设置你的 DEEPSEEK_API_KEY！")
        exit(1)

    if not os.path.exists(DB_PATH):
        print("⚠️ 数据库不存在，请先运行 init_db.py 创建示例数据！")
        exit(1)

    print("=" * 55)
    print("🤖 AI 数据库助手 v2.0 | Function Calling 实战")
    print("   你可以问：销售额排行 / 地区分析 / 某人业绩 / ...")
    print("   输入 quit 退出")
    print("=" * 55)

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user = input("\n👤 你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if user.lower() in ("quit", "exit", "退出", "q"):
            print("👋 再见！")
            break
        if not user:
            continue

        answer = chat(user, history)
        print(f"\n🤖 AI：{answer}")
