"""
【可直接运行】SQL 智能助手 —— 支持多轮对话
运行前请：
1. pip install openai
2. 把 DEEPSEEK_API_KEY 替换成你自己的 Key
"""
from openai import OpenAI

# ==================== 改这里！====================
DEEPSEEK_API_KEY = "sk-你的API-Key"
# =================================================

TABLE_SCHEMA = """
-- 订单表 orders
字段: order_id(BIGINT,订单ID), product_id(BIGINT,商品ID), 
      product_name(VARCHAR,商品名称), category(VARCHAR,商品类目),
      sale_amount(DECIMAL,销售额), quantity(INT,销售数量),
      order_date(DATE,订单日期), customer_id(BIGINT,客户ID), 
      region(VARCHAR,地区)

-- 客户表 customers  
字段: customer_id(BIGINT,客户ID), customer_name(VARCHAR,客户名称),
      customer_level(VARCHAR,客户等级:VIP/Gold/Silver/Normal),
      register_date(DATE,注册日期)
"""

SYSTEM_PROMPT = f"""
你是一个SQL智能助手，服务于数据工程师。你的能力包括：

1. 根据自然语言需求生成SQL语句
2. 解释复杂SQL的业务逻辑
3. 提供SQL优化建议

当前可用的数据表：
{TABLE_SCHEMA}

规则：
- 生成SQL时，只输出SQL代码，不要多余解释
- 解释SQL时，用通俗语言，假设读者只有基础SQL知识
- 优化建议要具体，给出改写前后的对比
"""

class SQLAssistant:
    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def chat(self, user_input):
        """发送消息并获取回复"""
        self.history.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=self.history
        )

        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        """重置对话"""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]


def main():
    assistant = SQLAssistant()

    print("=" * 60)
    print("  SQL 智能助手已就绪")
    print("  输入 'help' 查看示例 | 输入 'reset' 重置 | 输入 'quit' 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你：").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("\n再见！")
                break

            if user_input.lower() == 'reset':
                assistant.reset()
                print("\n对话已重置")
                continue

            if user_input.lower() == 'help':
                print("""
  你可以这样问我：

  【写SQL】
  - "查询上个月每个品类的销售额汇总"
  - "找出过去半年内没有下过单的客户"
  - "统计每个地区VIP客户的月均消费金额"

  【解释SQL】
  - "帮我解释这段SQL：[粘贴SQL]"
  - "这段代码的业务含义是什么：[粘贴SQL]"

  【优化SQL】
  - "帮我优化这段SQL：[粘贴SQL]"
  - "这段SQL有没有性能问题：[粘贴SQL]"
                """)
                continue

            print("\nAI：", end="", flush=True)
            reply = assistant.chat(user_input)
            print(reply)

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n出错了：{e}")


if __name__ == "__main__":
    main()
