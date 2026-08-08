"""
【可直接运行】实战一：AI 自动生成 SQL
运行前请：
1. pip install openai
2. 把 DEEPSEEK_API_KEY 替换成你自己的 Key
"""
from openai import OpenAI

# ==================== 改这里！====================
DEEPSEEK_API_KEY = "sk-你的API-Key"
# =================================================

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

system_prompt = """
你是一个资深的SQL专家。你需要根据用户的自然语言需求，生成对应的SQL语句。

当前数据库的表结构如下：

-- 订单表
CREATE TABLE orders (
    order_id        BIGINT PRIMARY KEY COMMENT '订单ID',
    product_id      BIGINT COMMENT '商品ID',
    product_name    VARCHAR(200) COMMENT '商品名称',
    category        VARCHAR(50) COMMENT '商品类目',
    sale_amount     DECIMAL(15,2) COMMENT '销售额',
    quantity        INT COMMENT '销售数量',
    order_date      DATE COMMENT '订单日期',
    customer_id     BIGINT COMMENT '客户ID',
    region          VARCHAR(50) COMMENT '地区'
);

-- 客户表
CREATE TABLE customers (
    customer_id     BIGINT PRIMARY KEY COMMENT '客户ID',
    customer_name   VARCHAR(100) COMMENT '客户名称',
    customer_level  VARCHAR(20) COMMENT '客户等级: VIP/Gold/Silver/Normal',
    register_date   DATE COMMENT '注册日期'
);

要求：
1. 只输出SQL语句，不要解释
2. SQL要兼容 MySQL 语法
3. 使用中文别名
4. 注意处理 NULL 值
"""

# 用户的问题
user_question = "帮我查一下上个月销售额最高的10个商品，按销售额降序排列"

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
)

print("=" * 60)
print("用户提问：", user_question)
print("=" * 60)
print("\nAI 生成的 SQL：\n")
print(response.choices[0].message.content)
