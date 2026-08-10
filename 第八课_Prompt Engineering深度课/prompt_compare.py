"""prompt_compare.py —— Prompt 对比测试器

直观感受不同质量的 Prompt 对 AI 输出的影响。
同一个问题、三个 Prompt、三份输出、并列对比。

用法：
    python prompt_compare.py
    python prompt_compare.py --question "你的问题"
    python prompt_compare.py --interactive    # 交互模式：自己写 Prompt 对比

依赖：pip install openai
"""

import os
import sys
import json
from openai import OpenAI

# ==================== 配置 ====================
API_KEY = os.getenv("DEEPSEEK_API_KEY", "你的DeepSeek Key填这里")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==================== 默认测试问题 ====================
DEFAULT_QUESTION = "查询每个产品类别的总销售额，按销售额从高到低排列"
DEFAULT_SCHEMA = """
sales 表：
- product_name VARCHAR(100)    -- 产品名称
- category VARCHAR(50)          -- 产品类别
- amount DECIMAL(10,2)          -- 销售金额
- quantity INT                  -- 销售数量
- order_date DATE               -- 订单日期
- customer_name VARCHAR(50)     -- 客户名称
"""

# ==================== 三个 Prompt 模板 ====================
PROMPT_TEMPLATES = {
    "随便写（对照组）": lambda q, s: f"写个SQL：{q}\n表结构：{s}",

    "有角色（基本质量）": lambda q, s: (
        f"你是资深SQL工程师。根据表结构写SQL：\n{s}\n问题：{q}"
    ),

    "完整三层 + Few-shot（最佳实践）": lambda q, s: f"""你是10年经验的Oracle DBA，擅长复杂SQL和性能优化。

## 表结构
{s}

## 问题
{q}

## 规范
- 使用 Oracle 语法（FETCH FIRST 代替 LIMIT）
- 聚合查询加别名
- 金额字段加 ROUND(..., 2) 保留两位小数
- 如果表结构中没有需要的字段，请明确指出

## 示例
输入：查询2024年每个月的订单数
```sql
SELECT TO_CHAR(order_date, 'YYYY-MM') AS month,
       COUNT(*) AS order_count
FROM sales
WHERE order_date >= DATE '2024-01-01'
GROUP BY TO_CHAR(order_date, 'YYYY-MM')
ORDER BY month;
-- 按月份汇总订单数，仅统计2024年数据
```

现在输出SQL："""
}


def run_comparison(question, schema, show_raw=False):
    """并行运行三个 Prompt，对比输出"""
    results = {}

    for name, template_func in PROMPT_TEMPLATES.items():
        prompt = template_func(question, schema)

        if show_raw:
            print(f"\n{'─'*60}")
            print(f"📤 【{name}】发送的 Prompt：")
            print(f"{'─'*60}")
            print(prompt[:500] + ("..." if len(prompt) > 500 else ""))

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        results[name] = {
            "prompt": prompt,
            "output": resp.choices[0].message.content,
            "tokens": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total": resp.usage.total_tokens
            }
        }

    return results


def print_results(results, question):
    """格式化打印对比结果"""
    print()
    print("=" * 70)
    print(f"🔍 测试问题：{question}")
    print("=" * 70)

    for idx, (name, data) in enumerate(results.items(), 1):
        tokens = data["tokens"]
        output = data["output"]

        print(f"\n{'─'*70}")
        print(f" 方案{idx}：{name}")
        print(f" Token消耗：提示词 {tokens['prompt_tokens']} + 回复 {tokens['completion_tokens']} = {tokens['total']}")
        print(f"{'─'*70}")
        print(output)

    # 总结
    print(f"\n{'='*70}")
    print("📊 对比总结")
    print(f"{'='*70}")
    print(f"{'方案':<20} {'Prompt Token':>12} {'回复Token':>10} {'总Token':>8}")
    for name, data in results.items():
        t = data["tokens"]
        print(f"{name:<20} {t['prompt_tokens']:>12} {t['completion_tokens']:>10} {t['total']:>8}")

    print(f"\n💡 结论：更好的 Prompt 消耗更多输入 token，但输出质量明显提升。")
    print(f"   多花的几十个 token（不到 1 分钱）换来的是可直接使用的正确结果。")


def interactive_mode():
    """交互模式：自己写两个 Prompt 对比"""
    print("\n📝 交互模式：对比你自己的两个 Prompt")
    print("=" * 60)

    question = input("\n输入问题（回车用默认）：").strip()
    if not question:
        question = DEFAULT_QUESTION

    schema = input("\n输入表结构或上下文（回车用默认）：").strip()
    if not schema:
        schema = DEFAULT_SCHEMA

    print("\n现在写两个 Prompt，回车后 AI 会同时调用，对比输出。\n")

    prompt_a = input("Prompt A（随便写版）：").strip()
    prompt_b = input("Prompt B（精心优化版）：").strip()

    if not prompt_a or not prompt_b:
        print("⚠️ 两个 Prompt 都不能为空！")
        return

    custom_templates = {
        "Prompt A（你的）": lambda q, s: prompt_a.replace("{question}", q).replace("{schema}", s),
        "Prompt B（你的）": lambda q, s: prompt_b.replace("{question}", q).replace("{schema}", s),
    }

    global PROMPT_TEMPLATES
    original = PROMPT_TEMPLATES
    PROMPT_TEMPLATES = custom_templates

    results = run_comparison(question, schema)
    print_results(results, question)

    PROMPT_TEMPLATES = original


def main():
    print("=" * 60)
    print("🔬 Prompt 对比测试器")
    print("=" * 60)

    # 解析命令行参数
    show_raw = "--raw" in sys.argv

    if "--interactive" in sys.argv:
        interactive_mode()
        return

    # 自定义问题
    if "--question" in sys.argv:
        idx = sys.argv.index("--question")
        question = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else DEFAULT_QUESTION
    else:
        question = DEFAULT_QUESTION

    print(f"\n📋 测试问题：{question}")
    print(f"⏳ 正在调用 DeepSeek API（3次并行请求）……")

    try:
        results = run_comparison(question, DEFAULT_SCHEMA, show_raw=show_raw)
        print_results(results, question)

    except Exception as e:
        print(f"\n❌ 出错了：{e}")
        print("\n💡 常见原因：")
        print("  1. API Key 没填或填错了 → 修改脚本第 15 行的 API_KEY")
        print("  2. 网络问题 → 检查代理设置")
        print("  3. 账户余额不足 → 登录 DeepSeek 平台查看")


if __name__ == "__main__":
    main()
