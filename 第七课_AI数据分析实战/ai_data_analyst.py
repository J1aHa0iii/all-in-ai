"""
第七课：AI 数据分析实战 —— 让 AI 自动分析 CSV/Excel 数据
========================================================
用法：
    python ai_data_analyst.py                    # 分析默认生成的示例数据
    python ai_data_analyst.py your_data.csv      # 分析你指定的 CSV 文件
    python ai_data_analyst.py your_data.xlsx      # 分析 Excel 文件

前提：
    1. pip install openai pandas
    2. 设置环境变量 DEEPSEEK_API_KEY=sk-xxx
       （或在代码里直接替换）

流程：
    读取数据 → 生成统计摘要 → DeepSeek API 分析 → 输出 Markdown 报告
"""

import os
import sys
import json
import pandas as pd
from openai import OpenAI


# ========== 配置 ==========

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-你的key填这里")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


# ========== 1. 创建示例数据（如果没有数据文件） ==========

def create_sample_data(output_path="sample_sales.csv"):
    """生成一份模拟电商销售数据，方便没有数据文件的读者直接测试"""
    import random
    from datetime import datetime, timedelta

    categories = {
        "数码": ["蓝牙耳机", "充电宝", "数据线", "手机壳", "笔记本支架"],
        "食品": ["坚果礼盒", "茶饮料", "方便面", "饼干", "巧克力"],
        "家电": ["电饭煲", "空气炸锅", "扫地机器人", "加湿器", "台灯"],
        "服装": ["T恤", "牛仔裤", "运动鞋", "帽子", "防晒衣"],
    }
    
    prices = {
        "蓝牙耳机": 199, "充电宝": 89, "数据线": 29, "手机壳": 19, "笔记本支架": 59,
        "坚果礼盒": 129, "茶饮料": 49, "方便面": 39, "饼干": 19, "巧克力": 69,
        "电饭煲": 399, "空气炸锅": 299, "扫地机器人": 1999, "加湿器": 149, "台灯": 89,
        "T恤": 79, "牛仔裤": 199, "运动鞋": 299, "帽子": 49, "防晒衣": 159,
    }

    rows = []
    start_date = datetime(2026, 1, 1)
    
    for i in range(300):  # 300 天数据，约 1500 行
        date = start_date + timedelta(days=i)
        # 每天随机 3-8 个订单
        for _ in range(random.randint(3, 8)):
            category = random.choice(list(categories.keys()))
            product = random.choice(categories[category])
            price = prices[product]
            
            # 周末销量高 30%
            is_weekend = date.weekday() >= 5
            quantity = random.randint(1, 15) if not is_weekend else random.randint(3, 25)
            
            # 模拟充电宝 3 月后销量下降
            if product == "充电宝" and date >= datetime(2026, 3, 1):
                quantity = max(1, quantity - random.randint(2, 8))
            
            # 10% 概率有打折
            discount = 1.0 if random.random() > 0.1 else round(random.uniform(0.7, 0.95), 2)
            
            sales = round(price * quantity * discount, 2)
            
            rows.append({
                "日期": date.strftime("%Y-%m-%d"),
                "星期": ["周一","周二","周三","周四","周五","周六","周日"][date.weekday()],
                "商品": product,
                "类别": category,
                "单价": price,
                "折扣": discount,
                "销量": quantity,
                "销售额": sales,
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 示例数据已生成：{output_path}（{len(df)} 行）")
    return output_path


# ========== 2. 数据 → 文字描述 ==========

def describe_data(file_path):
    """把 CSV/Excel 转成 AI 能理解的文字描述"""
    
    # 根据后缀选择读取方式
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    rows, cols = df.shape
    
    parts = []
    
    # 基本概览
    parts.append(f"## 数据概览\n")
    parts.append(f"- 文件：{file_path}")
    parts.append(f"- 行数：{rows}")
    parts.append(f"- 列数：{cols}")
    parts.append(f"- 字段：{', '.join(df.columns.tolist())}")
    parts.append("")
    
    # 数据类型
    parts.append("## 各列数据类型\n")
    for col, dtype in df.dtypes.items():
        parts.append(f"- {col}：{dtype}")
    parts.append("")
    
    # 数值列统计
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        parts.append("## 数值列统计\n")
        parts.append(df[numeric_cols].describe().round(2).to_string())
        parts.append("")
    
    # 缺失值
    missing = df.isnull().sum()
    if missing.sum() > 0:
        parts.append("## 缺失值统计\n")
        parts.append(missing[missing > 0].to_string())
        parts.append("")
    else:
        parts.append("## 缺失值统计\n无缺失值\n")
    
    # 分类列概况
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    if text_cols:
        parts.append("## 分类列概况\n")
        for col in text_cols:
            value_counts = df[col].value_counts()
            unique_count = len(value_counts)
            if unique_count <= 30:
                top_items = value_counts.head(10).to_string()
                parts.append(f"### {col}（{unique_count} 种取值）\n{top_items}\n")
            else:
                parts.append(f"### {col}（{unique_count} 种取值，太多不逐个展示）\n")
                parts.append(f"TOP3：{dict(value_counts.head(3))}\n")
    
    # 前 20 行样本
    parts.append("## 数据样本（前20行）\n")
    parts.append(df.head(20).to_string())
    
    return "\n".join(parts)


# ========== 3. AI 分析 ==========

def ai_analyze(data_description, question=None):
    """把数据描述发给 DeepSeek，生成分析报告"""
    
    if question is None:
        question = """请全面分析这份数据：
1. 核心指标概览（总销售额、总体情况等）
2. 关键发现（至少3条，每条用具体数据支撑）
3. 异常检测（哪些数据点可能有问题？为什么？）
4. 业务建议（基于数据给出可操作的改进方向）
5. 如果数据包含时间字段，请分析时间趋势"""
    
    prompt = f"""你是一个资深数据分析师。下面是一份数据的统计摘要和样本数据。

{data_description}

---
用户提问：{question}

请用 Markdown 格式输出一份专业的分析报告，要求：
- 用表格展示关键数字
- 每个发现必须有数据支撑
- 建议要具体、可执行
- 语言简洁专业，不要空话套话"""

    print("  ⏳ 正在调用 DeepSeek API 分析数据...")
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # 低温度减少幻觉
        max_tokens=4096,
    )
    
    return response.choices[0].message.content


# ========== 4. 生成图表（可选） ==========

def ai_generate_chart(data_description, chart_request="请生成一张最能体现数据核心趋势的图表"):
    """让 AI 生成 matplotlib 图表代码并执行"""
    
    prompt = f"""下面是一份数据的结构和样本：{data_description}

请为它生成一段 Python matplotlib 绘图代码，要求：{chart_request}

【重要规则】
1. 变量名统一用 df，df 已经由 pd.read_csv() 加载好了
2. 图片保存为 output_chart.png，dpi=150
3. 中文字体用 SimHei，如果报错就设置 matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
4. matplotlib.rcParams['axes.unicode_minus'] = False
5. 只需要输出代码，不要解释，不要 ```python``` 标记
6. 代码要能直接 exec() 执行"""

    print("  ⏳ 正在生成图表代码...")
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    
    code = response.choices[0].message.content.strip()
    
    # 去掉可能的 markdown 代码块标记
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    
    code = code.strip()
    
    print(f"  📝 AI 生成的图表代码：\n{code[:300]}...\n")
    
    return code


# ========== 5. 主流程：一键分析 ==========

def analyze_file(file_path, question=None):
    """完整流水线：读数据 → 描述 → AI分析 → 保存报告"""
    
    print(f"\n{'='*60}")
    print(f"📖 正在读取数据：{file_path}")
    
    # Step 1：描述数据
    description = describe_data(file_path)
    print(f"   ✅ 数据读取完成")
    
    # Step 2：AI 分析
    report = ai_analyze(description, question)
    
    # Step 3：保存报告
    base_name = os.path.splitext(file_path)[0]
    report_path = f"{base_name}_分析报告.md"
    
    full_report = f"""# 数据分析报告

**数据文件**：{file_path}
**分析时间**：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**分析工具**：AI 自动化分析（DeepSeek API）

---

{report}

---

*本报告由 AI 自动生成，关键结论请结合业务实际验证*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    print(f"\n✅ 分析报告已保存：{report_path}")
    print(f"{'='*60}")
    print(full_report)
    
    return full_report


# ========== 6. 交互式分析 ==========

def interactive_mode(file_path):
    """交互模式：分析完后可以继续追问"""
    
    print("\n" + "="*60)
    print("🤖 AI 数据分析师已启动")
    print("="*60)
    print(f"📂 当前数据：{file_path}")
    print()
    print("输入你的问题，AI 会基于数据分析回答。")
    print("例如：「充电宝销量为什么下降」「各品类利润率排名」「帮我找销量最高的10天」")
    print("输入 'exit' 或 'quit' 退出")
    print("输入 'chart' 让 AI 生成图表")
    print("="*60)
    
    description = describe_data(file_path)
    
    while True:
        question = input("\n💬 你的问题：").strip()
        
        if not question:
            continue
        if question.lower() in ('exit', 'quit', 'q'):
            print("👋 再见！")
            break
        
        if question.lower() == 'chart':
            chart_request = input("📊 想要什么图表？（直接回车=默认趋势图）：").strip()
            if not chart_request:
                chart_request = "请生成一张各品类月度销售额堆叠柱状图，体现各品类占比变化趋势"
            
            code = ai_generate_chart(description, chart_request)
            try:
                # 重新加载数据确保 df 可用
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                exec(code)
                print("✅ 图表已保存为 output_chart.png")
            except Exception as e:
                print(f"❌ 图表生成失败：{e}")
                print(f"AI 生成的代码：\n{code}")
            continue
        
        print("\n⏳ 正在分析...")
        answer = ai_analyze(description, question)
        print(f"\n🤖 AI 回答：\n{answer}")


# ========== 入口 ==========

if __name__ == "__main__":
    if "--interactive" in sys.argv or "-i" in sys.argv:
        # 交互模式
        # 去掉 -i 参数，取最后一个参数作为文件路径
        args = [a for a in sys.argv[1:] if a not in ('--interactive', '-i')]
        file_path = args[0] if args else None
        
        if not file_path:
            file_path = create_sample_data()
        
        interactive_mode(file_path)
    
    else:
        # 单次分析模式
        args = [a for a in sys.argv[1:] if not a.startswith('--')]
        file_path = args[0] if args else None
        
        if not file_path:
            print("📦 未指定数据文件，自动生成示例数据...\n")
            file_path = create_sample_data()
        
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在：{file_path}")
            sys.exit(1)
        
        analyze_file(file_path)
        
        print("\n💡 提示：用 --interactive 参数可以进入交互问答模式")
        print("   python ai_data_analyst.py --interactive")
