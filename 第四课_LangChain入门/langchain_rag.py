#!/usr/bin/env python3
"""用 LangChain 重构第三课的 RAG —— 实战知识库问答机器人

使用前请准备：
1. 在脚本同目录下创建「知识库」文件夹，丢进 .md 或 .txt 文件
2. 如果没有文档，脚本会自动创建示例文档
3. pip install langchain langchain-community langchain-openai faiss-cpu openai sentence-transformers

DeepSeek API Key 获取：https://platform.deepseek.com/
"""

import os
import sys

# ==================== 配置 ====================

DEEPSEEK_KEY = "sk-your-deepseek-api-key"  # 替换成你的 Key
KNOWLEDGE_DIR = "知识库"                    # 文档目录

# ==================== 自动创建示例文档（如果没有的话） ====================

if not os.path.exists(KNOWLEDGE_DIR):
    os.makedirs(KNOWLEDGE_DIR)

# 检查是否有文档文件
has_docs = any(f.endswith((".md", ".txt")) for f in os.listdir(KNOWLEDGE_DIR) if os.path.isfile(os.path.join(KNOWLEDGE_DIR, f)))

if not has_docs:
    print("📝 没有找到文档，自动创建示例知识库……\n")
    sample_docs = {
        "员工手册-请假流程.md": """# 员工手册 - 请假流程

## 请假类型
- 年假：工作满1年享有5天，每增加1年增加1天，上限15天
- 病假：需提供医院证明，每月累计不超过3天的不扣工资
- 事假：无薪假，每年累计不超过10天

## 请假流程
1. 提前1天在OA系统提交请假申请
2. 直属领导审批（1个工作日内）
3. 请假超过3天（含3天）：需HR部门复核
4. 请假超过5天：需部门总监审批
5. 病假需事后48小时内补交医院证明

## 注意事项
- 年假不可跨年累计，每年12月31日清零
- 事假和年假不可连休
- 婚假3天，需提前1个月申请，提供结婚证复印件
""",

        "IT管理制度-密码规范.md": """# IT管理制度 - 密码规范

## 密码要求
- 长度不少于8位，必须包含数字和字母组合
- 每90天强制更换
- 不能与最近5次密码重复
- 首次登录必须修改初始密码

## 数据库密码特殊要求
- 需包含特殊字符（!@#$%^&*）
- 长度不少于12位
- 禁止明文存储，必须使用密钥管理系统

## VPN使用
- 在OA系统提交VPN申请 → 部门主管审批 → IT部门开通
- VPN账号由IT部门统一分配
- 密码与域账号密码一致
- 离职时VPN权限自动撤销
""",

        "报销制度-差旅报销.md": """# 报销制度 - 差旅报销

## 住宿费标准
- 一线城市（北上广深）：≤500元/晚
- 其他城市：≤400元/晚
- 超出部分自理

## 交通费
- 高铁二等座可实报实销
- 飞机经济舱可实报实销（需提前3天申请）
- 市内交通：≤100元/天

## 餐补
- 出差期间：80元/天
- 半天出差：40元/半天
- 无需发票，随工资发放

## 报销流程
- 出差结束后7个工作日内提交报销申请
- 需附住宿发票、交通票据
- 部门主管审批 → 财务审核 → 打款到工资卡
""",

        "新员工入职指南.md": """# 新员工入职指南

## 入职第一天
1. 到HR部门报到，提交身份证、学历证明、银行卡复印件
2. 领取工牌和门禁卡
3. 到IT部门领取办公设备（笔记本、显示器、键鼠）
4. IT分配域账号、邮箱、OA系统账号
5. 首次登录需修改初始密码

## 入职第一周
- 参加公司文化和制度培训（HR统一安排）
- 认识团队成员，确定导师
- 安装开发环境（参照IT提供的环境配置文档）
- 阅读团队文档和代码仓库

## 试用期
- 试用期3个月
- 第2个月末进行中期评估
- 试用期结束前2周进行转正答辩
- 转正后享有全部福利待遇
""",

        "会议室使用规定.md": """# 会议室使用规定

## 会议室列表
- 201室：小会议室，容纳6人，有白板
- 301室：中会议室，容纳12人，有投影仪
- 501室：大会议室，容纳30人，有视频会议系统
- 502室：洽谈室，容纳4人，适合面试

## 预约规则
- 通过OA系统「会议室预约」模块预约
- 最少提前2小时，最多提前30天
- 单次预约最长4小时
- 同一时段不可重复预约

## 使用须知
- 会议结束后请关闭投影仪和空调
- 白板使用后请擦干净
- 超时15分钟会被系统自动释放
""",
    }

    for filename, content in sample_docs.items():
        filepath = os.path.join(KNOWLEDGE_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✅ 创建了 {filename}")

    print(f"\n📂 已创建 6 份示例文档到「{KNOWLEDGE_DIR}」目录\n")

# ==================== 1. 加载文档 ====================

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

print("📂 加载文档中……")
documents = []
for f in sorted(os.listdir(KNOWLEDGE_DIR)):
    if f.endswith((".md", ".txt")):
        filepath = os.path.join(KNOWLEDGE_DIR, f)
        loader = TextLoader(filepath, encoding="utf-8")
        loaded = loader.load()
        print(f"  ✅ {f} （{len(loaded)} 段）")
        documents.extend(loaded)

if not documents:
    print("❌ 没有找到任何文档！请在「知识库」目录下放入 .md 或 .txt 文件。")
    sys.exit(1)

print(f"\n📊 共加载 {len(documents)} 篇文档\n")

# ==================== 2. 切片 ====================

print("✂️ 智能切片中……")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)
chunks = text_splitter.split_documents(documents)
print(f"✅ 切成了 {len(chunks)} 个知识块\n")

# ==================== 3. 向量化存储 ====================

print("🧠 向量化中（首次运行会下载 Embedding 模型，约 100MB）……")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.from_documents(chunks, embeddings)
print("✅ 向量库构建完成\n")

# ==================== 4. 组装 RAG 链 ====================

print("🔗 组装 RAG 链……")
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=DEEPSEEK_KEY,
    temperature=0.3,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
)

system_prompt = """你是一个企业知识库助手。根据参考资料回答用户问题。

规则：
1. 只根据参考资料中的内容回答，不要编造
2. 如果参考资料中没有相关信息，明确说「知识库中未找到相关内容」
3. 回答时注明信息来源文档
4. 回答简洁清晰，用中文
"""

# 更新链的 prompt（用 stuff chain 的内部字段）
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
qa_chain.combine_documents_chain.llm_chain.prompt.messages[0].prompt.template = (
    system_prompt
    + "\n\n参考资料：\n{context}"
)
print("✅ 准备就绪\n")

# ==================== 5. 交互式问答 ====================

print("=" * 60)
print("   🤖 RAG 知识库机器人 —— LangChain 版")
print("=" * 60)
print(f"   已加载 {len(documents)} 篇文档，共 {len(chunks)} 个知识块")
print("   输入问题开始提问，输入 quit 退出")
print("=" * 60)

while True:
    try:
        user_input = input("\n👤 你：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n👋 再见！")
        break

    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit", "退出", "q"):
        print("👋 再见！")
        break

    # 调用 RAG
    result = qa_chain.invoke({"query": user_input})

    # 提取引用来源
    sources = {}
    for doc in result.get("source_documents", []):
        src = doc.metadata.get("source", "未知来源")
        sources[src] = sources.get(src, 0) + 1

    # 输出结果
    print(f"\n🤖 答：{result['result']}")

    if sources:
        source_list = [f"{name}（{cnt}处）" for name, cnt in sources.items()]
        print(f"📎 参考来源：{', '.join(source_list)}")
    print("-" * 60)
