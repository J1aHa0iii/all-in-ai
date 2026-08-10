# 🤖 AI 从入门到入土

> 公众号「菜鸟进阶站」AI 实战教程系列，保姆级手把手带你从 0 开始学大模型开发。

[![公众号](https://img.shields.io/badge/公众号-菜鸟进阶站-brightgreen)](https://mp.weixin.qq.com)
[![系列合集](https://img.shields.io/badge/系列-AI从入门到入土-blue)](#)
[![Stars](https://img.shields.io/github/stars/J1aHa0iii/all-in-ai?style=social)](https://github.com/J1aHa0iii/all-in-ai)

---

## 📚 课程目录

| 课次 | 标题 | 核心内容 | 代码 |
|------|------|---------|------|
| 第一课 | [10块钱入门大模型：用DeepSeek API搭建SQL助手](文章/第一课_10块钱入门大模型.md) | API调用、AI写SQL、多轮对话 | [代码](第一课_10块钱入门大模型/) |
| 第二课 | [Function Calling实战：让AI帮你查数据库](文章/第二课_FunctionCalling实战.md) | 工具注册、对话循环、SQLite实战 | [代码](第二课_FunctionCalling实战/) |
| 第三课 | [把200份文档喂给AI：RAG实战](文章/第三课_RAG实战.md) | TF-IDF检索、Embedding、企业知识库 | [代码](第三课_RAG实战/) |
| 第四课 | [LangChain入门：20行代码重构RAG](文章/第四课_LangChain入门.md) | LangChain框架、FAISS、智能切片 | [代码](第四课_LangChain入门/) |
| 第五课 | [Agent智能体：让AI自己决定先干嘛](文章/第五课_Agent智能体实战.md) | ReAct模式、多工具协作、手写+LangChain双版本 | [代码](第五课_Agent智能体实战/) |
| 第六课 | [上线！Gradio Web应用：把AI助手变成网页](文章/第六课_Gradio Web应用.md) | Gradio框架、ChatInterface、三合一Web应用 | [代码](第六课_Gradio Web应用/) |
| ... | 更多课程筹备中 | 多模态实战 / 部署上线 | ⏳ |

---

## 🚀 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/J1aHa0iii/all-in-ai.git
cd all-in-ai

# 2. 安装依赖
pip install openai

# 3. 注册 DeepSeek 获取 API Key（10块钱用一个月）
#    https://platform.deepseek.com/

# 4. 把代码里的 sk-your-deepseek-api-key 换成你自己的 Key

# 5. 跑起来！
# 第一课
python 第一课_10块钱入门大模型/demo_gen_sql.py
python 第一课_10块钱入门大模型/sql_assistant.py

# 第二课
python 第二课_FunctionCalling实战/init_db.py
python 第二课_FunctionCalling实战/ai_db_assistant.py

# 第三课
pip install openai numpy scikit-learn
python 第三课_RAG实战/rag_knowledge_base.py

# 第四课
pip install langchain langchain-community langchain-openai faiss-cpu sentence-transformers openai
python 第四课_LangChain入门/langchain_rag.py

# 第五课（手写版，只需 openai）
pip install openai
python 第五课_Agent智能体实战/agent_demo.py

# 第五课（LangChain版）
pip install openai langchain langchain-community langchain-openai faiss-cpu sentence-transformers
python 第五课_Agent智能体实战/agent_demo.py --langchain

# 第六课（Web应用，三种模式一键切换）
pip install gradio openai langchain-community faiss-cpu sentence-transformers
python 第六课_Gradio Web应用/ai_web_app.py

# 第六课（生成公网分享链接）
python 第六课_Gradio Web应用/ai_web_app.py --share
```

---

## 🎯 这个系列适合谁？

- 想学 AI 但不知从哪下手的技术人
- 每天写 SQL、做 ETL 的数据工程师
- 有 Python 基础，想转型 AI 开发的程序员
- 喜欢「手把手保姆级教程」的学习者

---

## 📖 系列特色

- **从舒适区切入**：第一课从你最熟悉的 SQL 场景切入 AI，不跳级、不劝退
- **10块钱玩到爽**：全程使用 DeepSeek API，性价比拉满
- **代码能跑才是王道**：每篇都附带完整可运行代码，改个 Key 就能跑
- **保姆级讲解**：Function Calling、RAG 这些概念都用人话拆开讲

---

## ⭐ Star History

如果这个系列对你有帮助，点个 Star 支持一下！

---

## 📬 联系

- 公众号：**菜鸟进阶站**（后台回复「AI从入门到入土」获取最新文章）
- GitHub：[J1aHa0iii](https://github.com/J1aHa0iii)
