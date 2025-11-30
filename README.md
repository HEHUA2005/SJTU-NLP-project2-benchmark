# RAG 智能课程助教系统

<div align="center">

**基于检索增强生成（RAG）技术的智能课程问答助手**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 项目简介

本项目是上海交通大学《自然语言处理》课程的大作业2，实现了一个基于 **RAG（Retrieval-Augmented Generation）** 技术的智能课程助教系统。

### 核心特性

- **多格式文档支持**：支持 PDF、PPTX、DOCX、TXT 等多种文档格式
- **智能文档解析**：自动提取文档内容，保留页码和文件结构信息
- **向量化检索**：基于 ChromaDB 的高效向量相似度搜索
- **上下文感知**：利用 LLM 生成准确、有依据的回答
- **来源追溯**：回答中标注具体的文件名和页码信息
- **交互式对话**：支持多轮对话，维护上下文历史

### 技术架构

```
┌─────────────┐
│  课程文档   │ (PDF/PPTX/DOCX/TXT)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 文档加载器  │ (DocumentLoader)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 文本切分器  │ (TextSplitter)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 向量数据库  │ (ChromaDB + Embeddings)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  RAG Agent  │ (检索 + 生成)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  用户交互   │
└─────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- OpenAI API Key（或兼容的 API，如阿里云百炼）
- 至少 2GB 可用磁盘空间

### 安装步骤

1. **克隆项目**

```bash
git clone <repository-url>
cd SJTU-NLP-project2
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

3. **配置 API**

编辑 [config.py](config.py) 文件，填写以下配置：

```python
# API配置
OPENAI_API_KEY = "your-api-key"
OPENAI_API_BASE = "https://api.openai.com/v1"  # 或其他兼容API
MODEL_NAME = "gpt-4"  # 或 qwen-max, deepseek-chat 等
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# 数据目录配置
DATA_DIR = "./data"

# 向量数据库配置
VECTOR_DB_PATH = "./vector_db"
COLLECTION_NAME = "course_materials"

# 文本处理配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_TOKENS = 8000

# RAG配置
TOP_K = 5
```

4. **准备课程文档**

将课程文档（PDF、PPTX、DOCX、TXT）放入 `data/` 目录：

```bash
mkdir -p data
cp /path/to/your/course/materials/* data/
```

5. **处理文档并建立向量库**

```bash
python process_data.py
```

6. **启动对话系统**

```bash
python main.py
```

---

## 📁 项目结构

```
SJTU-NLP-project2/
├── config.py              # 配置文件（API Key、模型参数等）
├── document_loader.py     # 文档加载模块：多格式文档解析
├── text_splitter.py       # 文本切分模块：长文档智能分块
├── vector_store.py        # 向量数据库模块：Embedding 与检索
├── rag_agent.py          # RAG 智能体：核心问答逻辑
├── process_data.py       # 数据处理流水线脚本
├── main.py               # 交互式主程序
├── requirements.txt      # 项目依赖
├── data/                 # 课程文档目录
│   ├── *.pdf
│   ├── *.pptx
│   ├── *.docx
│   └── *.txt
└── vector_db/            # 向量数据库存储目录（自动生成）
```

---

## 💡 使用示例

### 基本对话

```
欢迎使用智能课程助教系统！
============================================================

学生: 词的连续向量表示为什么又称作"分布式表达"？

助教: 根据课程文档《词向量.pdf》第 6 页的内容，词的连续向量表示被称为
"分布式表达"是因为：

在传统的 one-hot 表示中，一个词由且仅由一个维度表示，因此也被称为
"局部语义表达"或"非分布式表达"。而在连续向量表示中，一个词的语义
信息分布在向量的多个维度上，每个维度都贡献了部分语义信息，因此称为
"分布式表达"（Distributed Representation）。

这种表示方法的优势在于：
1. 能够捕捉词与词之间的语义相似性
2. 维度更低，表示更紧凑
3. 可以进行向量运算，如 King - Man + Woman ≈ Queen

来源：《词向量.pdf》第 6 页
```

### 多轮对话

系统会自动维护对话历史，支持上下文相关的追问。

---

## 🛠️ 核心模块说明

### 1. DocumentLoader（文档加载器）

负责加载和解析不同格式的文档：

- **PDF**：使用 PyPDF2 按页提取文本
- **PPTX**：使用 python-pptx 按幻灯片提取内容
- **DOCX**：使用 docx2txt 提取文档文本
- **TXT**：直接读取纯文本文件

详见：[document_loader.py](document_loader.py)

### 2. TextSplitter（文本切分器）

将长文档切分为适合向量化的小块：

- 支持自定义块大小（chunk_size）
- 支持块重叠（chunk_overlap）以保持上下文连续性
- 智能在句子边界处切分

详见：[text_splitter.py](text_splitter.py)

### 3. VectorStore（向量数据库）

基于 ChromaDB 实现向量存储与检索：

- 调用 OpenAI API 生成文本 Embeddings
- 存储文档块及其元数据
- 基于余弦相似度的向量检索

详见：[vector_store.py](vector_store.py)

### 4. RAGAgent（RAG 智能体）

系统核心，整合检索与生成：

- 根据用户问题检索相关文档
- 构建包含上下文的提示词
- 调用 LLM 生成准确回答
- 维护多轮对话历史

详见：[rag_agent.py](rag_agent.py)

---

## 🎯 核心任务实现

本项目包含四个核心任务，所有 TODO 标记的方法都需要实现：

### 任务一：环境与数据准备
- ✅ 安装依赖
- ✅ 配置 API
- ✅ 准备课程文档

### 任务二：文档处理模块
- 📝 实现 `load_pdf()` - PDF 文本提取
- 📝 实现 `load_pptx()` - PPT 文本提取
- 📝 实现 `load_docx()` - Word 文档提取
- 📝 实现 `load_txt()` - 纯文本读取
- 📝 实现 `split_text()` - 文本智能切分

### 任务三：向量数据库
- 📝 实现 `get_embedding()` - 获取文本向量
- 📝 实现 `add_documents()` - 文档入库
- 📝 实现 `search()` - 向量相似度检索

### 任务四：RAG Agent
- 📝 设计 System Prompt - 定义助教角色
- 📝 实现 `retrieve_context()` - 检索相关上下文
- 📝 实现 `generate_response()` - 生成回答

---

## 🌟 扩展方向

鼓励同学们在完成基础任务后，探索以下扩展方向：

### 技术优化
- **混合检索**：结合 BM25（稀疏检索）和向量检索（密集检索）
- **重排序**：使用 Reranker 模型提升检索精度
- **多模态支持**：处理课件中的图片、图表等非文本内容
- **查询改写**：优化用户问题以提高检索效果

### 功能扩展
- **Web UI**：基于 Gradio/Streamlit 构建可视化界面
- **习题生成**：根据课程内容自动生成练习题
- **知识图谱**：构建课程知识点关系图
- **多语言支持**：支持中英文混合问答

---

## 📊 API 资源说明

### 阿里云百炼（推荐）

新用户可获得 **90 天 100 万 token** 免费额度，支持：
- Qwen-Max
- Qwen-Plus
- DeepSeek-V3
- 多种 Embedding 模型

申请链接：https://help.aliyun.com/zh/model-studio/get-api-key

### 其他兼容 API

本项目支持任何 OpenAI 兼容的 API 接口，包括：
- OpenAI 官方 API
- Azure OpenAI
- 本地部署的 LLM（如 Ollama、vLLM）

---

## 🐛 常见问题

### Q1: 如何处理中文文档？
A: 确保文档使用 UTF-8 编码，代码中已设置 `encoding="utf-8"`。

### Q2: API 调用超时怎么办？
A: 检查网络连接，或在配置中增加 timeout 参数。

### Q3: 向量数据库占用空间过大？
A: 可以调整 `CHUNK_SIZE` 参数减少文档块数量，或定期清理旧数据。

### Q4: 如何提高回答质量？
A:
- 优化 System Prompt
- 调整 `TOP_K` 参数增加检索文档数
- 使用更强的 LLM 模型
- 改进文档切分策略

---

## 📝 作业提交

提交内容应包括：
1. 完整的代码实现
2. 运行截图和演示视频
3. 实验报告（包括设计思路、实现细节、测试结果）
4. 扩展功能说明（如有）

---

## 📚 参考资料

- [RAG 技术综述](https://arxiv.org/abs/2312.10997)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [OpenAI API 文档](https://platform.openai.com/docs)
- [LangChain RAG 教程](https://python.langchain.com/docs/use_cases/question_answering/)

---

## 👥 贡献者

- 项目作者：[Your Name]
- 课程：上海交通大学 CS3602 自然语言处理
- 学期：2025-2026-1

---

## 📄 License

本项目采用 MIT 协议开源，详见 LICENSE 文件。

---

<div align="center">

**Happy Coding!** 🎉

如有问题，欢迎提交 Issue 

</div>
