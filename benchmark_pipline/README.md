# Hugging Face Pipeline - RAG Benchmark 评测系统

独立的RAG系统评测pipeline，数据来自Hugging Face Hub，无需访问数据生成代码。

## 快速开始

### 1. 下载数据

```bash
cd huggingface_pipeline

# 下载所有数据（PDF + QA数据集）
python download_data.py

# 或只下载QA数据集
python download_data.py --download qa

# 或只下载PDF数据
python download_data.py --download pdf
```

### 2. 配置

```bash
cp config.yaml.example config.yaml
# 编辑 config.yaml，填写你的API密钥和配置
```

### 3. 测试

```bash
# 快速测试（每个数据集测试1个问题）
python test_pipeline.py --config config.yaml
```

### 4. 运行完整评测

```bash
python run_benchmark.py --config config.yaml
```

## Pipeline 流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. 数据准备                                                 │
│     download_data.py                                        │
│     ├── 下载PDF教材 → ../data/                              │
│     └── 下载QA数据集 → ./QA_data/                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 快速测试（可选）                                         │
│     test_pipeline.py                                        │
│     └── 每个数据集测试1个问题                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. 完整评测                                                 │
│     run_benchmark.py                                        │
│     ├── 检查QA数据集                                        │
│     ├── 初始化RAG Agent                                     │
│     ├── Step 4: 生成RAG回答（并行）                         │
│     ├── Step 5: LLM评分（并行）                             │
│     ├── 保存结果 → evaluation_results/{timestamp}/         │
│     └── 生成可视化 → evaluation_results/{timestamp}/       │
│                       visualizations/{split}/               │
└─────────────────────────────────────────────────────────────┘
```

## 配置说明

### config.yaml 主要配置项

```yaml
api:
  api_key: "your-api-key"
  base_url: "https://api.openai.com/v1"
  model_id: "gpt-4"

benchmark:
  splits: "all"                    # "all" 或单个split名称
  max_questions_per_split: null    # null/-1表示运行所有问题
  enable_visualization: true       # 是否生成可视化图表

judge_evaluation:
  workers: 4                       # 并行worker数量
```

### 可用的数据集 (splits)

- `Mao_Zedong_Thought`
- `Principles_of_Marxism`
- `Outline_of_Modern_and_Contemporary_Chinese_History`
- `Ideological_Morality_and_Legal_System`
- `An_Introduction_to_Xi_Jinping_Thought_on_Socialism_with_Chinese_Characteristics_for_a_New_Era`

## 评分系统

### 评分维度

- **Source Accuracy (来源准确性)** - 60%: 检查答案是否正确引用页码和来源
- **Content Accuracy (内容准确性)** - 20%: 答案内容的准确性
- **Completeness (完整性)** - 15%: 答案的完整程度
- **Relevance (相关性)** - 5%: 答案与问题的相关性

**最终得分** = Source × 0.6 + Content × 0.2 + Completeness × 0.15 + Relevance × 0.05

**通过标准**: Final Score ≥ 6.0

### 可视化图表

每个数据集生成6种图表：
1. 平均分柱状图
2. 分数分布箱线图
3. 雷达图
4. 最终得分分布直方图
5. 所有问题对比图（≤100题时）
6. 权重贡献分析图

## 结果目录结构

```
evaluation_results/
└── 20251201_143025/              # 时间戳文件夹
    ├── Mao_Zedong_Thought.csv   # 评分结果
    ├── Principles_of_Marxism.csv
    └── visualizations/           # 可视化图表
        ├── Mao_Zedong_Thought/
        │   ├── Mao_Zedong_Thought_average_scores.png
        │   ├── Mao_Zedong_Thought_score_distribution.png
        │   ├── Mao_Zedong_Thought_radar_chart.png
        │   └── ...
        └── Principles_of_Marxism/
            └── ...
```

## 常见问题

### 运行benchmark时提示数据集不存在

```bash
python download_data.py --download qa
```

### 只测试特定课程

修改 `config.yaml`:
```yaml
benchmark:
  splits: "Mao_Zedong_Thought"
```

### 限制测试问题数量

修改 `config.yaml`:
```yaml
benchmark:
  max_questions_per_split: 100
```

### 禁用可视化

修改 `config.yaml`:
```yaml
benchmark:
  enable_visualization: false
```

### 调整并行处理数量

修改 `config.yaml`:
```yaml
judge_evaluation:
  workers: 8
```

## 数据上传（数据提供者）

```bash
# 上传PDF数据
python upload_pdf_data.py
```

## Hugging Face 仓库

- **QA数据集**: `HEHUA2005/rag-benchmark-qa-dataset`
- **PDF数据**: `HEHUA2005/rag-benchmark-pdf-data`
