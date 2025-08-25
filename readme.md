#激活虚拟环境
venv_alloykg\Scripts\activate

$env:ZHIPU_API_KEY="a3d236a6017d4cfc9f15c509a3e7c786.eZIIBa9QmssnGEsv"


python parse_papers.py

python scripts\extract_triples.py

python scripts\train_openke.py

python scripts\build_kg.py

python scripts\query_demo.py


AlloyKG_Project/
│
├── data/                     # 数据存放
│   ├── raw/                  # 原始论文 PDF、图片、表格
│   ├── processed/            # 已解析成统一格式的多模态数据（text chunks, tables, figure captions）
│   └── annotations/          # 标注数据，JSON/CSV 格式（用于 IE/LLM 校验）
│
├── scripts/                  # 一些可单独运行的脚本
│   ├── parse_papers.py       # 使用 RAG-Anything 分析 PDF/图像/表格
│   ├── extract_triples.py    # 实体/关系抽取（IE/LLM 模块）
│   ├── train_openke.py       # OpenKE embedding 训练与评估
│   ├── build_kg.py           # Neo4j 图谱生成与导入
│   └── query_demo.py         # KG 查询/QA 演示脚本
│
├── modules/                  # 项目自研模块或库
│   ├── rag_parser/           # RAG-Anything 封装接口
│   │   ├── __init__.py
│   │   └── parser.py
│   ├── ie_module/            # 信息抽取模块（LLM + JSON Schema）
│   │   ├── __init__.py
│   │   └── extractor.py
│   ├── kg_embedding/         # OpenKE 封装
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── kg_db/                # Neo4j/ArangoDB 操作接口
│       ├── __init__.py
│       └── neo4j_client.py
│
├── notebooks/                # Jupyter/Colab 实验和分析
│   ├── pipeline_demo.ipynb   # 全流程示例
│   └── embedding_analysis.ipynb
│
├── configs/                  # 配置文件
│   ├── rag_config.yaml       # RAG-Anything 配置
│   ├── ie_config.yaml        # 信息抽取模型配置
│   ├── openke_config.json    # OpenKE 超参数配置
│   └── neo4j_config.json     # 图数据库连接配置
│
├── requirements.txt          # Python 环境依赖
├── README.md                 # 项目说明
└── main.py                   # 可作为 CLI/服务入口的主程序
