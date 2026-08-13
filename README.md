# KG-Agent

**A Knowledge Graph Enhanced Research Assistant for Scientific Discovery**

KG-Agent 是一个面向科研检索与论文问答的多路由 AI Agent。项目把 **LLM、OpenAlex 文献检索、证据质量控制、BGE 语义排序、PDF RAG、FAISS、Neo4j 科研知识图谱与 LangGraph 工作流**组合成一个统一的科研助手。

## 当前能力

系统支持 5 条完整路由：

| Route | 典型输入 | 核心能力 |
|---|---|---|
| `general` | “什么是过拟合？” | LLM 通用解释 |
| `research` | “分析近三年共情回复生成趋势” | OpenAlex → 质量过滤 → 语义排序 → 证据回答 → 自动入图 |
| `pdf` | 提供 PDF 并询问论文内容 | PDF → Chunk → BGE → FAISS → Top-K → Grounded Answer |
| `kg` | “知识图谱中哪些论文用了 EmpatheticDialogues？” | KG Planner → Neo4j Cypher → KG Grounded Answer |
| `hybrid` | 提供 PDF，并明确要求“结合知识图谱” | Neo4j KG Evidence + PDF RAG Evidence → Hybrid Answer |

## 总体架构

```text
User
  ↓
LangGraph Planner
  ├─ general ─────────────────────────→ General Answer
  │
  ├─ research → Search Planner → OpenAlex → Quality Filter
  │                              ↓
  │                        Semantic Ranking
  │                              ↓
  │                        KG Auto-Ingestion
  │                              ↓
  │                        Research Answer
  │
  ├─ pdf → PDF Loader → Chunker → BGE → FAISS → PDF Answer
  │
  ├─ kg → KG Query Planner → Neo4j Retrieval → KG Answer
  │
  └─ hybrid → KG Query Planner → Neo4j Retrieval
                                  ↓
                           PDF RAG Retrieval
                                  ↓
                           Hybrid Answer
```

## 知识图谱 Schema

```text
(:Author)-[:AUTHORED]->(:Paper)-[:PUBLISHED_IN]->(:Venue)
                           │
                           ├─[:USES_METHOD]->(:Method)
                           ├─[:EVALUATED_ON]->(:Dataset)
                           └─[:ADDRESSES_TASK]->(:Task)
```

OpenAlex 实体使用 `openalex_id` 作为稳定标识；LLM 抽取的 Method / Dataset / Task 使用 `normalized_name` 作为规范化身份字段。Neo4j 通过 UNIQUE Constraint + `MERGE` 保证重复执行不会不断创建重复实体。

## 环境要求

推荐环境：

- Python 3.11
- Windows / Linux 均可
- Neo4j Desktop 或可访问的 Neo4j Server
- OpenAI-compatible LLM API（当前配置可连接 DeepSeek）
- OpenAlex API Key

安装依赖：

```bash
pip install -r requirements.txt
```

如果 `models/embedding/bge-small-en-v1.5/` 不存在：

```bash
python -m scripts.download_embedding_model
```

## 配置

复制配置模板：

```bash
copy .env.example .env
```

Linux/macOS：

```bash
cp .env.example .env
```

填写：

```env
LLM_PROVIDER=deepseek
LLM_API_KEY=your_key
LLM_MODEL=your_model
LLM_BASE_URL=your_openai_compatible_base_url
OPENALEX_API_KEY=your_openalex_key

NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

`.env` 已加入 `.gitignore`，不要提交真实密钥。

## 初始化 Neo4j Schema

启动 Neo4j 后运行：

```bash
python -m scripts.init_kg_schema
```

系统会创建 Paper / Author / Venue / Method / Dataset / Task 的唯一性约束。

## 运行

交互模式：

```bash
python app.py
```

直接提问：

```bash
python app.py -q "请分析近三年共情回复生成的研究趋势"
```

PDF RAG：

```bash
python app.py -q "What reinforcement learning algorithm does EmpRL use?" --pdf "data/pdfs/EmpRL.pdf"
```

Hybrid KG + PDF：

```bash
python app.py -q "结合这篇 PDF 和知识图谱中的 CEM 论文信息，区分两类证据分别支持什么结论" --pdf "data/pdfs/EmpRL.pdf"
```

只有在提供 `--pdf` 且问题明确提到“知识图谱 / knowledge graph / Neo4j / KG”时，Planner 才确定性选择 `hybrid`；否则提供 PDF 时走 `pdf`。

## 关键测试

基础功能测试位于 `scripts/`。

完整 5 Route 回归测试：

```bash
python -m scripts.test_agent_regression "data/pdfs/EmpRL.pdf"
```

单独 Hybrid Route：

```bash
python -m scripts.test_langgraph_hybrid_route "data/pdfs/EmpRL.pdf"
```

其他重要测试：

```bash
python -m scripts.test_openalex
python -m scripts.test_entity_extraction
python -m scripts.test_kg_ingestion
python -m scripts.test_semantic_ingestion
python -m scripts.test_kg_queries
python -m scripts.test_langgraph_kg_route
python -m scripts.test_langgraph_research_kg_ingestion
python -m scripts.test_pdf_rag "data/pdfs/EmpRL.pdf"
```

## 目录结构

```text
KG-Agent/
├─ agent/                  # LangGraph、路由、KG/Hybrid Answer
├─ config/                 # 环境配置
├─ embedding/              # 本地 BGE 模型加载与缓存
├─ evidence/               # 论文质量过滤、词法/语义相关性
├─ knowledge_graph/        # Neo4j Schema、写入、查询、实体抽取
├─ llm/                    # OpenAI-compatible LLM Client
├─ rag/                    # PDF Loader、Chunk、Embedding、FAISS、QA
├─ scripts/                # 初始化、功能测试、回归测试
├─ tools/                  # OpenAlex paper search
├─ docs/                   # 项目技术、架构、面试资料
├─ app.py                  # CLI 入口
└─ requirements.txt
```

## 已完成阶段

```text
Phase 1   Project / Git Setup                  ✅
Phase 2   LLM Integration                      ✅
Phase 3   LangGraph                            ✅
Phase 4   Planner Routing                      ✅
Phase 5   OpenAlex Paper Search                ✅
Phase 6   Search Planner + Abstract Evidence   ✅
Phase 7   Evidence Quality + Semantic Ranking  ✅
Phase 8   PDF / Vector RAG                     ✅
Phase 9   Neo4j Knowledge Graph                ✅
Phase 10  KG + RAG Agent                       ✅
```

## 当前已知技术债

1. LLM `chat_json()` 在当前兼容接口上偶尔出现空 JSON / 截断 JSON；Research → KG Auto-Ingestion 已做 graceful degradation，语义抽取失败不会阻断 Research Answer。
2. PDF FAISS Index 当前每次查询重新构建，生产版本可做按文档缓存或持久化索引。
3. Chunking 当前是字符级固定窗口 `1200 / overlap 200`，后续可升级为 token-aware / section-aware chunking。
4. Method 关系目前统一建模为 `USES_METHOD`，V2 可进一步细分 `PROPOSES_METHOD`、`USES_METHOD`、`COMPARES_WITH`。
5. 当前 KG Query Planner 只支持 Paper / Method / Dataset / Task，后续可扩展 Author / Venue / multi-hop query。

## 安全说明

- 不要提交 `.env`。
- Neo4j 查询采用固定、参数化 Cypher；当前版本不让 LLM 直接生成并执行任意 Cypher。
- KG Answer、PDF Answer、Hybrid Answer 都采用 evidence-grounded prompt，明确限制模型不得使用未提供的外部事实。

## License

MIT License.
