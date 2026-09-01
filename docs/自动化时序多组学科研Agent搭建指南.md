# 自动化 bulk 时序多组学科研 Agent 搭建指南

> 面向 AI 初学者；数据对象限定为 bulk 类型多组学；目标是把“数据获取 → 预处理 → 多模态时序预测 → 自动调优 → 内部验证 → 可解释性 → 文献核验 → 候选湿实验假设”做成一条可复现、可审计的科研流水线。
>
> 文档日期：2026-09-01。本文最后附有可直接粘贴给 Cursor、Claude Code 等编码助手的主提示词和分阶段提示词。

---

## 一、先说结论：你真正要搭建的不是一个“万能 Agent”

最稳妥的系统是三层结构：

1. **确定性科研流水线**：真正负责下载、校验、预处理、数据划分、训练、计算指标和生成图表。相同输入、相同版本、相同随机种子应得到可复现结果。
2. **受约束的 Agent 编排层**：大模型负责读论文和元数据、填写数据清单、选择预先注册的模型、解释报错、根据验证集结果提出下一轮配置。它不能修改测试集、评价函数或数据划分，也不能未经批准执行任意网络代码。
3. **人工审批层**：对样本—时间—模态映射、外部仓库引入、超算费用、最终测试集解锁和“新发现”表述进行审核。

不建议一开始让 LLM 自由下载任意仓库、运行任意脚本并循环改代码。这样很容易发生数据泄漏、样本错配、供应链安全问题、测试集过拟合和无法复现。

### 第一版（MVP）应只完成这一件事

选择 **一个公开、已提供处理后矩阵的 bulk 时序双组学数据集**，例如 bulk RNA + bulk 蛋白，满足：

- 至少 3 个真实时间点，最好 5 个以上；
- 有清楚的 subject/donor/replicate、condition、batch 和 time 元数据；
- 两个模态可以在同一 biospecimen、同一生物个体和同一时间点层面对齐；
- 每个时间点有独立生物重复，技术重复不能冒充生物重复；
- 第一版只读取作者提供的 count/abundance matrix，不从 FASTQ 重新比对。

第一版成功标准不是“发表新算法”，而是：从一个 `dataset.yaml` 出发，一条命令可复现数据审计、预处理、基线、一个 ODE/CDE 模型、锁定测试集评估和解释性候选表。

---

## 二、先分清你的数据是哪一种

这是整个项目最容易出错、也最影响模型设计的部分。

### 2.1 纵向重复测量 bulk 数据

同一 donor/subject 在多个时间点被采样，并且每次有多个组学。可以研究“给定该个体过去，预测该个体未来”。划分时必须按 donor 分组；同一 donor 不能同时进入训练集和测试集。若是动物处死取材，则通常不属于这一类。

### 2.2 重复横断面 bulk 数据

有些 bulk 时间序列在每个时间点使用不同动物、培养皿或组织样本，例如动物处死取材。它们有真实时间顺序，但没有同一个体的纵向轨迹。此时任务应定义为“根据早期时间点的训练群体预测后期相同 condition 下的新生物重复”，而不是预测某一个体的未来。

此类数据应保留 `experimental_unit_id`，划分时按动物、培养批次或独立实验批次分组。若各时间点来自完全不同批次且没有交叉重复，time 与 batch 可能无法区分，系统必须停止并提示实验设计混杂。

### 2.3 bulk 模态的配对层级

RNA、蛋白、代谢物可能来自同一组织样本分装、同一个体的不同组织，或完全不同的生物重复。必须用 `pairing_level` 标记：

- `same_aliquot`：同一份样本；
- `same_biospecimen`：同一组织样本的不同 aliquot；
- `same_subject_time`：同一个体、同一时间点，但可能是不同组织/取样；
- `group_level_only`：仅 condition × time 层面对齐。

只有前三类可在相应层级学习样本级跨组学关系。`group_level_only` 只能研究群体平均关系，不能把行号相同当作真实配对。

### 2.4 样本量现实检查

深度 ODE 模型参数很多。若只有 3 个时间点、每点 2 个生物重复，即使每个样本有上万特征，也仍然是小样本问题。此时优先做：

- 通路、模块或低维潜变量建模；
- 高变特征或通路维度降维；
- Ridge/Elastic Net/VAR/MEFISTO；
- 强正则化的小型 latent ODE；
- bootstrap 不确定性；
- 不做“自动发现复杂网络”的夸张结论。

---

## 三、系统总体架构

```text
用户输入（GEO/PRIDE/论文 DOI/URL/本地文件）
                    │
                    ▼
         Dataset Curator（LLM，只产出清单）
                    │ dataset.yaml + 待确认事项
                    ▼
       人工确认样本、时间、模态和许可证
                    │
                    ▼
  Downloader → checksum → raw/processed immutable store
                    │
                    ▼
   Deterministic preprocessing → MuData/Parquet + QC report
                    │
                    ▼
       Split guard：先锁定 outer test，生成 split hash
                    │
                    ▼
  Baselines → registered candidate models → validation-only HPO
                    │
                    ▼
   Model Critic：泄漏检查、稳定性、消融、复杂度与预算检查
                    │
                    ▼
          人工批准一次性解锁最终测试集
                    │
                    ▼
  Explainability → 稳定候选 → PubMed/Europe PMC 文献核验
                    │
                    ▼
  “已有支持 / 矛盾证据 / 未检索到直接证据”的假设报告
```

### 3.1 推荐先做一个 Agent 进程，而不是很多会聊天的 Agent

代码可以有六个模块，但第一版不需要部署六个独立大模型：

- `curator`：将文章、accession 和补充材料转换为结构化清单；
- `data_engineer`：调用确定性下载与预处理任务；
- `model_selector`：只从模型注册表选配置；
- `optimizer`：调用 Optuna，在验证集上调参；
- `critic`：执行泄漏、稳定性和晋级规则；
- `literature_analyst`：为已通过稳定性筛选的少量候选查文献。

第二阶段可将 `literature_analyst` 独立为另一个 Agent，因为它需要网络检索、证据抽取和独立核验；训练流水线本身不应依赖它。

### 3.2 LLM 可以做什么、不能做什么

允许：

- 从论文和数据库元数据抽取 accession、物种、时间点、模态和文件列表；
- 生成 `dataset.yaml` 草稿和待人工确认问题；
- 从白名单模型中选择候选；
- 阅读训练日志并提出下一轮配置；
- 把解释性候选转换成 PubMed 检索式；
- 生成带证据级别的报告草稿。

禁止：

- 猜测缺失的 donor/time/sample 对应关系；
- 在划分后用全数据拟合标准化、特征选择或批次校正；
- 查看最终测试标签后继续调参；
- 修改评估代码、split 文件或晋级阈值来使模型“变好”；
- 自动执行下载仓库中的 shell/install 脚本；
- 将“没有搜到论文”写成“首次发现”；
- 把注意力权重或 SHAP/IG 直接写成因果关系。

---

## 四、推荐技术栈

第一版尽量少而稳：

| 目的 | 推荐 | 说明 |
|---|---|---|
| 语言 | Python 3.11 | 生态完整，暂不追最新小版本 |
| 环境/锁版本 | `uv` + `pyproject.toml` + lock file | 所有依赖固定版本 |
| 数据对象 | AnnData/MuData + Parquet | 单模态 `.h5ad`，多模态 `.h5mu` |
| 参数校验 | Pydantic | 拒绝缺字段、错误类型和非法路径 |
| 命令行 | Typer | 初学者可用一条命令运行 |
| 流水线 | Snakemake | 下载、预处理、训练、报告均显式声明输入输出 |
| 模型 | PyTorch Lightning 或纯 PyTorch | 第一版二选一，不要混用 |
| 连续时间 | `torchdiffeq`；需要 CDE 时用 `torchcde` | 不规则时间间隔可显式建模 |
| 图先验 | PyTorch Geometric 或稀疏矩阵实现 | 第一版图拉普拉斯正则可不依赖复杂 GNN |
| 调参 | Optuna | 固定 sampler seed、study name、最大 trial 和 pruning |
| 实验记录 | MLflow | 记录参数、代码版本、数据 hash、指标和模型文件 |
| 数据版本 | DVC（第二阶段） | 第一版也可先用 checksum manifest |
| 解释性 | Captum + 自定义 feature ablation | IG 必须配合扰动和稳定性分析 |
| 文献 | NCBI E-utilities / Europe PMC REST | 每项结论保存 PMID/DOI、检索式和检索日期 |
| 测试 | pytest + Ruff + mypy | 用合成数据做端到端测试 |

为什么用 Snakemake：它将每一步的输入、输出、环境和重跑条件写清楚，并支持 Conda/容器与可复现归档。不要同时再引入一个庞大的工作流平台。若以后要定时、重试和 Web 监控，再在 Snakemake 外层加 Prefect。

这里使用 AnnData/MuData 只是为了统一保存“bulk 样本 × feature 矩阵、样本元数据和多个模态”，不代表系统在做单细胞分析。每一行 `obs` 是一个 bulk 生物样本或明确的实验单位，而不是一个细胞。

---

## 五、统一输入和数据契约

### 5.1 用户入口

支持四类输入：

```bash
omics-agent ingest --accession GSEXXXX
omics-agent ingest --paper-doi 10.xxxx/xxxxx
omics-agent ingest --url https://...
omics-agent ingest --local /path/to/data
```

无论从哪里开始，都必须先生成 `dataset.yaml`，不能直接训练。

### 5.2 `dataset.yaml` 示例

```yaml
schema_version: "1.0"
dataset_id: "example_rna_protein_timecourse"
title: "..."
source:
  type: "geo"                 # geo|sra|biostudies|pride|mw|url|local
  accession: "GSEXXXX"
  paper_doi: "10.xxxx/xxxxx"
  landing_page: "https://..."
license:
  name: "..."
  redistributable: false
organism:
  taxon_id: 9606
  name: "Homo sapiens"
design:
  unit_of_independence: "donor"
  sampling_design: "longitudinal"  # longitudinal|repeated_cross_sectional
  longitudinal: true
  pairing_level: "same_biospecimen" # same_aliquot|same_biospecimen|same_subject_time|group_level_only
  paired_modalities: true
  time_unit: "hour"
modalities:
  rna:
    assay: "bulk_rnaseq"
    value_type: "raw_counts"
    feature_id_type: "ensembl_gene_id"
  protein:
    assay: "mass_spec"
    value_type: "intensity"
    feature_id_type: "uniprot_accession"
files:
  - url: "https://.../rna_counts.tsv.gz"
    sha256: null
    modality: "rna"
  - url: "https://.../protein.tsv.gz"
    sha256: null
    modality: "protein"
sample_sheet: "config/samples.tsv"
human_review:
  status: "required"
  unresolved:
    - "Confirm whether R1 and P1 originate from the same biospecimen"
```

注意：URL 第一次下载后计算 SHA-256，并将其写入不可变 manifest。数据库如有官方 checksum，优先与官方值核对。

### 5.3 `samples.tsv` 必需字段

```text
sample_id  experimental_unit_id  subject_id  biospecimen_id  time  time_unit  condition  batch  modality  file_id  replicate_type
```

必须满足：

- `(subject_id, biospecimen_id, time, modality)` 不应出现无法解释的重复；
- 时间为数值，并保留原始时间标签；
- `experimental_unit_id` 必填；纵向设计中通常等于 subject，重复横断面设计中通常等于动物、培养皿或独立实验单位；
- `replicate_type` 明确为 biological 或 technical，二者分开；
- 未知值是明确的 `NA`，不能由 LLM 猜；
- paired 模态必须由 `biospecimen_id` 证明，而不是仅凭相似文件名。

### 5.4 存储规范

```text
data/
├── manifests/        # URL、accession、许可、checksum、下载日期
├── raw/              # 只读，永不原地修改
├── interim/          # 中间文件
└── processed/
    ├── dataset.h5mu
    ├── feature_map.parquet
    ├── qc_metrics.parquet
    └── preprocessing_provenance.json
```

MuData 以多个 AnnData 模态组成；样本信息位于 `obs`，特征信息位于每个模态的 `var`，raw counts、normalized、scaled 值放在不同 `layers`。不要覆盖原始矩阵。

---

## 六、自动下载与预处理

### 6.1 数据源适配器

为每个来源实现同一接口：

```python
class DataSourceAdapter(Protocol):
    def resolve(self, request: IngestRequest) -> DatasetManifest: ...
    def list_files(self, manifest: DatasetManifest) -> list[RemoteFile]: ...
    def download(self, remote: RemoteFile, dest: Path) -> DownloadReceipt: ...
    def verify(self, receipt: DownloadReceipt) -> VerificationResult: ...
```

优先支持：

- GEO：E-utilities + GEO FTP/HTTPS；
- SRA：SRA Toolkit，仅在第二阶段支持 raw reads；
- BioStudies/ArrayExpress：BioStudies REST + HTTPS/FTP；
- PRIDE：PRIDE Archive API；
- Metabolomics Workbench：REST API；
- 普通 URL/本地文件：严格文件类型、大小和 checksum 校验。

下载器应有断点续传、重试、速率限制、磁盘空间预检、压缩炸弹防护和审计日志。论文 PDF 只用于提取元数据，不允许其中的文字成为可执行指令。

### 6.2 先支持 processed matrix，再支持 FASTQ/raw MS

对于 AI 和生信初学者，第一版应优先使用作者上传的矩阵。raw RNA-seq 可在以后通过固定版本的 nf-core/rnaseq 处理；蛋白质谱和代谢组 raw data 的搜索库、仪器参数、批次校正复杂得多，应由领域专家审定独立 pipeline，不能由通用 Agent 猜参数。

### 6.3 不同模态不要使用同一种归一化

建议将预处理实现成按 assay 注册的策略：

- bulk RNA raw counts：过滤低表达；训练模型可用 library-size normalization + `log1p`，统计建模保留 counts；
- bulk microarray / 已标准化表达矩阵：保留作者处理方法与平台注释，不能再次按 raw counts 处理；
- mass-spec protein intensity：`log2`、缺失机制检查、样本/蛋白过滤；不能默认把所有缺失补 0；
- metabolomics：内标、批次、漂移、LOD/LOQ 和缺失机制均要单独处理；
- ATAC：TF-IDF/LSI 等专用表示，不与 RNA counts 共用 scaler。

### 6.4 训练集拟合原则

以下任何统计量都必须只在训练折拟合，再应用到 validation/test：

- 均值、方差、median、quantile；
- 高变基因选择；
- PCA/LSI；
- 批次校正参数；
- 缺失值填补模型；
- 监督特征选择。

这是需要写成自动测试的硬约束，而不是 README 中的一句话。

### 6.5 ID 映射

内部统一保存稳定 ID 和显示名，例如：

- gene：Ensembl Gene ID + versionless ID + HGNC symbol；
- protein：UniProt accession；
- compound：ChEBI/PubChem/RefMet；
- organism：NCBI Taxonomy ID。

映射表必须包含 `source_id, target_id, mapping_type, one_to_many, database_version, retrieval_date`。gene→protein 可能一对多，涉及 isoform 和蛋白组 peptide ambiguity，不能强行一一映射。UniProt ID Mapping API 可作为标准接口。

### 6.6 数据质控门禁

出现以下情况时停止，不进入建模：

- 少于 3 个时间点；
- 无法确认 `experimental_unit_id` 或独立生物重复单位；
- paired/pairing_level 声明无法被 biospecimen 元数据证明；
- 所有 test subject 的某个 modality 缺失；
- 下载 checksum 不一致；
- 时间或 condition 与 batch 完全混杂；
- 经 QC 后样本过少；
- feature ID 大量无法映射且超过配置阈值；
- 数据许可不允许预期的使用或再分发。

系统应输出 `data_readiness_report.html`，用红/黄/绿显示这些门禁，而不是悄悄继续。

---

## 七、预测任务的正式定义

先写任务，再写模型。建议一个实验只定义一个 primary endpoint。

### 7.1 纵向 bulk 样本预测

给定 subject `s` 在 `t1...tk` 的多模态观测：

```text
X_s(t≤tk) = {RNA, protein, metabolite, ...}
```

预测 `tk + Δ` 的目标模态：

```text
Y_s(tk+Δ) = {RNA and/or protein ...}
```

必须明确：history window、forecast horizon、推断时可用模态、目标 feature panel、缺失模态策略和允许的 condition/dose/cell type 协变量。

### 7.2 重复横断面 bulk 预测

输入早期各时间点的独立生物重复，预测后期同一 condition 下新重复的组学表达。模型可以预测：

- condition × time 的均值轨迹；
- 每个 feature 的均值和不确定性；
- 后期独立样本的多输出表达矩阵。

训练样本的构造不能把不同动物强行串成“同一个体轨迹”。推荐将时间作为连续协变量，使用混合效应模型、Gaussian Process、MEFISTO、动态因子模型或带 condition 输入的 latent ODE，并在独立实验单位或批次上做外层验证。

### 7.3 时间延迟

RNA 对蛋白的影响可能有转录、翻译和降解延迟。模型输入中应保留实际时间差，允许显式 lag 或连续时间状态；解释报告也必须写“预测贡献/统计依赖”，而不是“该基因直接导致该蛋白”。

---

## 八、模型路线：从基线到带先验的动力学模型

### 8.1 必做基线

按从简单到复杂注册：

1. `LastValue`：未来等于最后观察值；
2. `MeanTrajectory`：训练集中相同 condition 的平均轨迹；
3. `LinearRidge` / `ElasticNet`；
4. 时间 spline / Gaussian Process / linear mixed-effects（按实验设计选择）；
5. PLS regression、动态因子模型或 MEFISTO；
6. `VAR` 或每个目标的 autoregression（时间点和重复数足够时）；
7. `MLP`：固定 history window；
8. `GRU/LSTM`：有真实纵向序列时；
9. `NeuralCDE` 或 `ODE-RNN`：有真实纵向且时间不规则时；
10. `LatentODE`：连续潜变量动力学；
11. `PriorLatentODE`：加入图、通路和 embedding。

若复杂模型不能稳定优于 LastValue 和 Ridge，它就没有晋级资格。不要只挑最弱的基线。

模型注册表必须根据 bulk 采样设计自动限制候选：

| bulk 设计 | 可以使用 | 不应默认使用 |
|---|---|---|
| 同一 subject 纵向重复测量 | mixed-effects、Ridge、MLP、GRU、CDE、latent ODE | 把各时间点当独立随机样本 |
| 不同动物/样本的重复横断面 | spline/GP、MEFISTO、动态因子、condition-level latent ODE | 把不同动物串成 subject sequence 的 GRU/CDE |
| 仅 group-level 多模态对齐 | 通路/群体均值动力学、group-level 多视图模型 | sample-level RNA→protein 配对回归 |

bulk 数据通常是典型的“小样本、高维特征”。有效样本量由独立生物重复数决定，而不是基因或蛋白数量决定。因此优先在预注册 feature panel、通路活性或 16–64 维潜空间内建动力学；只有外部验证证明有收益时才增加模型容量。

### 8.2 推荐主模型

```text
每个模态 x_m(t)
       │
       ▼
modality-specific encoder E_m
       │  可用缺失掩码、batch、condition
       ▼
gated fusion / product-of-experts
       │
       ▼
共享潜状态 z(t_k)
       │
       ▼
 dz/dt = fθ(z, t, condition, prior_context)
       │  torchdiffeq / CDE
       ▼
未来潜状态 z(t_k + Δ)
       │
       ├── RNA decoder
       ├── protein decoder
       └── metabolite decoder
```

第一版 latent dimension 建议 16–64，而不是直接在两万个基因上积分 ODE。每个 decoder 使用适合该模态的 likelihood/loss。

### 8.3 损失函数

```text
L = Σ_m w_m · L_forecast_m
  + β · L_latent_regularization
  + λ_graph · L_graph
  + λ_sparse · L_sparsity
  + λ_consistency · L_cross_modal_consistency
```

- `L_forecast_m` 可用标准化后的 MSE/Huber，或适合 counts 的 NB likelihood；
- 模态权重不能让高维 RNA 完全淹没低维 protein；
- graph/sparsity 权重只能在训练/验证中确定；
- 输出 uncertainty 时可用 probabilistic decoder 或 ensemble。

### 8.4 如何融合 PPI、基因网络和通路

按风险从低到高逐步加入：

**A. 图拉普拉斯正则（第一选择）**

```text
L_graph = Σ_(i,j) w_ij ||h_i - h_j||²
```

优点是容易实现、可做消融、不会把先验当成绝对真理。

**B. Soft mask**：先验边对应的连接有较小惩罚，非先验边仍可学习但惩罚更大。不要使用完全 hard mask，因为知识库不完整且有场景差异。

**C. 多关系图网络**：分别保留 directed gene regulation、undirected physical PPI、functional association、pathway co-membership、gene→protein 编码关系。不同 edge type 使用不同参数。PPI 不自动代表方向或因果调控。STRING 应保留证据 channel、score、版本和物种；若目标是物理互作，不能混入所有 functional association 后仍称为 physical PPI。

**D. 通路 token/模块**：先将高维特征聚合为 Reactome/GO pathway activity，再让动力学在通路层运行，之后解码回 feature。这通常更适合小样本。

### 8.5 如何融合大模型 embedding

建议把 embedding 作为**冻结的辅助先验**，先投影到 16–64 维：

```text
e_i' = MLP(LayerNorm(e_i))
h_i = gate · learned_feature_embedding_i + (1-gate) · e_i'
```

可选来源：蛋白序列用 ESM/ESMC；基因侧优先使用由 GRN、GO/Reactome、共表达或同源信息得到的可追溯表示；化合物可用经过验证的分子表示模型。Geneformer、scGPT 等主要以单细胞语料训练的表示不属于 bulk MVP，除非以后通过专门的 domain validation 证明其对 bulk 任务有增益。第一版不要同时引入全部 embedding。

强制做四组消融：no prior、graph only、embedding only、graph + embedding；再加一个**度分布匹配的随机图负对照**。如果真实先验与随机图没有稳定差异，就不能声称先验知识带来生物学增益。

还要记录 embedding 模型、权重版本、训练数据描述、许可和提取层。基础模型可能已经学习过大量公开知识，因此后续“文献一致性”不完全是独立验证，应在报告中披露。

### 8.6 参考仓库的正确用法

以下项目适合学习或做模型适配器：`torchdiffeq`、Latent ODE、Neural CDE，以及适合 bulk 多组学和时间结构的 MOFA2/MEFISTO。scNODE、MIOFlow、PRESCIENT、totalVI/MultiVI 等单细胞方法不纳入当前模型注册表。

外部仓库接入规则：

1. 记录 URL、license、paper、commit SHA、最后审查日期；
2. fork 或 vendor 前由人工批准；
3. 在隔离容器安装，禁止访问不必要的凭据和目录；
4. 先在合成小数据执行单元/集成测试；
5. 通过统一 `ModelPlugin` 接口接入，不能让仓库改写 evaluator；
6. 与原论文数据处理差异必须写入 model card；
7. 引用模型时遵守仓库许可证和论文要求。

```python
class ModelPlugin(Protocol):
    name: str
    def fit(self, train: DataBundle, val: DataBundle, cfg: ModelConfig) -> FitResult: ...
    def predict(self, data: DataBundle) -> PredictionBundle: ...
    def save(self, path: Path) -> None: ...
    def explain(self, data: DataBundle, targets: list[str]) -> AttributionTable: ...
```

---

## 九、正确的数据划分和内部验证

### 9.1 三层数据隔离

- **outer test**：最终一次性评估；建模 Agent 平时看不到标签；
- **inner validation/CV**：模型选择和 Optuna 使用；
- **train**：拟合参数和预处理统计量。

`splits.parquet` 生成后保存 SHA-256，并由 evaluator 校验。Optimizer 的进程只接收 train/validation dataloader，不应有 test path。

### 9.2 推荐划分策略

1. 纵向设计按 subject 做 group split；重复横断面设计按独立实验/动物批次做 group split，并保留 `experimental_unit_id`；
2. 只有纵向设计才能在 subject 内构造真实 history→future 样本；横断面设计构造 condition-level/time-level 训练目标；
3. 若任务是跨时间外推，test 必须包含未用于训练目标的后期时间；
4. 若要泛化到新 condition，则按 condition 留一；它与“同 condition 的未来预测”是不同任务，必须分开报告。

不能简单随机拆 sample rows，因为同一 subject、同一实验批次和同一时间点的强相关观测会泄漏。若重复横断面数据只有一次实验且最晚时间点被完整留作 test，内部验证应使用更早时间点的 rolling-origin 方案；这种情况下 test 信息量有限，报告必须明确。

### 9.3 指标

至少报告 MSE、MAE、RMSE、PCC、Spearman、R²，以及：

- 每个样本 across-feature 指标；
- 每个 feature across-sample 指标；
- macro 平均和分布，而不只是 pooled 数字；
- 不同 forecast horizon、time、condition、batch 和 subject subgroup 的结果；
- bootstrap 95% confidence interval。

PCC 对尺度和偏移不敏感，因此 PCC 高并不代表绝对表达准确，必须和 MSE/MAE 同时看。高维数据的 pooled PCC 也可能被少数高方差基因主导。

重复横断面设计还应报告 condition × time 均值轨迹误差、独立生物重复之间的方差覆盖，以及按独立实验单位 bootstrap 的区间。缺失值指标只在真实可观测目标位置计算，并报告覆盖率。

### 9.4 模型比较与晋级

所有模型必须使用完全相同的 split、输入历史、feature set 和 evaluator。建议用 donor-level bootstrap 比较相对基线差值。

```yaml
promotion:
  primary_metric: "protein_macro_pcc"
  minimum_relative_improvement_over_ridge: 0.02
  must_beat_last_value: true
  seeds: [11, 22, 33, 44, 55]
  max_failure_rate: 0.0
  require_ci_excludes_zero: true
  max_parameter_multiplier_vs_baseline: 20
```

阈值应在实验开始前配置。样本很小时，同时展示 effect size、CI 和各 donor 结果。

---

## 十、自动优化但不“作弊”

### 10.1 Agent 可调整的参数白名单

encoder/decoder hidden size、latent dimension、ODE depth、learning rate、weight decay、dropout、graph/embedding/gating 权重、solver/tolerance 安全范围、history length、batch size、loss weights。

### 10.2 Agent 永远不能调整

train/validation/test membership、primary metric、feature ID 映射、test 标签、evaluator 代码、为某模型单独改变预处理，或在看到 test 后改变模型。

### 10.3 优化预算

```yaml
hpo:
  max_trials: 40
  max_wall_hours: 24
  max_concurrent_trials: 2
  sampler: "TPE"
  sampler_seed: 20260901
  pruner: "Hyperband"
  study_name: "dataset-task-model-v1"
  early_stopping_patience: 20
```

每个 trial 记录数据 hash、split hash、git commit、environment lock hash、随机种子、GPU、参数、训练曲线、验证指标和失败原因。选择最优配置后，用多个 seed 重训。不要把几十次 trial 的最好一个单次结果直接当最终性能。

### 10.4 自我优化循环

```text
Run candidate → Validate → Critic checks
    ├─ crash/NaN：只允许安全修复或缩小参数范围
    ├─ overfit：增加正则/降低容量/更早停止
    ├─ underfit：在预算内增加容量或改变 latent dynamics
    ├─ no baseline gain：执行消融，必要时退回简单模型
    └─ stable gain：冻结 config，等待 test 解锁审批
```

Agent 每轮必须输出结构化 `OptimizationDecision`，包含 evidence、proposed_change、expected_effect、cost 和 stop_reason。不得只输出自然语言。

---

## 十一、可解释性和“新发现”流程

### 11.1 预测贡献不等于因果

“哪些基因对蛋白预测贡献大”可以通过模型 attribution 回答；“该基因调控该蛋白”需要干预或更强因果设计。报告中统一使用“对模型预测具有稳定贡献”“与预测方向一致/相反”“提出可检验假设”，不要写“证明调控”。

### 11.2 三种方法交叉验证

对最终冻结模型：

1. Integrated Gradients（Captum），至少用多个合理 baseline；
2. feature/group ablation：屏蔽基因、通路或模态，测目标预测变化；
3. permutation：在 donor/condition 合理分层内打乱特征，测性能下降。

若模型含图，还可做 edge ablation；注意力权重仅作辅助，不能单独作为解释。

### 11.3 稳定性筛选

每个 gene→target protein 候选至少跨 5 个随机种子、outer folds 或 donor bootstrap、多个 IG baseline，以及至少一种扰动法。保存平均 attribution、符号一致率、rank 中位数、selection frequency、bootstrap CI、ablation delta。只有达到预注册稳定阈值的候选才进入文献检索。

### 11.4 先验独立性

候选标记为：

- `prior_edge_used=true`：该关系直接/间接存在于训练先验；
- `embedding_supported=true`：embedding 可能包含相关知识；
- `de_novo_model_edge=true`：模型在无该边约束下也稳定出现。

如果候选本来就被 PPI/GRN 喂给模型，模型再把它排在前面不能作为独立生物发现。更有价值的是：去掉直接边仍稳定、随机图对照不出现、跨 seed/fold 保持，并有合理时序 lag。

### 11.5 文献核验 Agent

只给它少量稳定候选，例如 top 20–50。每个候选构造包含标准 ID、同义词、物种、组织、condition、关系词的检索式。通过 PubMed E-utilities/Europe PMC 获取摘要、引用和可用全文元数据。

```text
candidate_id
source_gene / target_protein
predicted_direction / time_lag
query_string / searched_at
pmid / doi / title / year
evidence_sentence_or_paraphrase
relation_type
same_species / same_tissue / same_condition
supports / contradicts / unrelated
evidence_level
reviewer_status
```

证据分级：

- **A**：同物种、相近场景的直接 perturbation/causal 实验；
- **B**：直接结合、调控或物理互作证据；
- **C**：同组织/疾病中的可靠相关或共同变化；
- **D**：通路/知识库或其他场景的间接支持；
- **N**：在预注册检索范围内未找到直接证据；
- **X**：存在相反证据。

“N”必须写成“在本次检索范围内未找到直接证据”，不能写成“全新发现”。LLM 抽取后应由第二个核验步骤确认 PMID、标题、关系方向和原文是否匹配；优先引用原始研究。

### 11.6 湿实验候选优先级

```text
priority =
  0.30 × attribution_stability
+ 0.20 × perturbation_effect
+ 0.15 × out_of_fold_reproducibility
+ 0.10 × temporal_precedence
+ 0.10 × biological_plausibility
+ 0.10 × novelty_with_caveat
+ 0.05 × experimental_feasibility
```

必须展示各分项。进入湿实验前，再检查 target 可操作性、试剂、细胞模型、毒性、正负对照、剂量与时间窗；由领域专家批准。

---

## 十二、项目目录和核心命令

```text
omics-dynamics-agent/
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── config/
│   ├── dataset.example.yaml
│   ├── experiment.example.yaml
│   ├── models/
│   └── priors/
├── workflow/
│   ├── Snakefile
│   ├── rules/
│   ├── envs/
│   └── scripts/
├── src/omics_agent/
│   ├── cli.py
│   ├── schemas/
│   ├── agents/
│   ├── data_sources/
│   ├── preprocessing/
│   ├── splitting/
│   ├── models/
│   ├── priors/
│   ├── evaluation/
│   ├── optimization/
│   ├── interpretation/
│   ├── literature/
│   └── reporting/
├── tests/{unit,integration,synthetic,golden}/
├── data/              # gitignored; DVC/checksum managed
├── mlruns/
└── reports/
```

```bash
omics-agent doctor
omics-agent ingest --accession GSEXXXX
omics-agent review-manifest config/dataset.yaml
omics-agent preprocess --dataset config/dataset.yaml
omics-agent split --experiment config/experiment.yaml
omics-agent benchmark --experiment config/experiment.yaml
omics-agent optimize --model prior_latent_ode --experiment config/experiment.yaml
omics-agent evaluate --checkpoint frozen_model.ckpt --unlock-test TOKEN
omics-agent explain --run-id RUN_ID
omics-agent literature-check --candidates reports/candidates.parquet
omics-agent report --run-id RUN_ID
```

`--unlock-test` 应迫使用户明确执行一次性测试步骤，并在 MLflow 记录时间与模型 hash。

---

## 十三、最小代码骨架

```python
from enum import StrEnum
from pathlib import Path
from pydantic import BaseModel, Field

class Stage(StrEnum):
    INGESTED = "ingested"
    NEEDS_REVIEW = "needs_review"
    READY = "ready"
    PREPROCESSED = "preprocessed"
    SPLIT_LOCKED = "split_locked"
    BASELINED = "baselined"
    OPTIMIZED = "optimized"
    FROZEN = "frozen"
    TESTED = "tested"
    EXPLAINED = "explained"

class ResearchState(BaseModel):
    dataset_id: str
    stage: Stage
    manifest_path: Path
    data_hash: str | None = None
    split_hash: str | None = None
    frozen_model_hash: str | None = None
    unresolved: list[str] = Field(default_factory=list)
```

每个 stage transition 都应有前置条件和审计事件。不要用一段长 prompt 维持系统状态。

```python
def assert_no_group_leakage(split_df, group_col="experimental_unit_id"):
    groups = {
        name: set(part[group_col])
        for name, part in split_df.groupby("split")
    }
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
```

还需检查 preprocessing artifacts 的 `fit_split == train`、时间 cutoff、重复 biospecimen 和 file checksum。PCC 在常数向量上未定义，必须返回 NA 并报告有效样本数，不能偷偷填 0。

---

## 十四、测试与验收标准

### 14.1 合成数据

建立一个已知真值的小型系统：20 个 gene、10 个 protein、5 个时间点、已知稀疏 gene→protein lagged network、两种 condition、多个 subject、batch noise 和缺失值；可用线性 ODE 或 Lotka–Volterra 生成 latent trajectory。

验收：模型能运行、未来预测优于 LastValue、解释性恢复一部分真边、随机图消融不能表现同样好。合成数据只用于发现工程错误，不是生物学证明。

### 14.2 必须通过的自动测试

- manifest schema 拒绝缺 time/unit/organism；
- checksum 错误会中止，raw 文件不会被修改；
- ID 一对多不会被静默丢弃；
- train/val/test group 无交集；
- scaler/HVG/PCA 只 fit train；
- evaluator 对所有模型一致；
- 常数特征 PCC 返回 NA；
- masked metrics 覆盖率正确；
- test labels 对 optimizer 不可见；
- 固定 seed 的小实验可复现到容差范围；
- MLflow run 包含 data/split/code/environment hash；
- 文献记录不存在伪造 PMID/DOI；
- report 区分 prediction、association、hypothesis 和 causal evidence。

### 14.3 每阶段 Definition of Done

**数据阶段**：`dataset.yaml` 经人工确认；checksum 完整；QC 通过；`.h5mu` 可加载。

**基线阶段**：LastValue、Ridge、MLP/GRU 至少三个基线在相同 split 完成；指标可复现。

**模型阶段**：latent ODE 能处理不规则时间和 missing mask；5 seeds；无 NaN；有消融。

**评估阶段**：test 只解锁一次；结果含 CI、分层指标和失败案例；不只报告最优 seed。

**解释阶段**：至少两类解释方法；稳定性表；prior-edge 标记；文献证据逐条可追溯。

---

## 十五、实施路线图

### 第 0 阶段：问题和数据审计（1–2 周）

选一个数据集和一个主要目标，例如“用 t0、t1 的 bulk RNA+protein 预测 t2 bulk protein”；手工确认样本表；决定是纵向重复测量还是重复横断面，并确认 bulk 模态的 pairing_level；写两个 YAML；先用 20–100 个 feature 跑通。

### 第 1 阶段：可复现基线（2–4 周）

完成下载/checksum/parser、MuData 输出、split guard、LastValue/Ridge/MLP、MLflow、HTML report 和合成测试。此阶段完全不需要 LLM 自动改模型。

### 第 2 阶段：动力学模型（3–6 周）

小型 latent ODE、实际 `Δt`、multi-decoder、CDE/ODE-RNN 比较、Optuna 白名单调参、多 seed/ablation/locked test。

### 第 3 阶段：先验知识（3–6 周）

按顺序加入 Reactome pathway → graph Laplacian → soft mask/多关系图 → 一个 embedding。每次只加一种，并与 no-prior/random-prior 比较。

### 第 4 阶段：解释与文献（2–4 周）

IG + ablation + permutation；fold/seed/bootstrap 稳定性；top 候选文献检索；证据分级和人工核验；生成湿实验假设。

### 第 5 阶段：扩展

raw FASTQ/MS pipeline、多数据集外部验证、perturbation 泛化、Web UI、HPC/云、独立文献 Agent、容器和 DVC remote。

对初学者，完整可靠版本通常是数月工程，不是一段 prompt 在几天内完成。编码助手能加速写代码，但不能替代样本设计、数据映射和生物学审核。

---

## 十六、交给 Cursor/Claude 前要准备的答案

1. 第一个数据集 accession/DOI/URL 是什么？
2. bulk 数据来自纵向重复测量，还是每个时间点使用不同生物重复的横断面设计？
3. 哪些 bulk 模态？每个模态值是什么（raw counts、已标准化表达或 intensity）？
4. 同一 subject/biospecimen 是否跨时间？模态配对层级是 same aliquot、same biospecimen、same subject-time，还是仅 group-level？
5. 有多少 subject、condition、time point、biological replicate？
6. 第一版预测哪个目标模态和哪个 horizon？
7. 有多少磁盘、RAM、GPU 和运行时间预算？
8. 是否涉及未公开/受控人类数据？

如果这些不知道，编码助手的第一项任务应该是生成 `data_readiness_report`，而不是开始写 ODE。

---

## 十七、可直接粘贴给 Cursor/Claude 的主提示词

```text
你是一名资深 Python/ML 工程师和谨慎的计算生物学工程师。请在当前空仓库中搭建一个“时序多组学预测科研流水线”。这不是自由自治、可任意执行网络代码的 Agent，而是确定性流水线 + 受约束的 LLM 编排层。

总体目标：
输入 bulk 多组学的 GEO/PRIDE/BioStudies/论文 DOI/URL/本地 processed matrix，生成经人工确认的 dataset manifest；下载并 checksum；按 bulk assay 预处理为 AnnData/MuData；按 experimental unit/subject/time 做无泄漏划分；训练 LastValue、Ridge、时间 spline/混合效应、PLS/动态因子、MLP 及适用条件下的 GRU/latent ODE；只用 validation 做 Optuna 调参；冻结后一次性评估 test；用 Integrated Gradients、feature ablation、permutation 做稳定解释；为稳定 gene→protein 候选生成可核验的 PubMed/Europe PMC 证据表。

硬性科研约束：
1. 同一 experimental unit/subject/biospecimen 不得跨 train/val/test；重复横断面设计还要防止 experiment batch 泄漏。
2. scaler、imputer、HVG、PCA、batch correction 只能 fit train。
3. test 标签对 optimizer 不可见；split/evaluator/primary metric 不允许 Agent 修改。
4. 重复横断面 bulk 数据不得把不同动物/样本串成同一个体轨迹；`group_level_only` 模态不得伪装成样本级配对。
5. PCC 与 MSE/MAE 同报；输出 per-sample、per-feature、macro、coverage 和 bootstrap CI。
6. 所有输入、代码、环境、数据、split、prior 和模型均记录 hash/version/seed。
7. 外部仓库必须固定 commit、记录 license、容器隔离并通过 toy-data adapter test；不得自动执行陌生 install/shell 脚本。
8. 解释值不是因果。文献未检索到只能写“在本次检索范围内未找到直接证据”，不能写“首次发现”。
9. PPI/GRN/pathway/embedding 必须做 no-prior、单 prior、组合 prior、degree-matched random graph 消融。
10. 遇到样本映射、时间、配对或许可不确定时，停止并产出 needs_review，不得猜。

技术选择：
- Python 3.11、uv、Pydantic、Typer、Snakemake、pytest、Ruff、mypy；
- AnnData/MuData + Parquet；
- PyTorch（只选择纯 PyTorch 或 Lightning 其中一种并保持一致）；
- torchdiffeq，必要时 torchcde；
- Optuna、MLflow、Captum；
- 配置均为 YAML，业务对象均用 Pydantic schema。

工程结构：
建立 src/omics_agent 下的 schemas、data_sources、preprocessing、splitting、models、priors、evaluation、optimization、interpretation、literature、reporting、agents；建立 workflow、config、tests/synthetic、tests/integration 和 CLI。raw 数据只读，outputs 不覆盖 inputs。

模型接口：
所有模型实现 fit/predict/save/explain；evaluator 由独立模块统一调用。基线至少含 LastValue、Ridge、MLP；第二里程碑再加 GRU、latent ODE。主模型使用 modality-specific encoder → gated fusion → latent state → continuous dynamics → modality-specific decoder，并支持 actual delta_t、missing modality mask、condition covariates。

实现方式：
- 先阅读仓库中的 AGENTS.md/README/已有代码并报告发现；
- 先给出不超过 8 步的实施计划；
- 每次只完成当前 milestone，展示修改文件与验证命令；
- 不留伪实现、静默 fallback 或仅能跑 happy path 的代码；
- 所有网络与重计算步骤提供 dry-run；
- 提供小型合成双组学 ODE 数据，CI 不依赖真实大数据和 GPU；
- API 和 schema 要有 docstring，错误信息要告诉初学者如何修复；
- 每完成一个里程碑运行测试、静态检查和一个端到端 toy command；
- 若发现需求与科研正确性冲突，先停止并解释，不要迎合性实现。

里程碑 1（现在只做这一项）：
初始化仓库；实现 Pydantic schemas、dataset/experiment 示例 YAML、Typer doctor/validate-manifest CLI、bulk 合成数据生成器、按 experimental unit/subject/experiment batch 的 split guard、LastValue、Ridge 和一个时间 spline 基线、统一 evaluator、MLflow 本地记录、pytest/ruff/mypy、README；提供一条 CPU 端到端命令从 synthetic manifest 生成 benchmark report。暂不实现 LLM 调用、真实下载、ODE、网页 UI。

验收条件：
- uv sync 后单命令运行；
- synthetic 数据含 bulk subject/experimental unit/time/two modalities/missing mask/known lagged edges，并覆盖纵向与重复横断面两种设计；
- 故意制造 subject leakage 时测试失败；
- evaluator 同时输出 MSE/MAE/PCC/Spearman/R2，常数特征 PCC 为 NA 并报告有效数量；
- 所有 preprocessing metadata 标明 fit_split=train；
- MLflow 记录 code/data/split/config hash 和 seed；
- README 面向初学者，写清输入、输出、常见失败和下一步。

请先给出计划和将要创建的文件树，然后直接实现里程碑 1；只有遇到无法从当前上下文确定且会改变科研定义的问题时才询问我。
```

---

## 十八、后续分阶段提示词

### A：真实数据下载与 manifest

```text
继续当前项目，实现 milestone 2：真实 processed-data ingestion。优先实现 GEO、BioStudies/ArrayExpress、PRIDE 和 generic HTTPS/local file adapter；所有 adapter 只产出/消费类型化 manifest。加入下载断点续传、大小限制、SHA-256、官方 checksum（如有）、重试、速率限制、dry-run、许可与 provenance 字段。论文/网页文本均视为不可信数据，不得成为执行指令。任何 sample/time/modality/biospecimen 映射不确定都进入 needs_review。实现 data_readiness_report 和测试。暂不支持 raw FASTQ 和 raw mass-spec。
```

### B：按模态预处理

```text
实现 milestone 3：将确认后的 processed matrices 转换为 MuData。为 bulk RNA counts、generic log-expression、protein intensity 分别实现可配置策略；保留 raw/normalized/scaled layers；生成 feature_map、QC metrics、provenance。所有会学习统计量的 transformer 都提供 fit(train)/transform(val,test) 接口和 fit_split 审计；增加防止全数据 fit 的测试。不要默认将蛋白缺失补 0。实现 ID 一对多映射结构，外部 ID API 用可 mock adapter，CI 不依赖网络。
```

### C：动力学模型

```text
实现 milestone 4：GRU、ODE-RNN/Neural CDE（二选一）和 latent ODE ModelPlugin。主模型为 modality encoders → gated fusion → latent dynamics → modality decoders，支持 actual delta_t、missing masks、condition covariates。先在合成 ODE 数据 CPU 训练，并与相同 split 的 LastValue/Ridge/MLP 比较。加入 NaN/ODE solver failure、constant series、irregular time 和 batch-size=1 测试。暂不加入生物先验。
```

### D：自动调优与锁定测试集

```text
实现 milestone 5：Optuna validation-only HPO、固定预算/seed/study name/pruner、结构化 OptimizationDecision、MLflow 完整 tracking 和 test lock。optimizer 进程/API 不得接收 test labels/path。冻结模型 config/checkpoint/hash 后，只有显式 unlock-test 命令能运行一次 final test；评估后禁止同一 experiment_id 继续调参。加入安全回归测试，证明修改 split/evaluator/hash 会被拒绝。
```

### E：生物先验

```text
实现 milestone 6：版本化 PriorBundle 和三种可消融先验：Reactome pathway features、带 edge type/evidence/score 的 graph Laplacian、冻结 embedding projection/gate。不要把 STRING functional association 标成 physical/causal。实现 no-prior、graph-only、embedding-only、combined 和 degree-matched random-graph 配置；所有配置使用同一 evaluator/split/HPO budget。输出效果差、参数量、运行时间和多 seed CI 对照表。
```

### F：解释性和文献证据

```text
实现 milestone 7：对冻结模型做 Captum Integrated Gradients（多个 baseline）、group feature ablation 和 stratified permutation；跨 seed/fold/bootstrap 输出 attribution stability。候选表必须标记 prior_edge_used、embedding_supported 和 ablation_delta。仅对通过稳定阈值的 top-N 候选调用 PubMed E-utilities/Europe PMC adapter；保存 query/date/PMID/DOI/关系方向/场景/supports-or-contradicts/A-B-C-D-N-X level。添加 PMID/DOI 真实性校验和人工 reviewer_status。报告只能称 hypothesis，不得由 absence of evidence 宣称 novelty/causality。
```

### G：独立代码审计

```text
请作为独立审计者审查整个项目，不新增功能。重点寻找：experimental-unit/subject/time/batch leakage、把重复横断面样本伪装为纵向个体、把 group-level 模态伪装成 sample-level paired、全数据 preprocessing fit、optimizer 接触 test、pooled metric 误导、PPI 类型误标、ID 一对多静默丢失、缺失值被当 0、随机种子不完整、外部仓库未固定版本、解释性被写成因果、伪造或不匹配文献引用。按 P0/P1/P2 给出具体文件和行号；为每个 P0/P1 添加能先失败的测试，再做最小修复，运行全套验证。
```

---

## 十九、如何实际使用编码助手

1. 新建 Git 仓库，把本指南放入 `docs/`。
2. 将主提示词交给 Cursor/Claude，只做 milestone 1。
3. 要求运行全部测试和 toy end-to-end，不接受“理论上可运行”。
4. 将错误日志原样给它，不要只说“运行失败”。
5. 每个 milestone 单独 commit；确认能复现再继续。
6. 到真实数据阶段，把 accession 和论文补充表一起提供，但自己/合作者必须核对 samples.tsv。
7. 第一次真实实验只用小 feature set 和小预算；基线跑通后再扩大。
8. 请生信同事审核预处理、ML 同事审核 split/evaluation、湿实验专家审核候选。

不要让同一个 Agent 同时充当模型作者、裁判和“发现宣布者”。Evaluator、split guard 和 literature verifier 要保持独立。

---

## 二十、权威资料与参考实现

### 数据获取与数据结构

- [NCBI GEO programmatic access](https://www.ncbi.nlm.nih.gov/geo/info/geo_paccess.html)
- [NCBI GEO download guide](https://www.ncbi.nlm.nih.gov/geo/info/download.html)
- [NCBI SRA Toolkit](https://www.ncbi.nlm.nih.gov/home/tools/)
- [BioStudies/ArrayExpress](https://www.ebi.ac.uk/biostudies/arrayexpress/)
- [BioStudies REST/download help](https://www.ebi.ac.uk/biostudies/SourceData/help)
- [PRIDE Archive API](https://www.ebi.ac.uk/pride/ws/archive/v2/docs/api-guide.html)
- [Metabolomics Workbench REST API](https://www.metabolomicsworkbench.org/tools/MWRestAPIv1.2.pdf)
- [AnnData documentation](https://anndata.readthedocs.io/en/stable/tutorials/notebooks/getting-started.html)
- [MuData specification](https://mudata.readthedocs.io/stable/io/spec.html)
- [UniProt ID mapping](https://www.uniprot.org/help/id_mapping_prog)

### 可复现流水线与实验记录

- [Snakemake reproducibility/deployment](https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html)
- [nf-core documentation](https://nf-co.re/docs)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking)
- [Optuna reproducibility FAQ](https://optuna.readthedocs.io/en/stable/faq.html)
- [DVC workflow/reference](https://dvc.org/doc/command-reference/)

### 多组学与连续时间模型

- [MOFA2/MEFISTO](https://biofam.github.io/MOFA2/)
- [MEFISTO paper](https://www.nature.com/articles/s41592-021-01343-9)
- [Neural ODE paper](https://arxiv.org/abs/1806.07366)
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
- [Latent ODE paper](https://papers.nips.cc/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html)
- [statsmodels linear mixed-effects models](https://www.statsmodels.org/stable/mixed_linear.html)

### 生物先验、解释性和文献

- [STRING API](https://string-db.org/help/api/)
- [Reactome downloads/API](https://reactome.org/download-data/)
- [ESM protein models](https://github.com/evolutionaryscale/esm)
- [Geneformer](https://github.com/epigen/Geneformer)
- [Captum Integrated Gradients](https://captum.ai/api/integrated_gradients.html)
- [Captum algorithms/quality metrics](https://captum.ai/docs/attribution_algorithms)
- [SHAP documentation](https://shap.readthedocs.io/en/stable/)
- [NCBI E-utilities APIs](https://www.ncbi.nlm.nih.gov/home/develop/api/)
- [Europe PMC REST API](https://europepmc.org/RestfulWebService)

---

## 二十一、科研底线清单

在相信结果前，逐项回答“是”：

- 数据的独立重复单位明确吗？
- 多模态是真配对还是推断配对，报告写清了吗？
- 训练/验证/测试按 donor 隔离了吗？
- 所有预处理只在训练集拟合了吗？
- 至少击败 LastValue 和 Ridge 吗？
- 结果跨 seed/fold/donor 稳定并有 CI 吗？
- 先验和 embedding 做了无先验及随机先验消融吗？
- 最终 test 在模型冻结后才解锁吗？
- 解释性至少由梯度法和扰动法共同支持吗？
- 候选是否标记了原本就在先验中的关系？
- 每条文献证据的 PMID/DOI、方向和场景都人工抽查了吗？
- “未找到证据”是否只被称为待验证假设？

任意关键项为“否”时，系统可以生成探索性结果，但还不能支撑可靠生物学结论或“新发现”声明。
