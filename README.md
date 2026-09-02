# omics-agent

确定性的 **bulk 时序多组学预测流水线**。它不是会自己下载陌生代码、也不会猜测样本对应关系的 Agent。

里程碑 1：从合成双组学数据出发，按实验单位划分、训练三个基线，写出带 hash 的 benchmark 报告。

里程碑 2：从 GEO / BioStudies / PRIDE / HTTPS / 本地 processed matrix 生成类型化 ingest manifest，下载并校验，写出 data-readiness 报告。**不会猜测** sample/time/配对，也**不会**把论文 HTML 当成指令执行。

里程碑 3：把人工批准后的 processed matrices 转成 MuData（raw / normalized / scaled 三层），按 assay 选择策略（bulk RNA counts → CPM+log1p；log-expression → 原样；protein intensity → 0 视为缺失 + log2），生成 QC 指标、feature_map 和 preprocessing provenance。**蛋白缺失不会被补 0**；所有学习统计量的 transformer 只在 train 上 fit。

## 你需要什么

- Python 3.11（`uv` 会帮你安装）
- CPU 即可，不需要 GPU
- 合成 toy / 测试不需要网络；真实 GEO / BioStudies / PRIDE ingest 才访问仓库 API
- 本仓库（已包含 `docs/自动化时序多组学科研Agent搭建指南.md`）

## 一条命令跑通

```bash
# 1. 安装 uv（若还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 进入仓库，同步环境（含开发工具）
uv sync --extra dev

# 3. 检查依赖
uv run omics-agent doctor

# 4. 端到端 toy：纵向 + 重复横断面合成数据 → 划分 → 基线 → 报告
uv run omics-agent run-toy --output-dir outputs/toy
```

跑完后看：

- `outputs/toy/longitudinal/run/reports/benchmark.md`
- `outputs/toy/rcs/run/reports/benchmark.md`
- `outputs/toy/*/run/mlruns/`（本地 MLflow，记录 code/data/split/config hash 和 seed）

校验训练清单：

```bash
uv run omics-agent validate-manifest config/dataset.example.yaml
```

解析真实仓库（默认 dry-run 安全；CI 用 mock，不打真实网络）：

```bash
# 只演练：不写文件、不下载
uv run omics-agent ingest --accession GSE12345 --dest outputs/ingest/GSE12345 --dry-run

# 本地 processed matrix：写入 ingest_manifest + readiness（仍为 needs_review）
uv run omics-agent ingest --source local --local-path path/to/counts.tsv --modality rna --dest outputs/ingest/local

uv run omics-agent data-readiness outputs/ingest/local/ingest_manifest.yaml
```

论文 DOI **不是**文件定位符，流水线不会去抓 PDF：

```bash
uv run omics-agent ingest --paper-doi 10.1234/xxxx --dest outputs/ingest/doi --dry-run
```

ingest 写出（dry-run 不写盘）：

| 路径 | 内容 |
|---|---|
| `ingest_manifest.yaml` | 类型化 adapter 输出：文件角色、许可、provenance、`needs_review` |
| `raw/` | 已下载的 processed 文件（SHA-256；有官方 checksum 则核对） |
| `data_readiness_report.html` / `.json` | 红/黄/绿灯；红灯阻止训练，不会自动补全映射 |

不确定的 sample / time / modality / biospecimen 一律留在 `unresolved`，不会被猜成训练集。FASTQ、mzML、vendor `.raw` 只标 `rejected_raw`，不下载。zip/tar 不自动解压。

批准后的数据 → MuData（里程碑 3）：

```bash
uv run omics-agent preprocess --experiment config/experiment.example.yaml --output-dir outputs/prep
```

写出：

| 路径 | 内容 |
|---|---|
| `dataset.h5mu` | 每个模态含 `raw` / `normalized` / `scaled` 三层；`X` 是 scaled |
| `qc_metrics.json` | per-sample / per-feature 缺失率、信号总量、零方差 feature |
| `preprocessing_provenance.json` | 无状态 per-sample 步骤标 `learns_statistics: false`；fit 过的 scaler 一律 `fit_split: train` |
| `feature_map.json` | 一对多映射显式保留；映射不到就记 `unmapped`，不猜 |

策略按 manifest 的 `value_type` 自动选择（`raw_counts` → CPM+log1p；`intensity` → 0 视为缺失 + log2；log 类 → 原样），也可以在 `experiment.yaml` 的 `preprocessing.per_modality` 覆盖。`value_type: undeclared` 会直接报错，不会猜。加 `--id-map map.tsv`（列：`modality  source_id  target_id  [target_id_type]`）用离线表映射 ID；外部 API（mygene.info）adapter 走可注入的 HTTP transport，CI 全部 mock。

只演练、不写文件：

```bash
uv run omics-agent generate-synthetic --output-dir outputs/toy --design both --dry-run
uv run omics-agent benchmark --experiment config/experiment.example.yaml --dry-run
```

## 输入是什么

流水线 **只接受已经人工确认的 `dataset.yaml`**。合成生成器会写出一份 `status: approved` 的清单，因为样本、时间和配对都是已知的。

真实数据（里程碑 2）必须先有：

| 文件 | 作用 |
|---|---|
| `dataset.yaml` | 来源、许可、物种、设计、模态、文件、人工审核状态 |
| `samples.tsv` | `sample_id observation_id experimental_unit_id subject_id biospecimen_id time time_unit condition batch modality file_id replicate_type` |
| 每个模态一份矩阵 | 行为 `sample_id`，列为 feature |

两种实验设计不能混用：

- **纵向（longitudinal）**：同一个体在多个时间点取样。任务是 `subject_forecast`（用该个体的过去预测未来）。
- **重复横断面（repeated_cross_sectional）**：每个时间点是不同动物。任务只能是 `group_time_forecast`。禁止把不同动物串成一条个体轨迹。

合成数据包含：12 个 gene、8 个 protein、不规则时间 `{0,1,2,4,8}`、两种 condition、缺失掩码、以及写在 `true_edges.parquet` 里的已知滞后边。

## 输出是什么

`outputs/<run>/` 下：

| 路径 | 内容 |
|---|---|
| `splits.parquet` | 锁定的 train/val/test，按实验单位（RCS 再按 experiment batch） |
| `dataset.h5mu` | 预处理后的 MuData；scaler 只在 train 上 fit |
| `preprocessing_provenance.json` | 每条记录都有 `fit_split: train` |
| `models/<name>/` | LastValue / Ridge / time_spline |
| `reports/benchmark.md` | MSE / MAE / RMSE / PCC / Spearman / R2，含 per-sample、per-feature、macro、coverage、bootstrap CI |
| `mlruns/` | 本地 MLflow |

常数特征的 PCC 是 **NA**，同时报告有效特征数，不会偷偷写成 0。

## 常见失败

| 现象 | 原因 | 怎么修 |
|---|---|---|
| `needs human review` | `human_review.status` 不是 `approved`，或还有 `unresolved` | 人来核对 sample/time/配对/许可，不要让程序猜 |
| `leaks across train and test` | 同一个体或同一 biospecimen 出现在两个 split | 按 `experimental_unit_id` 划分，不要按行随机拆 |
| `illegal for repeated cross-sectional` | 对处死取材数据用了 `subject_forecast` | 改成 `group_time_forecast`，并给每个动物唯一 unit id |
| `fitted on train` 报错 | scaler/imputer 看到了 val/test 行 | 只对 train 调用 `fit()` |
| `group_level_only` 报错 | 把仅 condition×time 对齐的模态当成样本配对 | 不要做样本级 RNA→protein 回归 |
| `spline_df is too large` | 样条自由度 ≥ 训练时间点数 | 把 `spline_df` 调到 `n_times - 1` 以下 |
| `doctor` 失败 | 依赖没装上 | 在仓库根目录执行 `uv sync --extra dev` |
| 想改 primary metric 让模型“更好看” | 不允许 | 换一个新的 `experiment_id`，不要改 evaluator |
| ingest 后 readiness 全是红灯 | 正常：adapter 不会猜设计/配对/许可 | 人填写 `samples.tsv` 并批准 `dataset.yaml` 后再训练 |
| `looks like raw sequencing` | 指向了 FASTQ / mzML / vendor .raw | 改用作者提供的 processed matrix |
| Official checksum mismatch | 下载字节与仓库公布的摘要不一致 | 不要使用该文件；重新下载或核对官方 checksum |
| 只给了论文 DOI | DOI 不能定位矩阵 | 再提供 GSE/PXD/E-MTAB 或矩阵 URL |
| `no preprocessing strategy can be chosen` | `value_type: undeclared` | 人工确认矩阵是 counts 还是 intensity 后写进 manifest |
| `declared raw counts but contains negative values` | 矩阵其实已经 log/中心化 | 把 `value_type` 改成 log 类，用 log_expression 策略 |
| 蛋白某些值变成了 NaN | 0 强度被视为「未定量」 | 这是有意的；不要求补 0。模型输入需要填充时用 train-mean 且只对输入 |

## 验证

```bash
uv run pytest
uv run ruff check src tests
uv run mypy src/omics_agent
uv run omics-agent run-toy --output-dir outputs/toy
```

故意制造 subject 泄漏时，`assert_no_group_leakage` 必须失败（见 `tests/synthetic/test_split_guard.py`）。

## 下一步（还没做，也不会假装做了）

1. GRU / latent ODE（纯 PyTorch，不用 Lightning）  
2. 仅 validation 的 Optuna，以及一次性 unlock-test  
3. 先验消融与解释性、文献核验  

解释值不是因果。文献以后若检索不到，只能写「在本次检索范围内未找到直接证据」。

## 设计选择

- 深度学习框架：**纯 PyTorch**（里程碑 4 才引入；里程碑 1 基线用 sklearn / patsy）
- 配置全部 YAML，业务对象全部 Pydantic
- `data/raw/` 只读；结果只写到 `outputs/`
