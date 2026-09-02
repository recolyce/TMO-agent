# omics-agent

确定性的 **bulk 时序多组学预测流水线**。它不是会自己下载陌生代码、也不会猜测样本对应关系的 Agent。

里程碑 1：从合成双组学数据出发，按实验单位划分、训练三个基线，写出带 hash 的 benchmark 报告。

里程碑 2：从 GEO / BioStudies / PRIDE / HTTPS / 本地 processed matrix 生成类型化 ingest manifest，下载并校验，写出 data-readiness 报告。**不会猜测** sample/time/配对，也**不会**把论文 HTML 当成指令执行。

里程碑 3：把人工批准后的 processed matrices 转成 MuData（raw / normalized / scaled 三层），按 assay 选择策略（bulk RNA counts → CPM+log1p；log-expression → 原样；protein intensity → 0 视为缺失 + log2），生成 QC 指标、feature_map 和 preprocessing provenance。**蛋白缺失不会被补 0**；所有学习统计量的 transformer 只在 train 上 fit。

里程碑 4：纯 PyTorch 的时序动力学模型 `gru` / `ode_rnn` / `latent_ode`（modality encoders → gated fusion → latent dynamics → modality decoders），支持真实 delta_t、缺失掩码和 condition 协变量，外加一个 sklearn `mlp` 基线。**只对纵向 subject_forecast 合法**——重复横断面会直接报错，不会把不同动物拼成轨迹。ODE 用自带的固定步长 RK4；solver 出现 NaN/inf、训练 loss 发散都会抛出类型化错误而不是继续汇报垃圾结果。

里程碑 5：Optuna **仅 validation** 调参——固定预算、固定 sampler seed、固定 study name、median pruner，全程 MLflow tracking，产出结构化 `OptimizationDecision`。**optimizer 的目标函数拿不到 test 行**（有运行时防护 + 回归测试证明）。调参结束自动冻结 best config / checkpoint / 全套 hash；只有显式 `unlock-test --confirm` 能跑**一次** final test（评分前先烧掉锁，fail-closed）；之后同一 `experiment_id` 的调参和再测一律拒绝。任何对 split / checkpoint / 冻结 config / decision / evaluator 代码的修改都会被 hash 校验拒绝。

## 你需要什么

- Python 3.11（`uv` 会帮你安装）
- CPU 即可；`gru`/`ode_rnn`/`latent_ode` 需要 `uv sync --extra dev --extra torch`，`device: auto` 有 CUDA 时自动用 GPU、没有就退回 CPU（测试全部在 CPU 上可跑）
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

动力学模型对比基线（里程碑 4，同一 split 上 6 个模型）：

```bash
uv sync --extra dev --extra torch
uv run omics-agent generate-synthetic --output-dir outputs/m4/data --design longitudinal
uv run omics-agent benchmark --experiment config/experiment.dynamics.example.yaml --unlock-test
```

在合成 ODE 数据上（V100，300 epochs，val 早停），test split 的结果示例：
`last_value` MSE 0.689 → `ridge` 0.304 → `mlp` 0.230 → `gru` 0.211 → `ode_rnn` 0.197 → `latent_ode` 0.178。
三个动力学模型的超参在 `params` 里（`epochs`、`hidden_dim`、`recon_weight`、`rk4_substeps`、`device` 等）；`latent_ode` 是确定性的 encoder-ODE-decoder（无 VAE 采样），由 seed 完全复现。

调参并冻结，然后花掉唯一一次 final test（里程碑 5）：

```bash
# 仅 validation 的 Optuna：固定预算/seed/study/pruner，写 OptimizationDecision + 冻结
uv run omics-agent tune --experiment outputs/m4/experiment.yaml --model gru --n-trials 12

# 一次性 final test：先校验全部冻结 hash，再烧锁，再评分。不加 --confirm 会拒绝。
uv run omics-agent unlock-test --experiment outputs/m4/experiment.yaml --model gru --confirm

# 之后再 tune / unlock-test 同一个 experiment_id 都会被拒绝：
#   "Start a new experiment_id (new hypothesis, new budget)."
```

写出 `reports/optimization_decision_<model>.json`（预算、seed、pruner、逐 trial 记录、best params、`objective_split: val`、`test_labels_visible: false`）、`frozen/<model>/`（checkpoint + frozen_experiment.yaml + freeze_manifest.json）和 `test_lock.json`。`optimization:` 块可在 experiment.yaml 里调 `n_trials` / `objective_metric`（mse|mae）/ pruner 参数；搜索空间是代码不是配置，optimizer 改不了 split、evaluator 和 primary metric。

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
| `models/<name>/` | LastValue / Ridge / time_spline / MLP / GRU / ODE-RNN / latent ODE |
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
| `needs PyTorch, which is not installed` | 没装 torch extra 就点了 `gru`/`ode_rnn`/`latent_ode` | `uv sync --extra dev --extra torch` |
| `needs longitudinal subject histories` | 对重复横断面数据用动力学模型 | 这类数据只能用 `last_value`/`ridge`/`mlp` + `group_time_forecast` |
| `ODE integration produced NaN/inf` / `loss became non-finite` | solver 或训练发散 | 调低 `lr`、增大 `rk4_substeps` 或减小 `hidden_dim`；发散的 run 不会出报告 |
| 条件在 val/test 出现但 train 没见过 | condition one-hot 编码未定义 | 检查 split：每个 condition 必须在 train 中出现 |
| `already ran its one-shot final test` | 该 `experiment_id` 的 test 锁已消耗 | 换新的 `experiment_id`；已冻结的结果就是最终结果 |
| `Frozen-artifact check failed` | 冻结后 split/checkpoint/config/decision/evaluator 代码被改过 | 恢复原文件或重新 tune 冻结；final test 不跑被改过的输入 |
| `unlock-test` 不带 `--confirm` 被拒 | final test 每个 experiment_id 只有一次 | 想清楚再加 `--confirm` |

## 验证

```bash
uv run pytest
uv run ruff check src tests
uv run mypy src/omics_agent
uv run omics-agent run-toy --output-dir outputs/toy
```

故意制造 subject 泄漏时，`assert_no_group_leakage` 必须失败（见 `tests/synthetic/test_split_guard.py`）。

## 下一步（还没做，也不会假装做了）

1. 先验消融与解释性（integrated gradients）、文献核验  
2. 生物先验图（里程碑 4/5 有意不加）  

解释值不是因果。文献以后若检索不到，只能写「在本次检索范围内未找到直接证据」。

## 设计选择

- 深度学习框架：**纯 PyTorch**（不用 Lightning；ODE 求解用自带 RK4，不引入 torchdiffeq）；基线用 sklearn / patsy
- 配置全部 YAML，业务对象全部 Pydantic
- `data/raw/` 只读；结果只写到 `outputs/`
