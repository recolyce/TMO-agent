# omics-agent

确定性的 **bulk 时序多组学预测流水线**。它不是会自己下载陌生代码、也不会猜测样本对应关系的 Agent。

里程碑 1 只做一件事：从合成双组学数据出发，按实验单位划分、训练三个基线（LastValue / Ridge / 时间样条），用独立 evaluator 写出带 hash 的 benchmark 报告。

## 你需要什么

- Python 3.11（`uv` 会帮你安装）
- CPU 即可，不需要 GPU，也不需要网络
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

也可以只校验清单：

```bash
uv run omics-agent validate-manifest config/dataset.example.yaml
```

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

## 验证

```bash
uv run pytest
uv run ruff check src tests
uv run mypy src/omics_agent
uv run omics-agent run-toy --output-dir outputs/toy
```

故意制造 subject 泄漏时，`assert_no_group_leakage` 必须失败（见 `tests/synthetic/test_split_guard.py`）。

## 下一步（还没做，也不会假装做了）

1. 真实 GEO / PRIDE / BioStudies 下载与 checksum  
2. 按 assay 的预处理策略  
3. GRU / latent ODE（纯 PyTorch，不用 Lightning）  
4. 仅 validation 的 Optuna，以及一次性 unlock-test  
5. 先验消融与解释性、文献核验  

解释值不是因果。文献以后若检索不到，只能写「在本次检索范围内未找到直接证据」。

## 设计选择

- 深度学习框架：**纯 PyTorch**（里程碑 4 才引入；里程碑 1 基线用 sklearn / patsy）
- 配置全部 YAML，业务对象全部 Pydantic
- `data/raw/` 只读；结果只写到 `outputs/`
