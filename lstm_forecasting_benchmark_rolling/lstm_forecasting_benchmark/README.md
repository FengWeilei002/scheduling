# LSTM Forecasting Benchmark (中文说明)

**概述** 🔧

本项目生成合成的日度多变量时间序列（Yd, TSE, TSEC），并比较不同预测方法的性能（MEAN、ARIMA、RNN、LSTM），同时提供滚动源（rolling origins）评估与基于 step_1 的敏感性成本分析。

---

## ✅ 快速开始

1. 克隆/进入项目目录后，安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行 **全部 28 个案例**（M = 4..10，yd_level = HUR/LUR，solar_level = HUR/LUR）：

```bash
python run_rolling_all_cases.py
```

3. 或者运行单个案例 / 合成数据：

- 合成数据（默认）：

```bash
python run_rolling_auto.py
```

- 指定单案例（例如 M=5, yd=HUR, solar=LUR）：

```bash
python run_rolling_auto.py --M 5 --yd HUR --solar LUR --outdir outputs/cases/M5_YHUR_SLUR --seed 123
```

---

## 🔍 工作流程与文件说明

- `run_rolling_all_cases.py`：遍历所有组合案例（M ∈ [4,10], yd ∈ {HUR,LUR}, solar ∈ {HUR,LUR}），对每个案例调用 `run_one`（来自 `run_rolling_auto.py`），并将每个案例的指标合并为 `outputs/metrics_all_cases.csv`。
  - 本脚本使用一个确定性的 case seed（见下）以避免不同案例之间重用相同随机抽样。
  - 运行完成后会生成一个可选的 step_1 WAPE pivot 表 `outputs/metrics_all_cases_step1_wape_pivot.csv`，方便跨模型比较。

- `run_rolling_auto.py`：核心执行逻辑，包含数据生成、训练（RNN/LSTM）、ARIMA/MEAN 基线、滚动预测、评估指标与敏感性成本计算。
  - 输出到每个 `outputs/cases/M{M}_Y{yd}_S{sol}/` 目录下的文件：
    - `data.csv`：本案例用于训练与测试的时间序列
    - `metrics_summary_rolling.csv`：按模型与 step 汇总的评估指标
    - `rolling_predictions_tidy.csv`：整洁格式的逐 origin/step 预测与真值
    - `rolling_step1_table.csv`：仅 step_1 的观测 & 各模型预测
    - `sensitivity_costs_rolling_step1.csv`：基于 newsvendor 成本的敏感性分析结果

- 配置文件：`src/config.py`，包含重要参数（随机种子、历史天数、滚动参数、训练轮次等）。

---


## 输出文件（汇总）

- `outputs/metrics_all_cases.csv`：包含所有案例、模型与步长（step）的评估指标（如 WAPE, MASE 等）。
- `outputs/metrics_all_cases_step1_wape_pivot.csv`：针对 `bucket == 'step_1'` 的 WAPE 做透视，便于横向对比模型在 step_1 的表现。
- 每个案例目录下的文件详见上文“工作流程与文件说明”。

---




**快速参考**：

```bash
# 运行全部案例
python run_rolling_all_cases.py

# 运行单案例示例
python run_rolling_auto.py --M 6 --yd LUR --solar HUR --seed 2024
```

