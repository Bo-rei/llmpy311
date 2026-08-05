# 训练参与式自适应多中心实验报告

## 1. 实验目的

本阶段回答一个此前没有被真正验证的问题：**如果局部中心参与 MiniLM 表示训练，并且每个意图
是否增加中心由 Known-only calibration 决定，能否在 StackOverflow 上避免固定 K=2 的 OOS
false acceptance 爆炸？**

本阶段不是 E2/E3 的重复，也不是把后处理 KMeans 改名为 adaptive-K。实验独立使用
`joint_adaptive_multicenter_v1`，不修改 E2、E3、R1、RC-AMBL、RACAL 或 MOGB 的历史 artifact。

## 2. 方法实现

代码入口：

- `src/protocol_v2/experiments/joint_adaptive_v1/runner.py`
- `scripts/experiments/run_joint_adaptive_multicenter_v1.py`
- `scripts/experiments/analyze_joint_adaptive_v1.py`
- `configs/experiments/protocol_v2_textoir_v1/joint_adaptive_multicenter_v1.yaml`

每个 Known intent 从一个父中心开始。候选 intent 由 Known train embedding 的残差 P90 和样本数
筛选；候选中心使用 PCA 第一主方向的确定性 median split 产生。候选产生后，MiniLM 最后两层、
residual projection 和 intent prototypes 一起训练；损失包含最近 prototype 的 intent CE、类内
紧致项和类间 margin。候选 split 只有在 Known calibration 上同时满足以下条件才被接受：

1. Known Recall 下降不超过 1 个百分点；
2. calibration compactness gain 达到预设门槛；
3. objective regression 不超过上限；
4. 子簇样本量合格；
5. 子中心受到当前父级边界保护，不能扩大父级安全区域。

测试 OOS 只用于最终指标，未用于训练、候选选择、中心数、半径、阈值或 checkpoint 选择。

## 3. 实验范围与 provenance

| 项目 | 设置 |
|---|---|
| protocol | `protocol_v2_textoir_v1` |
| dataset | StackOverflow |
| KIR | 0.50 |
| seeds | 13, 42, 87 |
| 初始模型 | RACAL Trainable K=1 checkpoint |
| 可训练部分 | MiniLM 最后两层 + residual projection + intent prototypes |
| 候选训练 | 2 epochs；候选学习率 `5e-6` |
| 最大候选 split | 1 |
| 最大接受 split | 1 |
| 结构选择 | Known train residual + Known calibration |
| OOS 训练/选择 | `false / false` |

正式 artifact：

`../artifacts/s2c/runs/protocol_v2_textoir_v1/joint_adaptive_multicenter_v1/repair6/`

其中 `JOINT_ADAPTIVE_PROVENANCE.json`、`JOINT_ADAPTIVE_CODE.patch`、source manifest、每 seed
checkpoint、`split_events.json`、`metrics.json` 和 `predictions.csv` 均已生成。

## 4. 结果

### 4.1 主结果

三 seed 均值 ± seed 间总体标准差：

| 方法 | OOS F1 | F1-All | F1-K | Accuracy | Known Recall | False Acceptance | Mean K_y |
|---|---:|---:|---:|---:|---:|---:|---:|
| RACAL Trainable K=1 | 0.8671 ± 0.0079 | 0.8565 ± 0.0030 | 0.8555 ± 0.0027 | 0.8582 ± 0.0063 | 0.8392 ± 0.0032 | 0.1114 ± 0.0165 | 1.0 |
| joint adaptive（repair6） | 0.8661 ± 0.0111 | 0.8563 ± 0.0050 | 0.8553 ± 0.0047 | 0.8577 ± 0.0089 | 0.8388 ± 0.0045 | 0.1129 ± 0.0228 | 1.0 |

相对 RACAL Trainable K=1，joint adaptive 的均值变化为：

- OOS F1：`-0.10pp`；
- F1-All：`-0.03pp`；
- Known Recall：`-0.04pp`；
- false acceptance：`+0.14pp`。

因此本阶段没有得到新的多中心增益，也没有制造一个虚假的多中心正结果。

### 4.2 候选 split 事件

| seed | 候选 intent | calibration score gain | Known Recall delta | 结果 | 原因 |
|---:|---|---:|---:|---|---|
| 13 | osx | +0.00056 | -0.001 | reject | `no_known_only_compactness_gain` |
| 42 | cocoa | -0.00900 | -0.009 | reject | `no_known_only_compactness_gain` |
| 87 | osx | -0.02172 | 0.000 | reject | `no_known_only_compactness_gain` |

3/3 候选都实际生成了子中心并完成候选训练：三个候选 checkpoint 都包含 11 个 prototypes 和
2 个训练 epoch，训练 loss 分别从 `0.2771→0.2184`、`0.3435→0.2660`、`0.3176→0.2457` 下降。
但没有一个满足 Known-only 结构收益门槛；最终每个 seed 的 10 个 Known intent 均为 `K_y=1`。

### 4.3 union 过覆盖诊断

为确认拒绝机制不是“代码没有进入多中心路径”，对相同候选子中心做了事后固定 K=2 诊断：

| 诊断 | OOS F1 | False Acceptance | 解释 |
|---|---:|---:|---|
| joint K=1 | 0.8661 | 0.1129 | 安全父结构 |
| K=2 union（无父边界保护） | 0.6777 | 0.4513 | 新增球的并集扩大错误接受区域 |
| K=2 + parent guard | 0.8663 | 0.1118 | 只保留父边界内的诊断版本，非正式 adaptive 结果 |

这与 RACAL fixed K=2 的逐样本审计一致：多球 union 主要增加了 OOS false acceptance，而恢复的
Known 样本不足以抵消这个代价。

## 5. 结论

### 已经验证

1. 训练参与式实现已经存在：encoder、projection 和 prototypes 确实共同优化；候选训练不是
   伪代码，也不是固定 K 后处理的重命名。
2. Known-only calibration 可以安全地拒绝本轮候选 split，避免在 StackOverflow 上扩大错误接受区域。
3. 在当前强 K=1 表示基线下，StackOverflow 的候选局部划分没有提供足够 Known-only compactness
   gain。

### 尚未验证/不能声称

- 不能声称已经得到成功的 adaptive-K 方法；
- 不能声称训练参与式多中心已经超过 K=1、MOGB、ADB、DA-ADB 或 DCLOOS；
- 不能把 parent-guard 事后诊断当作正式自适应结果；
- 不能将本阶段 3 seed pilot 外推到其他数据集或 KIR。

## 6. 当前瓶颈

当前瓶颈不是“没有训练参与”，而是**候选 split 的训练目标没有在 Known-only calibration 上产生足够可验证的局部收益**；
一旦放开父边界保护，多个接受区域会出现明显 union 过覆盖。也就是说，表示训练解决了 K=1 的语义适配，
但没有证明某个 intent 的局部多峰结构值得增加第二个开放边界。

## 7. 下一步建议

本阶段完成后停止扩大实验。若继续研究，只能先登记一个新的候选 split 目标函数/父边界规则，在同一
StackOverflow、KIR=.50、3 seed 上做一次小 pilot；不得调 test OOS、不得直接扩 K=3--5、不得扩展
其他数据集。若新规则仍不能同时保持 OOS F1、F1-All、Known Recall 和 false acceptance，则应把
Trainable K=1 作为当前自有最佳结果，固定多中心作为结构性失败证据。

## 8. 验证命令与状态

```bash
env -u PYTHONSTARTUP JOINT_ADAPTIVE_ATTEMPT=repair6 PYTHONPATH=src \
  CONDA_NO_PLUGINS=true conda run -n bo \
  python scripts/experiments/run_joint_adaptive_multicenter_v1.py verify \
  --config configs/experiments/protocol_v2_textoir_v1/joint_adaptive_multicenter_v1.yaml

env -u PYTHONSTARTUP PYTHONPATH=src CONDA_NO_PLUGINS=true conda run -n bo \
  pytest -q tests/unit/test_joint_adaptive_v1.py
```

验证结果：3/3 seed 完成，失败数 0；专项单元测试 3/3 通过；`compileall` 通过；完整实验目录与
E2/E3/RACAL 目录隔离。
