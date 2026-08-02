# 当前方法与方法边界

## 活动系统

当前活动协议是 `protocol_v2_textoir_v1`，系统保持
`Gate → Router → Expert` 结构。本轮只整理 Gate 研究与实验入口，不改变历史
detector 的行为。

Frozen MiniLM Gate 的已冻结定义是：对 `all-MiniLM-L6-v2` 句向量做 L2 归一化，
在每个 Known intent 内拟合固定数量 `K` 的局部中心，使用欧氏或对角马氏距离，
半径为 `μ + λσ`（当前默认 `λ=1`），并以历史 `nearest_sphere` 语义计算
`s(x)=min_{y,k} d(x,c_{y,k})/r_{y,k}`。超出全部接受球的样本判为 OOS。

`K=1..5`、11 个 KIR、5 个正式 seed 的 E2，以及 KMeans/random-balanced 和
Known-only 机制诊断 E3 均已收口，禁止作为同一协议重复运行。

## 已知边界

固定多中心不是跨数据集默认配置：Banking77 只有条件性收益，CLINC150 收益弱且
不稳定，StackOverflow 的固定后处理多中心存在明显的接受区域并集风险。MOGB 是
外部基线，不是 s2c 的原创模块；其原始代码、适配层和 negative reproduction 证据
均在 `docs/archive/mogb_reproduction/` 与对应 artifacts 中隔离保存。

## split–merge adaptive-K 原型

当前只实现安全门和 dry-run，不启动正式矩阵。每个意图从 `K=1` 开始，候选二分
必须同时满足：

1. compactness gain `> tau_compact`；
2. 两个子簇的最小样本数 `>= n_min`；
3. bootstrap ARI 稳定性 `>= tau_stability`；
4. Known-only calibration 上跨意图接受率增加 `<= epsilon`；
5. 扣除复杂度惩罚后的 gain `> 0`。

极小簇先合并到最近的可用簇；缺少跨意图接受率证据时默认拒绝。实现位于
`src/protocol_v2/experiments/adaptive_split_merge.py`，入口
`scripts/experiments/run_adaptive_split_merge.py --dry-run`。它是候选机制骨架，
不代表已有 adaptive-K 结果，也不改写 BRAK 历史结果。

## 代码布局

活动代码只允许放在 `src/protocol_v2/`、`scripts/`、`tools/` 和 `tests/`；禁止
创建 `src/s2c/`。大模型、数据和原始 artifacts 分别位于 `../assets/`、`data/`
和 `../artifacts/s2c/`，后两者不进入公开结果提交。
