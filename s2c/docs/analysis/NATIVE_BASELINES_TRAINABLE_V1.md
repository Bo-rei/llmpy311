# Trainable MiniLM 表示上的原生 OOS 检测器对照 V1

> 目的：在同一批已完成 Trainable MiniLM 表示上运行 MSP、Energy、kNN、LOF，区分表示本身的收益与当前多中心/单中心 Gate 公式的收益。该阶段不训练、不使用测试 OOS 选参数，也不覆盖 Frozen native baseline。

## 完成状态

- 组合：9 个 dataset×KIR×seed；检测器单元：36；失败：0。
- 范围：3 数据集、KIR=.50、seed=13/42/87；Trainable checkpoint 按组合复用。
- 阈值：每个 native detector 只用 Known calibration 的 conformal alpha=.05 选择；测试 OOS 只做最终评价。

## 解释边界

Trainable native 与 Trainable Gate 的比较回答“同一表示换检测器后是否仍有优势”；Trainable native 与 Frozen native 的比较回答“同一检测器换表示后是否改善”。这两组结果不能与 ADB、DA-ADB、MOGB 官方或 DCLOOS 兼容性单格混成 SOTA 排名。

## 输出

- `results/analysis/native_baselines_trainable_v1/trainable_native_per_seed.csv`：逐 seed 结果。
- `results/analysis/native_baselines_trainable_v1/trainable_native_vs_gate_paired.csv`：配对差值与 bootstrap CI。
- `results/analysis/native_baselines_trainable_v1/trainable_vs_frozen_native_paired.csv`：表示替换差值。
- `figures/native_baselines_trainable_v1/`：KIR 曲线与 KIR=.50 权衡图。

## 初步用途

若 Trainable 表示在 MSP/Energy/kNN/LOF 上也改善，优势主要来自表示；若只有 Trainable Gate 明显改善，优势主要来自 Gate 的 Known-only 几何与校准合同。正式结论以 CSV 配对结果为准。
