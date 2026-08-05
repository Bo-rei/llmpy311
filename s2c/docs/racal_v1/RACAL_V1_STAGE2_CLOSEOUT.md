# RACAL-v1 阶段二收口

- 阶段：`racal_v1_stage2_fixed_k2`
- 范围：StackOverflow、KIR=0.50、seed 13/42/87、Trainable MiniLM 同 checkpoint、K=1/K=2
- 完成：3/3 seed，失败/缺失/重复/无效为 0
- K=1：重新编码后与阶段一指标最大差异 0
- K=2：每个 Known intent 内 KMeans-2，未重新训练表示
- 平均 OOS F1：K=1 `0.8671±0.0079`，K=2 `0.6765±0.0615`
- 平均 false acceptance：K=1 `0.1114±0.0165`，K=2 `0.4526±0.0782`
- 平均 Known Recall：K=1 `0.8392±0.0032`，K=2 `0.9362±0.0029`
- 判断：A 主导、伴随 C；K=2 的 Known 覆盖收益不足以抵消 OOS 过接受
- 决策：停止 K=3--5；不自动运行风险门控阶段；RACAL 仍可作为后续 intent-level 风险激活方向登记
- 未运行：proxy-OOS、risk gate、adaptive K、parent guard、energy/gap、其他数据集/KIR、完整 Cascade

详细结果见 `RACAL_V1_STAGE2_REPORT.md`。完整运行产物和哈希样本审计保存在阶段二 artifact 根中。
