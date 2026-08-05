# RACAL-v1 阶段二：Trainable MiniLM 下固定 K=1/K=2 对照

## 1. 研究问题与范围

本阶段只回答一个问题：在阶段一已经验证有效的 Trainable MiniLM 表示上，把每个 Known intent
从一个中心固定改成两个中心，是否仍会造成 StackOverflow OOS false acceptance 上升。

固定条件：`protocol_v2_textoir_v1`、StackOverflow、KIR=0.50、seed 13/42/87、同一阶段一
Trainable MiniLM checkpoint、对角 Mahalanobis、`mean + 1.0 * std` 半径、阈值 1.0、历史
E2 `nearest_sphere` 判定。K=2 仅在每个 intent 内执行 KMeans-2，未重新训练或选择表示。

没有使用 proxy-OOS、risk gate、adaptive K、parent guard、energy/gap、阈值调参、K=3--5、
其他数据集、其他 KIR 或完整 Cascade。

## 2. 数据和复现契约

每个 seed 的 train/calibration/test 数量为 6000/1000/6000，test 中 Known/OOS 各 3000；三组
split 的 sample ID 交集均为 0。阶段一 checkpoint 的 SHA256、registry SHA256、canonical
manifest SHA256 和 sample-id 顺序均逐 seed 校验。

阶段二重新编码 train、calibration、test，并同时拟合 K=1 与 K=2。重新编码后的 K=1 指标与
阶段一 checkpoint 结果最大绝对差为 0，说明本阶段没有发生表示或评价契约漂移。

## 3. 三 seed 结果

|方法|OOS F1|OOS Precision|OOS Recall|F1-All|F1-K|Accuracy|Known Recall|False Acceptance|False Rejection|AUROC|AUPR-OOS|FPR95|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Trainable K=1|0.8671 ± 0.0079|0.8468 ± 0.0008|0.8886 ± 0.0165|0.8565 ± 0.0030|0.8555 ± 0.0027|0.8582 ± 0.0063|0.8392 ± 0.0032|0.1114 ± 0.0165|0.1608 ± 0.0032|0.9175 ± 0.0087|0.8878 ± 0.0202|0.2118 ± 0.0090|
|Trainable fixed K=2|0.6765 ± 0.0615|0.8944 ± 0.0102|0.5474 ± 0.0782|0.7681 ± 0.0126|0.7772 ± 0.0080|0.7156 ± 0.0347|0.9362 ± 0.0029|0.4526 ± 0.0782|0.0638 ± 0.0029|0.8945 ± 0.0136|0.8754 ± 0.0247|0.4386 ± 0.0324|
|K=2 − K=1|-0.1906|-|-0.3411|-0.0885|-0.0783|-0.1426|+0.0970|+0.3411|-0.0970|-0.0230|-0.0124|+0.2268|

K=2 提高 Known Recall，但代价是大量 OOS 被误接受；因此它不是一个可接受的安全改进。

## 4. 逐 seed false acceptance 审计

|seed|K=1 OOS F1|K=2 OOS F1|K=2 − K=1|K=1 false acceptance|K=2 false acceptance|新增 OOS false acceptance|恢复 Known false rejection|净收益|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|13|0.8566|0.6217|-0.2349|0.1330|0.5220|1169|298|-871|
|42|0.8755|0.7624|-0.1131|0.0930|0.3433|753|309|-444|
|87|0.8692|0.6453|-0.2239|0.1083|0.4923|1154|285|-869|

三 个 seed 均出现相同方向：K=2 新增接受的 OOS 远多于恢复的 Known 样本。

## 5. intent-level 诊断

每个 seed 的 10 个 intent 均输出 K=1 半径、K=2 两个子簇的样本数/半径/方差、bootstrap ARI、
silhouette、Known Recall 变化和净收益。bootstrap ARI 与 silhouette 仅作解释，不参与 K 选择。

跨三个 seed 共 30 行 intent 诊断中，净收益为正 16 行、为负 14 行，说明存在异质性；但总体
风险由过覆盖主导。`osx`、`drupal`、`scala`、`sharepoint`、`cocoa` 等 intent 在三个 seed
均表现为新增 OOS 误接受大于恢复 Known。相反，`excel`、`linq`、`oracle` 等 intent 在部分
或全部 seed 上有正净收益。该异质性支持后续只允许对 calibration 风险下降的 intent 激活 K=2，
而不是对所有 intent 固定使用 K=2。

## 6. 结果判定

本阶段属于 **A：fixed K=2 明显退化**，同时具有 **C：intent-level 高度异质** 特征：

- 平均 OOS F1 下降 19.06 个百分点，远超 1 个百分点门槛；
- 平均 false acceptance 增加 34.11 个百分点，远超 3 个百分点门槛；
- 三个 seed 方向一致；
- K=2 改善 Known Recall，但破坏 OOS 拒识。

因此停止 K=3--5，不运行其他数据集或 KIR，也不把 fixed K=2 写成 RACAL 成功结果。

## 7. 下一阶段边界

本阶段结束后不自动运行下一阶段。若继续 RACAL，只允许新登记一次最小风险门控实验：在同一
StackOverflow/KIR=0.50/三个 seed 下，仅对 Known calibration 风险下降且稳定的 intent 激活
K=2，其余 intent 保持 K=1。该实验必须重新冻结配置和 provenance，且不能使用本阶段 test
结果选择激活规则。

## 8. 证据入口

- 本阶段报告：`docs/racal_v1/RACAL_V1_STAGE2_REPORT.md`
- 轻量逐 seed 汇总：`results/diagnostics/racal_v1/stage2_fixed_k2/RACAL_V1_STAGE2_PER_SEED.csv`
- 均值与差值：`results/diagnostics/racal_v1/stage2_fixed_k2/RACAL_V1_STAGE2_MEAN_STD.csv`
- intent 诊断：`results/diagnostics/racal_v1/stage2_fixed_k2/RACAL_V1_STAGE2_INTENT_DIAGNOSTICS.csv`
- 哈希样本审计：`results/diagnostics/racal_v1/stage2_fixed_k2/RACAL_V1_STAGE2_SAMPLE_AUDIT_HASHES.csv`
- 本地完整 artifact：`../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1/stage2_fixed_k2/`
- 验证 manifest：`../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1/stage2_fixed_k2/RACAL_STAGE2_VERIFY.json`
