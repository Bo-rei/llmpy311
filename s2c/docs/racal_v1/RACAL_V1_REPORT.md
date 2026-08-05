# RACAL-v1 第一阶段结果报告

## 1. 范围

本阶段只运行 `protocol_v2_textoir_v1` 下的 StackOverflow、KIR=0.50、seed `{13,42,87}`。
比较 Frozen MiniLM + E2 K=1 与 Trainable MiniLM + K=1。没有实现或运行 fixed K=2、proxy-OOS、
自适应中心激活、parent guard、其他数据集或完整 Pipeline。

## 2. E2 K=1 精确复现

RACAL runner 通过 `load_e2_bundle()` 读取冻结 E2 cache，并使用与 E2 相同的
`class_centroid_mixture + diagonal Mahalanobis + mean+1.0*std + nearest_sphere`。

|seed|sample-id mismatch|prediction mismatch|score max abs delta|metric max abs delta|结果|
|-|-|-|-|-|-|
|13|0|0|0|0|通过|
|42|0|0|0|0|通过|
|87|0|0|0|0|通过|

因此 Frozen K=1 是同一评价契约下的有效基线；E2 原始 artifacts 没有被修改。

## 3. Trainable MiniLM + K=1

训练结构为：本地 `all-MiniLM-L6-v2` → mean pooling → 384 维两层残差 projection → L2 normalization。
第一阶段只训练 projection head 1 个 epoch，第二阶段解冻最后两个 Transformer block 并以较小学习率训练。
损失只有 Known intent CE、类内中心紧致和最近异类中心 margin。checkpoint 只由 Known calibration
选择，`test_used_for_selection=false`。

|方法|seed|OOS F1|F1-All|F1-K|Accuracy|Known Recall|False Acceptance|AUROC|AUPR-OOS|
|-|-|-|-|-|-|-|-|-|
|Frozen K=1|13|0.7185|0.7752|0.7809|0.7128|0.8353|0.3470|0.8612|0.8157|
|Frozen K=1|42|0.8197|0.7766|0.7723|0.7715|0.8380|0.1930|0.9065|0.8927|
|Frozen K=1|87|0.7805|0.8061|0.8087|0.7747|0.8380|0.2563|0.8780|0.8321|
|Trainable K=1|13|0.8566|0.8524|0.8520|0.8497|0.8427|0.1330|0.9059|0.8625|
|Trainable K=1|42|0.8755|0.8577|0.8559|0.8648|0.8350|0.0930|0.9268|0.9120|
|Trainable K=1|87|0.8692|0.8596|0.8586|0.8602|0.8400|0.1083|0.9199|0.8889|

三 seed 均值：

|方法|OOS F1|F1-All|F1-K|Accuracy|Known Recall|False Acceptance|AUROC|AUPR-OOS|
|-|-|-|-|-|-|-|-|-|
|Frozen K=1|0.7729 ± 0.0417|0.7860 ± 0.0143|0.7873 ± 0.0156|0.7530 ± 0.0284|0.8371 ± 0.0013|0.2654 ± 0.0632|0.8819 ± 0.0187|0.8468 ± 0.0331|
|Trainable K=1|0.8671 ± 0.0079|0.8565 ± 0.0030|0.8555 ± 0.0027|0.8582 ± 0.0063|0.8392 ± 0.0032|0.1114 ± 0.0165|0.9175 ± 0.0087|0.8878 ± 0.0202|

Trainable K=1 相对 Frozen K=1 的均值变化：

- OOS F1：`+9.42pp`；
- F1-All：`+7.06pp`；
- F1-K：`+6.82pp`；
- Accuracy：`+10.52pp`；
- Known Recall：`+0.21pp`；
- false acceptance：`-15.40pp`；
- AUROC：`+3.56pp`；
- AUPR-OOS：`+4.09pp`。

## 4. 阶段判断

Trainable K=1 满足本阶段晋级条件：三个 seed 的 OOS F1 均高于 Frozen，平均 F1-All 提升，Known
Recall 没有下降，false acceptance 显著降低，且没有方向相反的 seed。

这只证明“多中心感知表示学习的 K=1 表示控制”有效，不证明多中心有效，也不证明 RACAL 完整方法
已经成立。

## 5. 下一步授权边界

允许登记但不自动执行下一阶段：

1. Trainable representation + fixed K=2；
2. proxy-OOS 训练控制；
3. 风险约束中心激活；
4. parent guard 消融。

下一阶段仍必须保持 StackOverflow/KIR=0.50，并先完成 fixed K=2 对照后再激活中心。不得直接扩展
其他数据集、KIR、K=3--5 或完整 Pipeline。

## 6. 证据入口

- 完整本地运行：`../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1/`；
- 轻量逐 seed 结果：`results/diagnostics/racal_v1/RACAL_V1_STAGE1.csv`；
- 均值标准差：`results/diagnostics/racal_v1/RACAL_V1_STAGE1_MEAN_STD.csv`；
- 验证结果：`../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1/RACAL_VERIFY.json`；
- 代码/配置/模型和数据 hash：各 run manifest 与 `RACAL_PROVENANCE.json`。
