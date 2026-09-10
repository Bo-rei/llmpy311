# 可复现性契约

## 数据与版本

从 2026-08-26 起，**新实验默认使用 `historical_v19_paper_main`**。当前可执行的是已
核验的 H1 controlled v19 快照，保持旧数据家族和旧分层；它是后续同协议对照的唯一稳定
旧输入，不是严格 H0 主 Cascade 的逐字输入。论文 H0 的不同数据集/KIR 锚点本身存在路径
和语义 Gate 漂移，必须逐行恢复。严格 H0 缺少必要文件时必须停止，不能自动用 H1 代替。已有
`protocol_v2_textoir_v1` 结果冻结为参考，不再作为新实验默认协议。

当前可执行的历史 H1 数据根是
`../assets/datasets/s2c/prepared/data/multidataset/v19`，三项实际数据键为：

- `clinc150`：CLINC150 `data_full.json`，保留 held-out intent 与原生 `oos_val/oos_test`；
- `stackoverflow`：20 个标签、title-only、12k/2k/6k split，Known 列表由旧 v19 的
  `seeded_random` 规则生成；
- `banking77_oos`：50 个 in-domain intent 加 `id-oos/ood-oos`，这是论文实际锚点使用的
  数据键，虽然 `fulltex.tex` 表头写成了 `Banking77`。

旧协议的 Gate train、Router train/val/test 和 Expert train/val/test 都只使用 Known；Gate
val/test 才加入 held-out/native OOS。论文主 Cascade 对齐使用 `seed=42`、KIR
`.25/.50/.75`、每意图 `K_y=2`、对角 Mahalanobis、`mean+lambda*std`（CLINC
`lambda=.5`，其余 `lambda=1`）以及历史 Router/Expert/Cascade 设置。Frozen/Trainable
的 K=1 对照仍使用同一旧数据根，但属于 Gate-only 控制，不能回填论文主 Cascade。

旧快照由 `scripts/data/active/rebuild_multi_dataset_v19.py` 生成；CLINC 的旧单数据重构
逻辑还保留在 `scripts/data/active/rebuild_v19_2_strict.py`，归档中的旧实现和数据位于
`../archives/submissions/s2c-submission/`。论文锚点保存在
`../artifacts/s2c/outputs/paper_results/`，已据此挂载 `data/v19` 的锚点重建版本（75 个
Known、Gate test 5499 条）；该目录明确标记为非字节级 H0，不能把它或 H1 快照冒充为
原始 H0 逐字重跑。

当前主分析将 OOS 作为主任务：OOS F1、OOS precision/recall、AUROC/AUPR、false acceptance
和同样本 OOS 状态转移进入主视觉；Known Recall/false reject 只作为 coverage guard，避免把
全拒绝误读成 OOS 方法优势。

结果命名也属于复现契约：`Frozen E2 K=1` 专指当前 E2 的冻结 MiniLM、对角 Mahalanobis、
`mean+std` Gate；`Frozen single-centroid（MOGB-Fair；Euclidean）` 专指
`cross_protocol_tradeoff_v1` 的 `single_centroid` 组件行。两者不能用同一个简写或数字互换。

每次新 run 只需明确记录数据根、数据集、KIR、seed、Known 列表、representation、K、
distance、boundary 和评估器；只有大文件身份差异会改变下一步判断时，才补充 hash。不要
为了形式完整新增 manifest、注册表或校验文件。

复现活动协议时必须显式传入正式 seeds `13/42/87/100/123`。当前
`protocol_v2.data.validate_protocol` 的无参数默认值仍来自旧的 compatibility 约定，
会尝试检查不存在的 `seed_0..9` registries；这是校验器默认值的待修复缺口，不代表活动
manifest 或已落盘视图失效。以 `seed=42, KIR=0.50` 对三数据集显式校验并要求 views/exports
均已通过。

## TEXTOIR 其他方法的输入合同

这里要区分三种复现强度：

1. **源数据严格复现**：直接使用干净、固定 commit 的 `../textoir/data/`，其中上游目录名为
   `oos/`、`banking/` 和 `stackoverflow/`，让 TEXTOIR 原生 `DataManager` 按其
   `benchmark_labels`、`RandomState(seed)` 和 `known_cls_ratio` 选择 Known 类。
   `data/sources/textoir/dffe2b1b848a069a6808f8089b4cb9bd16e2062b/` 是已完成字节级核对的
   本地 source mirror；它的标准化目录名不能未经显式映射直接传给上游 runner。
2. **冻结的 protocol_v2 fair compatibility**：使用 `data/exports/protocol_v2_textoir_v1/{textoir,adb,da_adb}/`
   下的 seed-specific Known-only train/dev 和 combined test，配合同一 registry 与
   `known_labels.json`。它适合把 S2C 与 BERT/TextOIR 方法放到同一当前样本合同上比较，
   但不是上游原始 TSV 的 byte-identical 运行：当前 `test_combined` 的样本集合相同，顺序
   是 Known→OOS 分组，而不是源 test 原顺序；adapter 统一把 OOS 写成字面量 `oos`，
   而上游非 CLINC 数据的内部 unknown token 是 `<UNK>`，运行时会再映射到同一个 unknown id。
3. **论文数值严格复现**：还需要上游方法的原始代码、依赖、checkpoint/训练参数、GPU
   行为和论文当时的 seed/评估实现；仅有当前数据快照不能推出论文表数值已经复现。

当前已审计 TEXTOIR 注册的方法包括 MSP、DOC、ADB、OpenMax、KNNCL 和 DA-ADB。已有
MSP/DOC/ADB native compatibility 已完成 3 数据集×3 KIR×3 seed；DA-ADB 已完成
StackOverflow/KIR=.50 的 3 seed。它们仍应写成“冻结 protocol_v2 下的 BERT/TextOIR compatibility”，
不能写成完整官方论文复现。以后若要把这些方法与历史 S2C 主表放在同一实验中，必须将
旧 `multidataset/v19` 的 Known 列表和 Gate test 通过显式适配转换为上游输入，不能直接沿用
当前 protocol_v2 的 Known registry。KNNCL 的上游 label-alias typo 已在隔离 overlay 中修复，但
当前尚无有效 final metrics；OpenMax 也未形成完整主表。若要复现上游方法本身，直接使用
已审计、干净且固定 commit 的 `../textoir/data`（上游目录名为 `oos/banking/stackoverflow`）；
不要把 canonical `data/sources/.../clinc150/banking77/` 目录直接传给上游 runner，除非先
显式完成目录名映射并记录 provenance。若要做当前 fair 对比，才使用 protocol export 和固定
`known_labels.json`。

若只做一个最小 smoke，优先选 StackOverflow/KIR=.50：它同时覆盖旧论文差异最大的本地
快照、当前 6,000 条 test 和 ADB/DA-ADB/KNNCL 的兼容边界；正式结论仍需回到三数据集，
不能用单一 smoke 代替全矩阵。`run_textoir_matrix.py` 的默认 seeds `0/1/2` 与 KIR
`.25/.50/.75` 是 TextOIR compatibility matrix；S2C fair 主矩阵仍使用
`13/42/87/100/123`，两者不直接合并。

## 已冻结实验

E0/E1/E2/E3 以及历史 R1、MOGB、ADB/DA-ADB/DCLOOS 单元均有独立 artifact root；
`docs/EXPERIMENT_LEDGER.csv` 的 `repeat_policy=do_not_repeat` 行不得被新计划覆盖。
不要删除或重建 `../artifacts` 中的任何历史结果。

## 只读审计命令

```bash
cd /home/bo/bo01/llmpy311/s2c
python tools/maintenance/audit_asset_catalog.py
python tools/analysis/audit_adaptive_k.py
python tools/analysis/diagnose_mogb_diff.py
python scripts/experiments/run_adaptive_split_merge.py --dry-run
python tools/analysis/audit_experiment_registry.py
python tools/maintenance/check_data_tracking.py
```

`audit_asset_catalog.py` 只检查 registry 中的分析 bundle 与结果、图、报告、manifest 的关系，
未登记目录只作为待归档 warning，不会自动移动或删除。上述其余命令不训练模型；adaptive-K 和
MOGB 输出分别写入
`results/diagnostics/adaptive_k/`、`results/diagnostics/mogb_diff/`。

## 代码与环境

活动包只从 `src/protocol_v2/` 导入，禁止创建 `src/s2c/`。第三方 MOGB checkout
`third_party/mogb_official` 保持未修改；兼容层和来源说明必须分离。环境至少记录
Python、NumPy、SciPy、scikit-learn、PyTorch、Transformers 版本以及 GPU/CPU 设备。

## 结果隔离

轻量 CSV/JSON 放在 `results/`；模型、embedding、checkpoint、逐样本 predictions、
完整语料和日志只放本地 artifact。Gate-only 与完整 Cascade、官方 MOGB 与 MiniLM
fair adapter、真实 OOS 与 Known-only 结果必须分栏，不能仅因文件名相似而合并。
