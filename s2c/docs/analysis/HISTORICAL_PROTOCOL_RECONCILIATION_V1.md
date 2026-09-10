# 旧论文协议、当前 protocol_v2 与 Frozen/Trainable 结果对账

更新时间：2026-08-27  
状态：论文受控 Gate 表已复现；旧数据与处理代码已定位；H1 旧协议的 3×3×3 Frozen/Trainable 判因实验已完成；论文主表的严格 H0 Cascade 仍未逐字重跑。

阶段声明：本报告只服务于实验协议、结果和 OOS 机制分析；当前不修改 `fulltex.tex`，也不进入论文正文写作。

## 1. 先给结论

“当前 Frozen K=1 只有 78.64%，而论文明显更高”这个说法混合了三行不同的结果。

1. **78.64% 不是旧论文完整方法，也不是当前 canonical E2 Frozen K=1。** 它是当前 `protocol_v2` 公平矩阵里的 `single_centroid` 组件：冻结 MiniLM、Euclidean、单中心、`mean+std` 边界，跨 3 个数据集、3 个 KIR、5 个 seed 汇总。
2. **旧论文主表的 OOS F1 是完整 Cascade 的系统级数字。** 旧方法包含多中心 Gate、SmolLM Router/Expert，以及额外的语义 Gate/原型逻辑；不能拿 Gate-only 单中心结果直接和它比较。
3. **“旧协议”不能只写成 v19。** `fulltex.tex` 主表显示 `Banking77`，但现有历史 Ours artifact 的实际数据键是 `banking77_oos`；archive 的论文式 KIR=.25 `banking77` 线是 19-Known/58-OOS，属于另一条协议。后部受控 K 表又固定了 Gate-only、`lambda=1` 的控制合同。数据键、方法合同和 reported table 仍需分开记录。
4. **归档结果本身还存在评价字段不一致。** 例如 StackOverflow/KIR=.50 的 full-anchor 文件同时记录 `primary_metrics.oos_f1=0.5910`、顶层 `metrics.oos_f1=0.8971`，而同一文件的混淆矩阵/错误分解对应的是前者附近的结果；因此不能把 `0.8971` 当作已复现证据。
5. **同一旧协议的 Trainable K=1 已超过受控 Frozen K=1。** KIR=.50、seed=42 的 OOS F1 为 CLINC `89.56%`、StackOverflow `91.49%`、BANKING77-OOS `88.85%`，对应 Frozen `88.16%`、`84.36%`、`86.16%`。

所以，历史结果失去可复现性不是一次随机波动，而是“历史数据链混用 + 当前数据协议变化 + 系统层级变化 + 校准合同变化 + 历史结果字段不一致”共同造成的。

本报告从现在起把 `historical_v19_paper_main` 作为后续新实验默认：保留旧快照以对齐论文，
同时把原始数据中的跨 split 重复和论文文字与实际输入的矛盾明确标出；不再把
`protocol_v2_textoir_v1` 的新数据结果写回历史表。

## 2. 四条数据血缘先分开

| 代码 | 物理输入 | KIR=.50、seed=42 的 Gate test | 作用 | 当前状态 |
|---|---|---:|---|---|
| H0 strict Cascade | 论文主 Cascade 配置中的 `data/v19`；当前已挂载锚点重建数据 | 5499（2249 ID+3250 OOS） | 解释论文主表 `Ours` | 数据入口已按锚点重建并通过计数测试；原始字节快照、历史 Gate detector 和匹配模型仍未完整恢复 |
| H1 controlled table | `archives/submissions/s2c-submission/data` 与 `assets/datasets/s2c/prepared/data/multidataset/v19` | CLINC `2250+3250=5500`；SO `2993+2997=5990`；BANKING77-OOS `1000+3080=4080` | 复现论文后部固定 K 表 | Gate 表已由 `results/gate_only/kir_k_fixed_mean_std.csv` 闭合；CLINC/SO 归档与 prepared 内容一致 |
| H1 archive control | 归档 `data/{clinc150,stackoverflow,banking77}` | CLINC `2250+3250`；SO `2993+2997`；标准 Banking77 `1520+1560=3080` | 在旧样本上运行当前 Trainable/Frozen K=1 | 9 个 KIR×数据集 seed=42 已完成；标准 Banking77 与论文实际的 BANKING77-OOS 必须分开 |
| A active | `s2c/data/canonical|registries|views|exports/protocol_v2_textoir_v1` | CLINC `2250+2250+1200=5700`；SO `3000+3000=6000`；Banking77 `1520+1560=3080` | 当前正式 fair lane 与 TextOIR 输入 | source commit `dffe2b1b848a069a6808f8089b4cb9bd16e2062b`，正式 S2C seeds 为 `13,42,87,100,123` |

H0 保存的主表锚点还显示：CLINC 为 `2249 ID+3250 OOS=5499`，StackOverflow 为 `2993+2997=5990`，BANKING77-OOS 为 `1000+3080=4080`。它们的配置路径分别指向 `data/v19`、`data/multidataset/v19/stackoverflow` 和 `data/multidataset/v19/banking77_oos`；当前 checkout 有旧数据重建结果和部分下游 checkpoint，但没有形成与 H0 路径、Known 列表、语义/原型资产完全一致的可启动链。

因此，“归档数据能否复现论文”要具体化：H1 能复现受控 Gate 表；H1 的部分旧下游 Cascade 已有 36 个 KIR=.50 评估单元；H0 的完整 Cascade 目前只能核验保存下来的结果锚点，不能声称已逐字重跑。

### 2.2 H0 锚点能恢复到的边界

H0 的稳定结果锚点位于 `../artifacts/s2c/outputs/paper_results/`。其中
`anchors/clinc150_kir50_frozen/prototype_gate_pipeline_frozen/predictions.json` 保存了
5499 条 CLINC 测试预测：2249 条 Known、3250 条 OOS，并可反推出 75 个 Known intent。
将这 5499 条文本与当前 `assets/datasets/s2c/source/clinc150/data/data_full.json` 的
`test + oos_test` 做多重集合比较，锚点文本集合完全包含于当前源测试集合，只缺少一条
`test` 中 `user_name` 的样本 `what's your designation`；没有发现锚点独有文本。

这份证据已经用于重建并挂载 `s2c/data/v19`：包含 75 个 Known intent、Gate
`7500/1600/5499`、Router `7500/1500/2250` 以及各领域 Expert 数据；重建目录的
`KNOWN_INTENTS.json` 和 `MANIFEST.json` 明确标注为 `anchor_reconstructed`。它解决了
严格 replay 的数据入口缺失，但不能把它升级为原始 H0 的字节级恢复：对应的原始
`data/v19` 文件、历史 Gate detector、Router/Expert checkpoint、语义 verifier 和
selective-prototype payload 仍缺失。`configs/v19/clinc150_historical_best_reference.json`
记录的这些路径在当前 checkout 逐项检查均不存在；代码盘点文档中提到的若干
`tools/archive/code_cleanup_2026-03-28/...` 路径也不存在，因此不能把文档路径当成归档文件。

公开补充源也没有提供缺失链：论文的 arXiv v1 源包只包含 LaTeX、参考文献和论文插图，
没有代码、数据或模型文件；按论文标题和旧运行目录名检索 GitHub 也没有命中可用仓库。

`archives/research/legacy-research-20260714/clinc-lora/` 中的 `clinc_data_semantic` 和
semantic LoRA checkpoint 也已排除：它的 train/validation/test 为 `15250/3100/5500`，
任务是 13 个层级语义簇的生成式 prompt 分类，checkpoint 为另一套约 600M 参数模型的
LoRA（`r=8, alpha=16`）。H0 需要的是 75 个 Known intent 的 `data/v19`、10-domain
SmolLM-135M Router/Expert 和 multisphere Gate；两者只共享部分原始 CLINC 文本，不能互相替代。

还需要区分一个容易混淆的路径：在当前 checkout 的 `system_pipeline.py` 中，`gate_mode=multisphere`
会先加载 multisphere detector 并返回；`semantic_gate_mode=prototype` 的语义原型随后由
`gate_train` 和 Router/SmolLM 在运行时构建。配置中的 `multi_prototype_path` 只有在
`gate_mode=multi_prototype` 时才直接加载。因此 H0 当前真正不可替代的缺口是严格
`data/v19`、历史 Gate detector 以及与之匹配的 Router/Expert（和若使用 verifier 模式时的
verifier checkpoint）；不能把一个未使用的 prototype alias 文件单独当作全部缺口。

H0 锚点还证明“旧论文协议”在不同 KIR 行之间也不是一个完全固定的快照：CLINC 的
KIR=.50 锚点使用 `data/v19`，而 KIR=.25/.75 使用 `data/multidataset/v19/clinc150/...`；
StackOverflow 三个 KIR 使用 `data/multidataset/v19/stackoverflow/...`，BANKING77-OOS
三个 KIR 使用 `data/multidataset/v19/banking77_oos/...`。语义 Gate 也不是全表固定：
StackOverflow 的 KIR=.50 锚点为 `prototype`，KIR=.25/.75 为 `llm_verifier`。因此 H0
必须按“数据集×KIR×具体锚点”逐行恢复，不能声称存在一份覆盖论文全部主表行的单一 H0
协议。后续若需要同协议对照，采用 H1 controlled v19 作为唯一稳定旧协议；论文 H0 只作为
逐行 historical reference。

### 2.1 回答 3 所称的“同一旧协议”

回答 3 中 Frozen/Trainable 的三组数字使用的是同一份 **H1 controlled v19 数据快照**：

- CLINC150、StackOverflow 使用 `archives/submissions/s2c-submission/data`；其关键 JSON
  与 `assets/datasets/s2c/prepared/data/multidataset/v19` 逐字一致；BANKING77-OOS 使用
  后者由旧构建代码生成的 `banking77_oos` 快照。
- 固定 `KIR=.50`、`seed=42`，Known 列表不重新抽样；测试规模为 CLINC `2250+3250`、
  StackOverflow `2993+2997`、BANKING77-OOS `1000+3080`。
- Gate train 只用 Known；Gate val 保留旧快照中的 OOS，但 Frozen/Trainable K=1 的
  checkpoint 和边界不使用 OOS；最终只在 Gate test 计算 OOS-positive F1。
- Frozen 与 Trainable 唯一要比较的变化是表示；两者都用 K=1、对角 Mahalanobis、
  `mean+std`、`lambda=1`、score threshold `1`。

因此“同一旧协议”准确地说是**同一旧数据协议下的受控 Gate-only 对照**，不是“已经复现
了论文完整 Cascade”。

## 3. 证据来源

| 内容 | 当前证据 |
|---|---|
| 论文写明的数据集、6:1:3、数据来源、KIR、模型与超参 | `fulltex.tex` 的 Experiments/Datasets/Baselines 段落 |
| 论文主表与受控 K 消融 | `fulltex.tex` 的 `main_results_all` 与 `multicluster_k_ablation_v19` |
| 旧数据、Known intent、旧 Gate 训练脚本 | H1 归档 `../archives/submissions/s2c-submission/data` 与 `.../tools/gate/train_multisphere_corrected.py`；H0 配置 `configs/v19/clinc150_historical_best_reference.json` |
| 当前协议 registry/view/export | `data/registries/protocol_v2_textoir_v1`、`data/views/protocol_v2_textoir_v1`、`data/exports/protocol_v2_textoir_v1` |
| 当前 78.64% 组件行 | `results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv` |
| 归档历史 Cascade 结果 | `../artifacts/s2c/outputs/paper_results` |
| 本轮旧协议复跑 | `../artifacts/s2c/runs/historical_protocol_v1` |

## 4. 旧协议与当前协议的逐项差异

以下数字固定在 `KIR=.50`、`seed=42`；H1 表中的 BANKING77-OOS 与 H1 archive 的标准 Banking77 单独列出。当前侧来自 `data/views/protocol_v2_textoir_v1` 和 `data/exports/protocol_v2_textoir_v1`。当前 calibration 是 Known-only，因此不能把它直接当作旧 Gate 的混合验证集。

| 维度 | 旧论文/归档 v19 | 当前 `protocol_v2` | 直接后果 |
|---|---|---|---|
| 原始来源 | H1 来自归档 JSON；论文写明 CLINC `oos-eval`、StackOverflow title、PolyAI Banking77；H0 另有论文主 Cascade 的 `data/v19` 引用 | TEXTOIR commit `dffe2b1b848a069a6808f8089b4cb9bd16e2062b`；三套 train/dev/test TSV 字节级固定 | 源文件、行序和数据血缘不同 |
| Known intent | H1 CLINC/SO/Banking77 为 75/10/38；H1 controlled BANKING77-OOS 为 25；均由显式清单冻结 | CLINC/SO/Banking77 为 75/10/38，由 registry 的 MT19937-compatible RandomState 抽样 | KIR 相同不代表 Known 类相同；H0 锚点 Known 与 H1 archive overlap 为 CLINC 37/75；H1 archive 与 active 的其余 overlap 为 SO 6/10、Banking77 17/38 |
| KIR/seed | 主表标注 `.25/.50/.75`；H0 主锚点主要 seed=42；受控表为 `13,42,87` | S2C 正式 seeds `13,42,87,100,123`；KIR 视图为 `.25/.50/.75` 等 | 聚合单元与随机抽样改变 |
| Gate train | H1 archive：CLINC/SO/Banking77=`7500/5995/4407`；H1 controlled BANKING77-OOS=`2933` | CLINC/SO/Banking77=`7500/6000/4482` | 训练集合和数量变化 |
| Gate val/calibration | H1 archive：CLINC `1500 ID+1600 OOS`、SO `998+1000`、Banking77 `490+510`；H1 controlled BANKING77-OOS `749+1487` | Known-only：CLINC 1500、SO 1000、Banking77 495 | 旧协议允许用 OOS 做边界校准；当前正式 checkpoint/边界选择不看 OOS |
| Gate test | H1 archive：CLINC `2250+3250`、SO `2993+2997`、Banking77 `1520+1560`；H1 controlled BANKING77-OOS `1000+3080` | CLINC `2250+2250+1200`、SO `3000+3000`、Banking77 `1520+1560` | OOS 数量、正类先验和难度分布改变 |
| OOS 构造 | CLINC=held-out intent + native OOS；SO=held-out intent；Banking77=held-out intent；BANKING77-OOS 额外含 id-oos/ood-oos | CLINC=held-out + native；SO/Banking77=held-out，native OOS=0 | 不是同一 OOS 分布 |
| 文本处理 | H1 JSON 保存 `text/intent/domain/split/label`，StackOverflow 保存 `title/tag/source_id`；H0 原始快照未找回，但 CLINC 已有锚点重建数据 | canonical 增加 `sample_id`、source row/hash 和 duplicate audit；CLINC exact duplicate=5、SO=15、Banking=0 | 即使总量接近，也不能默认逐样本相同 |
| Gate 几何 | 论文主 Cascade `K_y=2`、对角 Mahalanobis、L2、均值+λ标准差；历史脚本可在混合 val 上选 λ/γ | E2 Frozen K=1 为对角 Mahalanobis、固定 λ=1；78.64% 组件为 Euclidean | K、距离、边界合同都不同 |
| 监督/选择 | 论文明确写过 OOS 用于验证阶段的 λ 学习；主系统还训练 Router/Expert | 当前 Trainable 只用 Known train 训练、Known calibration 选 checkpoint，不用 test OOS | 旧结果拥有更强的边界选择信息，不能与当前 clean control 混比 |
| 系统层级 | Gate → Router → Expert，主表是系统级结果 | 78.64%/E2/Frozen/Trainable 行是 Gate-only；当前 cascade bridge 另列 | 同名 OOS F1 不是同一预测对象 |
| 评价 | 论文主表 Known F1/OOS F1/Acc；历史 full-anchor 文件中另有多套 metrics 字段 | 当前 evaluator 明确 OOS 为正类，输出 OOS F1/AUROC/AUPR/false acceptance 等 | 必须绑定 evaluator 和字段，不能只看一个 JSON 数字 |

注意：`KIR=.50` 是请求比例，不一定是有效比例；CLINC/SO 为 75/150、10/20，BANKING77-OOS 为 25/50，而标准 Banking77 为 38/77=`0.493506`。报告中必须同时写数据集名和有效 Known 类数。

### 4.1 旧协议正确性结论

旧 v19 的**结构性规则是对的**：Gate train 没有 OOS，Router/Expert 只保留 Known，Gate
val/test 才加入 held-out 或 native OOS；`archives/submissions/s2c-submission/data` 与
当前 `assets/datasets/s2c/prepared/data/multidataset/v19` 的 CLINC、StackOverflow 和
标准 Banking77 关键 JSON 文件逐字一致。旧数据处理代码仍在
`scripts/data/active/rebuild_multi_dataset_v19.py` 和
`scripts/data/active/rebuild_v19_2_strict.py`。

但旧协议不是“干净无泄漏协议”：

- CLINC150/KIR=.50 的 Gate train/val 有 2 条相同文本，train/test 有 1 条；这些重复来自
  原始 `data_full.json`，不是当前评估器制造的。
- 标准 Banking77 的 train/test 有 3 条相同文本；BANKING77-OOS 的 Gate val/test 有 4 条
  相同 OOS 文本。
- 原审计器漏把 val/test 交集纳入 `text_isolation_ok`；该逻辑已修正，但没有修改旧数据，
  因为删除重复会破坏对论文快照的忠实复现。
- `fulltex.tex` 写“所有数据集 6:1:3”，但历史 Ours artifact `paper_results/banking77_oos/kir25_seed42/full_anchor/eval_results.json`
  使用 `12` Known、`66` OOS、test `4080`，并实际得到 `93.9852%` OOS F1；archive `banking77/kir25_seed42`
  的 `19` Known、`58` OOS、test `3080` 是另一条标准 Banking77 数据线。

所以旧协议的正确用法是：**作为 paper-faithful historical protocol 使用，但不能称为
leakage-free protocol；所有新结果必须保留实际数据键和这些限制。** 若以后需要无重复版本，
只能另称 corrected protocol，不能与论文数字直接比较。

## 5. 78.64% 到底是哪一行

当前公平汇总文件中的这行是：

```text
method=single_centroid
representation=frozen MiniLM
distance=Euclidean
boundary=mean_std
protocol=protocol_v2_fair_mean_over_5_seeds
```

该行的聚合值是 OOS F1 `78.6439%`、Known Recall `83.4976%`、false acceptance `23.0140%`，共 45 个当前 fair Gate 单元；它不是一个可以直接命名为 Frozen K=1 的旧论文结果。

它的标签已经在当前报告中固定为 **Frozen single-centroid（MOGB-Fair component；Euclidean）**。它不是：

- 旧论文完整 `Ours`；
- 旧论文后部受控表里的 Frozen K=1；
- 当前 canonical E2 的对角 Mahalanobis Frozen K=1；
- 任何包含 SmolLM Router/Expert 的 Cascade。

当前 canonical E2 Frozen K=1 的跨 9 个 dataset×KIR 单元汇总为 81.07%，而不是 78.64%。旧论文受控表对 KIR=.50 的 Frozen K=1 报告为：CLINC Mahalanobis 88.02%、StackOverflow 79.02%、Banking77-OOS 84.82%（均为 3 个 seed 的均值）。三者都不能与 78.64% 互换。

## 6. 旧 Gate 与旧完整 Cascade 的实际对账

### 6.1 H1 archive Gate 复跑（诊断，不是论文受控表）

使用 H1 archive 的归档数据、归档 `train_multisphere_corrected.py`、本地 `all-MiniLM-L6-v2`，设置 `class_centroid_mixture`、每 intent 2 个 subcenter、L2、对角 Mahalanobis；这里保留旧脚本在混合验证集上选择 λ 的行为，最终按标准 OOS F1 重新计算。因此它是“旧实现可运行”的诊断，不是论文后部固定 K 表：

| 数据集 | Gate ID Recall | Gate OOS Rejection | 标准 OOS F1 | 最终 λ |
|---|---:|---:|---:|---:|
| CLINC150 | 81.78% | 87.48% | 87.44% | 1.75 |
| StackOverflow | 85.97% | 75.08% | 79.41% | 1.00 |
| Banking77 | 82.63% | 80.32% | 81.44% | 1.00 |

这三份结果已落盘于：

```text
../artifacts/s2c/runs/historical_protocol_v1/gate_repro/
```

它们证明旧 Gate 可以在 H1 archive 上复跑，但不能证明论文主表的 H0 完整 Cascade 已复跑，也不能拿来替代下一个小节的固定 λ 受控表。

### 6.2 论文后部受控 K 表：已复现

论文后部表格固定 `lambda=1`，只改变每个 intent 的中心数，使用 H1 controlled 的 `clinc150`、`stackoverflow` 和 `banking77_oos` 快照。现有 `results/gate_only/kir_k_fixed_mean_std.csv` 与 `fulltex.tex:446-451` 的 KIR=.50 数值逐项一致；关键 K=1/K=2 如下（3 个数据 seed 的均值±总体标准差）：

| 数据集 | 距离 | K=1 | K=2 |
|---|---|---:|---:|
| CLINC150 | Diag. Mahalanobis | 88.02±1.00 | 88.07±1.06 |
| BANKING77-OOS | Diag. Mahalanobis | 84.82±1.44 | 85.57±1.37 |
| StackOverflow | Diag. Mahalanobis | 79.02±4.63 | 72.80±7.02 |

新的独立回放脚本在 H1 controlled、KIR=.50、seed=42 上得到：CLINC K=1/K=2 为 `88.1636/87.8375%`，StackOverflow 为 `84.3580/79.4071%`，BANKING77-OOS 为 `86.1560/84.6263%`；与对应逐 seed artifact 一致。它使用固定 λ=1，不读取 validation OOS。

### 6.3 同一旧协议下的 Frozen/Trainable K=1 对照

本轮新增的 `run_trainable_minilm_historical_v1.py` 固定归档 train/val/test、Known 清单、MiniLM、K=1 对角 Mahalanobis、`mean+std λ=1` 和标准 OOS evaluator；Trainable 的表示训练与 checkpoint 选择只使用 Known train/validation，OOS 不进入训练或选择。

| H1 数据集 | Frozen K=1 OOS F1 | Trainable K=1 OOS F1 | Frozen Known Recall | Trainable Known Recall |
|---|---:|---:|---:|---:|
| CLINC150 | 88.16% | 89.56% | 73.11% | 72.80% |
| StackOverflow | 84.36% | 91.49% | 82.93% | 84.00% |
| Banking77 | 78.98% | 82.99% | 85.59% | 82.70% |

上表是 H1 archive 标准 Banking77 的 seed=42 控制；论文受控表真正对应的 BANKING77-OOS 结果另列如下：

| H1 controlled 数据集 | Frozen K=1 | Trainable K=1 | Δ OOS F1 | Frozen FA | Trainable FA |
|---|---:|---:|---:|---:|---:|
| CLINC150 | 88.16% | 89.56% | +1.40pp | 6.49% | 3.63% |
| StackOverflow | 84.36% | 91.49% | +7.14pp | 14.61% | 2.20% |
| BANKING77-OOS | 86.16% | 88.85% | +2.70pp | 21.30% | 15.36% |

这三行都用 H1 controlled 的同一 test、同一 Known 列表、同一 K=1/Diag-Mahalanobis/λ=1/evaluator；不是 3-seed/5-seed 均值。Trainable 在论文受控单中心基线上三项均为正，且没有通过“全部拒绝”获得收益：Known Recall 分别为 CLINC `72.80%`、StackOverflow `84.00%`、BANKING77-OOS `81.90%`。

| 数据集 | Δ AUROC | Δ false acceptance | Δ Known Recall |
|---|---:|---:|---:|
| CLINC150 | +3.92pp | −2.86pp | −0.31pp |
| StackOverflow | +4.17pp | −12.41pp | +1.07pp |
| BANKING77-OOS | +0.12pp | −5.94pp | −5.80pp |

这里的 Δ 均为 Trainable−Frozen。CLINC/StackOverflow 的 AUROC 提升和误接收下降说明表示训练改变了 OOS score 的排序；BANKING77-OOS 的主要收益是工作点改善，伴随一定 Known coverage 代价，不能写成三个数据集都以相同机制获益。

边界合同会改变绝对 OOS F1：CLINC 的 H1 主几何 λ=.5 回放为 Frozen `86.06%` → Trainable `87.01%`；λ=1.5 回放为 `87.42%` → `90.41%`。BANKING77-OOS 的 λ=1 结果已超过论文主表 Banking `88.23%`，但这不等于完整 Cascade 已被重跑。

机制上，Trainable 的收益主要是 OOS score 排序和误接收率下降，而不是简单改变阈值或把所有样本都拒掉。

运行目录：

```text
../artifacts/s2c/runs/historical_protocol_v1/minilm_k1/
```

### 6.3.1 补充判因实验：协议变化与表示训练分开

为回答“是不是换了数据集”和“Trainable 是否真的优于 Frozen”，补充实验固定了同一个
`all-MiniLM-L6-v2`、K=1、对角 Mahalanobis、`lambda=1`、threshold=1 和 OOS-positive
evaluator。旧 H1 v19 使用 3 个数据集、3 个 KIR 和 3 个 seed，共 27 个配对运行单元；
每个单元同时产生 Frozen 和 Trainable 结果。

KIR=.50 的三 seed 均值如下：

| 数据集 | Frozen K=1 | Trainable K=1 | Trainable−Frozen |
|---|---:|---:|---:|
| CLINC150 | 88.02±1.00% | 89.50±0.84% | +1.47 pp |
| StackOverflow | 79.02±4.63% | 88.78±2.71% | +9.76 pp |
| BANKING77-OOS | 84.82±1.44% | 88.47±0.82% | +3.65 pp |

Trainable−Frozen 的 OOS F1 差值在全部 9 个 dataset×KIR 单元均为正：

| 数据集 | KIR=.25 | KIR=.50 | KIR=.75 |
|---|---:|---:|---:|
| CLINC150 | +0.80 pp | +1.47 pp | +1.84 pp |
| StackOverflow | +1.52 pp | +9.76 pp | +12.03 pp |
| BANKING77-OOS | +1.00 pp | +3.65 pp | +3.05 pp |

协议归因实验把 Frozen K=1 固定在 KIR=.50、seed={13,42,87}：

| 输入协议 | CLINC150 | StackOverflow | Banking 类数据 |
|---|---:|---:|---:|
| H1 old v19 | 88.02±1.00% | 79.02±4.63% | BANKING77-OOS：84.82±1.44% |
| 当前 protocol_v2 | 89.31±0.79% | 77.29±5.10% | Banking77：79.58±2.04% |
| 解释 | 同名数据但源文件、Known 清单和 OOS 构造不同 | 同名数据但结果下降，说明不能只看数据集名称 | 两行不是同一数据集键，不能计算严格配对差值 |

因此，问题 1 的答案不是“单纯换了一个数据集”：CLINC150 和 StackOverflow 也发生了
源文件、Known intent、split 和 OOS 构造变化；Banking 还从历史 `BANKING77-OOS` 变成了
当前标准 `Banking77`。补充实验说明协议确实会改变绝对结果，但不能把所有差异归因于
一个单独因素。

本轮结果和图源：

- 逐运行结果：[historical_protocol_supplement_results.csv](../../results/analysis/historical_protocol_v2/historical_protocol_supplement_results.csv)
- KIR 聚合：[historical_protocol_kir_summary.csv](../../results/analysis/historical_protocol_v2/historical_protocol_kir_summary.csv)
- 协议归因：[historical_protocol_attribution.csv](../../results/analysis/historical_protocol_v2/historical_protocol_attribution.csv)
- OOS 转移：[historical_oos_transitions.csv](../../results/analysis/historical_protocol_v2/historical_oos_transitions.csv)
- 运行与出图脚本：[build_historical_protocol_supplement_v2.py](../../tools/analysis/build_historical_protocol_supplement_v2.py)

在上述性能表之外，实验阶段又生成了面向论文实验解释的 OOS 机制证据：使用同一 OOS 样本
配对计算 `Δ nearest distance`、`Δ radius` 和 `Δ score`，并将总 score 变化精确拆分为
distance contribution 与 radius contribution；同时按固定规则导出 Trainable-only、
Frozen-only 和 both-wrong 的 OOS 案例。该机制图组不修改训练或 checkpoint 选择。

- OOS 样本级机制表：[oos_pairwise_mechanism.csv](../../results/analysis/historical_protocol_v3/oos_pairwise_mechanism.csv)
- OOS 机制汇总：[oos_mechanism_summary.csv](../../results/analysis/historical_protocol_v3/oos_mechanism_summary.csv)
- OOS 案例表：[oos_case_examples_seed42.csv](../../results/analysis/historical_protocol_v3/oos_case_examples_seed42.csv)
- 训练动态：[training_dynamics.csv](../../results/analysis/historical_protocol_v3/training_dynamics.csv)
- v3 出图脚本：[build_historical_oos_mechanism_v3.py](../../tools/analysis/build_historical_oos_mechanism_v3.py)

### 6.4 为什么不能直接用论文主表的高数字作复现证据

论文主表在 KIR=.50 报告的 `Ours` OOS F1 为 CLINC 91.96%、StackOverflow 89.71%、Banking77 88.23%。这些是 H0/主 Cascade 的系统级 reference，不是 H1 Gate-only replay。当前 H0 的原始字节级 `data/v19`、历史 Gate detector、匹配的 Router/Expert 和必要语义输入仍缺失；CLINC 的 75 个 Known 与 5499 条测试数据已有锚点重建版本。现有能找到的 Router/Expert checkpoint 均由日志确认属于 H1/multi-dataset，不是 H0；它们位于
`artifacts/s2c/outputs/experiments/components/` 找到，H1 还有
`artifacts/s2c/outputs/experiments/cascade_full/gpu_kir50/` 的 36 个下游评估，但它们不能
直接替代 H0。因而目前只能说“锚点已审计”，不能说主表已逐字复现。

其中 StackOverflow/KIR=.50 的 full-anchor 文件还同时出现：

- `fast_gate_metrics`: ID Recall 85.97%、OOS Rejection 75.08%；
- `primary_metrics.oos_f1`: 59.10%；
- 顶层 `metrics.oos_f1`: 89.71%；
- 同一错误分解中 OOS false accept=1598/2997，已不可能对应 89.71% 的 OOS F1。

此外，CLINC 的主表 reference `0.9196` 与另一个保存的 frozen prototype anchor `0.909619` 也不是同一字段/配置。上述差异不是“随机种子导致的轻微差异”，而是 H0 输入缺失、系统级配置不同以及历史 anchor 的字段/评价版本不一致。当前应把论文表数字标记为 **historical reported reference**，把可由预测和混淆矩阵核验的数字标记为 **replay evidence**，两者不能混称。

## 7. 最终采用的稳定协议

后续报告固定分成四条 lane，不再把结果合成一张排名表：

### Lane H0：论文主 Cascade

- 输入：恢复论文配置实际引用的 `data/v19`、旧模型、旧语义 Gate、旧原型和评估环境；
- 主 KIR：论文表的 `.25/.50/.75`，主锚点 seed=42；
- 规则：Gate、Router、Expert 和语义 Gate 必须作为一个整体评估，不能用 H1 单中心 Gate 填回；
- 状态：`paper_results` 保存了 H0 锚点；CLINC 的 `data/v19` 已有标注的锚点重建版本，
  但原始 H0 数据、对应 Known 列表、历史模型和语义/原型输入仍未形成同一可重跑链。

### Lane H1：历史 Gate 与 Trainable 控制

- 输入：H1 controlled 的 prepared `multidataset/v19`，或 H1 archive 的 `archives/submissions/s2c-submission/data`；
- 主对照：同一数据根、Known 清单、K=1、Diag-Mahalanobis、固定 λ、标准 OOS-positive F1；
- 规则：论文受控表固定 λ=1；论文主几何的 CLINC λ=.5 单独标记；含 OOS validation 的 λ 选择另列为 historical OOS-assisted；
- 状态：论文受控 Frozen K=1/K=2 表已复现；Trainable K=1 已完成三数据集×三 KIR×三 seed 的控制实验；
  H1 KIR=.50 下已有 36 个固定下游 Cascade 评估单元，但不冒充 H0 主表。

### Lane A：当前 protocol_v2 / TextOIR

- 数据准入：`s2c/data/canonical|registries|views|exports/protocol_v2_textoir_v1`；
- S2C：Known-only train/calibration，test OOS 只作最终评估，正式 seeds=`13,42,87,100,123`；
- TextOIR 原生基线：使用干净 TEXTOIR commit `dffe2b1b848a069a6808f8089b4cb9bd16e2062b` 的 `textoir/data/{oos,banking,stackoverflow}`，由原生 `DataManager` 用 RandomState(seed) 选 Known，保持 raw test 顺序；
- compatibility export：只用于当前同样本适配，不能写成 byte-identical 论文数值。

### Lane C：OOS 机制分析

- 只在同一 lane 内比较 score ranking、OOS precision/recall/F1、AUROC/AUPR 和 false acceptance；
- Known Recall/false reject 只作 coverage guard；
- 不把历史 OOS-assisted λ 选择、当前 Known-only checkpoint 选择、Gate-only 和完整 Cascade 混写。

## 8. TextOIR 基线当前闭合度

当前固定 TEXTOIR commit 的 native runner 使用原始 `textoir/data/{oos,banking,stackoverflow}`、原生 `DataManager` 和 RandomState Known 抽样；下面的 seed=0/1/2 是已有 compatibility matrix，不冒充当前 S2C 的正式五 seed fair 主表。

| 方法 | 已完成范围 | 当前证据 | 结论 |
|---|---|---|---|
| MSP | 3 数据集×3 KIR×3 seed，27/27 | `artifacts/s2c/outputs/experiments/cluster_separability_v19/textoir_protocol/official_runs` | 可用 native compatibility 结果；KIR=.50 OOS F1 均值：CLINC 61.68%、Banking 44.67%、SO 33.49% |
| DOC | 3 数据集×3 KIR×3 seed，27/27 | 同上 | 可用 native compatibility 结果；KIR=.50 OOS F1 均值：CLINC 88.36%、Banking 72.57%、SO 71.67% |
| ADB | 3 数据集×3 KIR×3 seed，27/27 | 同上 | 可用 native compatibility 结果；KIR=.50 OOS F1 均值：CLINC 89.22%、Banking 78.65%、SO 86.98% |
| DA-ADB | StackOverflow/KIR=.50/seed 42,87,100，3/3 | `artifacts/s2c/external/da_adb_gpu_runtime_v1` | 可用外部 compatibility 结果，OOS F1=72.48±6.24%；三数据集矩阵尚未闭合 |
| KNNCL | 0 个有效 final-metrics 单元 | `textoir_current_protocol_v1` 的失败 attempts；上游 `args.anum_labels` typo 已加入隔离修复 | 仍 blocked；不能用旧 `results.csv` 中残留的 EliDecide 行代替 KNNCL |

原始 TEXTOIR 表格 `textoir/open_intent_detection/results/results.md` 只作为 reported reference；native compatibility 结果、当前 S2C fair 结果和论文主表不放在同一排名列。KIR=.50 的可读汇总见 [textoir_native_compatibility_kir50.csv](../../results/analysis/historical_protocol_v1/textoir_native_compatibility_kir50.csv)。

## 9. OOS 对照图

本轮图件改为面向论文实验解释的 OOS 机制证据，不再把论文 full Cascade reference 和
Gate replay 放进同一组柱状图。v2 的协议组成/描述性图保留为 rejected draft，不作为活动入口。

### 主图

![Trainable 对 OOS 性能的改善](../../figures/historical_protocol_oos_v3/main_oos_performance.png)

![OOS score 形成与决策修正](../../figures/historical_protocol_oos_v3/main_oos_mechanism.png)

### 补充图

- [Trainable MiniLM 训练动态](../../figures/historical_protocol_oos_v3/supp_training_dynamics.png)：训练 loss 与 ID-only checkpoint selection score。
- [距离/半径/score 分解](../../figures/historical_protocol_oos_v3/supp_distance_radius_score.png)：解释总 OOS score 变化来自哪里。
- [OOS 几何密度](../../figures/historical_protocol_oos_v3/supp_oos_geometry.png)：只展示 OOS 密度的 Frozen/Trainable PCA 视图。

所有图均提供 PNG、PDF、SVG 和 600 dpi TIFF；源表位于
`results/analysis/historical_protocol_v3/`，出图脚本为
`tools/analysis/build_historical_oos_mechanism_v3.py`。

## 10. 当前进度判定

已完成：

- 旧协议与当前协议的逐项差异已确认并有文件证据；
- 论文受控 Gate KIR=.50 的三数据集、两种距离、K=1..5 表已复现；
- H1 archive 的 Frozen/Trainable K=1 三数据集×三 KIR seed=42 已跑通，并补充 H1 controlled BANKING77-OOS；
- 78.64% 的真实来源和它与论文方法的关系已厘清；
- 历史 full-anchor 的评价字段不一致已定位。
- 旧 v19 的数据处理代码、归档快照和实际重复样本已核对；后续新实验默认使用
  `historical_v19_paper_main`，不再新开 `protocol_v2_textoir_v1`。
- 旧协议 3 个数据集×3 个 KIR×3 个 seed 的 Frozen/Trainable 配对实验已完成，协议归因表和 OOS 状态转移表已生成。
- 面向论文实验解释的 OOS 机制图组已生成：OOS 性能、score 形成、训练动态、距离/半径分解和 OOS 几何密度。
- TextOIR 的 MSP、DOC、ADB native compatibility 矩阵已闭合，DA-ADB/KNNCL 的剩余边界已定位。

尚未完成：

- H0/H1 归档 KIR=.25/.75 的完整 Gate/Cascade 对账；本轮只完成旧 v19 的 K=1 Frozen/Trainable 控制；
- 原始字节级 H0 `data/v19`、历史 Gate detector、匹配模型和语义输入的完整恢复及 Cascade 重跑；当前已完成 CLINC 数据入口的锚点重建；
- 当前 Lane A 的 DA-ADB 全数据集矩阵和 KNNCL final metrics 尚未闭合；
- H0 严格 full Cascade 的输入恢复与完整重跑仍未完成。

因此，当前可以明确回答 78.64% 的来源、复现论文受控 Gate 表、并证明 Trainable 在同协议 K=1 上全面优于 Frozen；但仍不能声称“论文 H0 完整主表和 TextOIR 所有方法已全部严格复现”。
