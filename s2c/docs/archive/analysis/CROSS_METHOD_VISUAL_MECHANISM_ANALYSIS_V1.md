## 核心结论

本报告只把图像中能够区分机制的证据写成结论，第一指标始终是 OOS F1。所有外部方法都保留合同层，不把不同骨干、监督或历史 Cascade 混成一个排名。

1. 当前 S2C-Trainable-K1 的 OOS F1 主要来自两层共同作用：Known-only 训练最后两层 MiniLM + projection 后，Known/OOS 的 gate score 排序明显改善；K=1 单中心又避免了固定多中心接受区域并集带来的 open-space over-coverage。现有阈值无关图、score gap 图和同一 Known Recall frontier 都支持“不是单纯换了一个阈值工作点”。强度：较强支持。

2. S2C 相对 ADB 的差异不是一个统一机制。CLINC150 和 Banking77 的 OOS F1 优势主要伴随 OOS Recall 上升、OOS Precision 下降以及 Known rejection 增加；更像是在默认工作点用更激进的拒绝换取 OOS Recall。StackOverflow 则不同：Known rejection 没有增加，KIR=.50/.75 的 OOS Precision 上升，S2C 的优势更像是错误排序和 coverage 重新分配。强度：跨 KIR 为中等；StackOverflow 同样本转移为中等偏强，但不能把 BERT/MiniLM 差异排除。

3. S2C 相对当前 MOGB-Fair 组件的核心差异是相反的失败模式。MOGB-Fair 的低 False Acceptance 与极低 Known coverage 同时出现，属于 local under-coverage；S2C 恢复大量 Known，同时承受有限的 OOS Recall 损失。matched-Known-Recall frontier、AUROC/AUPR 以及 default-versus-oracle 图共同表明，MOGB 的差距不只是阈值。这里的结论只针对当前 MiniLM 组件实验，不等价于完整论文 MOGB。

4. “固定多中心 K>1”与“MOGB local balls”不是同一种坏处。固定 K=2 的失败是接受区域并集扩大，导致 OOS False Acceptance 和 open-space risk 增加；MOGB-Fair 的失败是球/半径太保守，导致 Known 被拒绝。不能用“多中心不好”概括二者。

5. DA-ADB 当前 protocol_v2 结果比 S2C 低且 False Acceptance 更高；这已经否定了“只是 DA-ADB 选了更严格阈值”这一简单解释，但没有把低分归因到 DA-ADB 算法本身，因为当前图没有同骨干、同样本的 score frontier 或 error transition。

6. 当前 S2C 低于历史 fulltex Ours 的最大事实差距确实发生在 Banking77/KIR=.75：历史 Ours 为 86.49，当前 S2C 五 seed 均值为 68.65，下降 17.84pp。现有图能可靠定位差距集中在哪些 dataset/KIR，但不能区分 representation、boundary、Cascade 和 protocol 的因果贡献；必须补 2×2 拆解。

## 证据分层与阅读规则

| 层 | 本报告中的名称 | 可审计设置 | 允许的结论 |
|---|---|---|---|
| A | Published/public results | 论文、公开仓库、TextOIR/历史表中的公开结果 | 只能作为公开参考；必须保留原合同 |
| B | Current-protocol reproduction / compatibility rerun | 当前 protocol_v2_textoir_v1；S2C 为 MiniLM，ADB/DA-ADB 为 BERT/TextOIR 外部合同 | 可做合同感知的机制对照；不能做同骨干 SOTA |
| C | Historical Ours | fulltex.tex 的 historical_full_cascade；Gate→Router→Expert | 只能与历史表内方法比较 |
| D | Fair component experiment | 当前同 split/KIR/seed/evaluator 的 MOGB-MiniLM、partition/boundary swap 等 | 可做组件机制归因；不能称完整 MOGB |

本报告使用的主要数字都来自可追溯 CSV，并在相应段落标明 dataset、KIR、seed 数、backbone、supervision 和 metric。9 格当前 fair 均值来自 3 dataset × 3 KIR × 5 seeds；ADB KIR 结果为 45/45 个 BERT/TextOIR 外部合同单元；DA-ADB 当前结果为 StackOverflow/KIR=.50、seed 42/87/100 的三 seed 外部合同；DCLOOS reduced 结果为 BERT + pseudo-OOS + 外部 SQuAD OOS 的 KIR=.75/seed=888 单格。

## 为什么当前 S2C-Trainable-K1 得到这个 OOS F1

![Trainable 与 Frozen 的 score gap 和 false acceptance](../../figures/archive/analysis/representation_geometry_visuals_v1/score_gap_false_accept.png)

图中 Trainable 的 Known/OOS median score gap 在三个数据集都高于 Frozen，且 False Acceptance 下降；StackOverflow 的变化最明显。它回答的是：训练后的表示让 gate score 更容易排序，而不是只把同一个 score distribution 的阈值往一侧移动。该图仍是 S2C 内部控制，不是外部方法的同骨干比较。

![同一 Trainable 表示下不同 detector 的 Known/OOS 工作点](../../figures/archive/analysis/native_baselines_trainable_v1/kir050_trainable_representation_tradeoff.png)

同一 Trainable MiniLM 表示下，Gate 位于较高 OOS F1、较低 False Acceptance 的区域；MSP/Energy/kNN/LOF 的工作点不同，尤其 StackOverflow 的 native MSP/Energy 处于高 Known Recall 但高 False Acceptance 区域。这支持“表示 + 几何 Gate”是组合机制，而不是“微调后任何 detector 都会自动变好”。但这张图不是完整 KNNCL 的公平重跑。

在当前 fair 9 格中，S2C-Trainable-K1 OOS F1 均值为 85.75%，F1-All 均值为 83.36%，OOS F1 在 9 格中 8 格第一。这个结果本身不是机制结论；与上面 score gap、AUROC/AUPR 和 error-budget 图合读后，较强支持的因果链是：

训练表示改善 Known/OOS 的 score ranking
→ 单中心 Gate 在默认工作点减少高风险 OOS 误接收
→ 同时不承担固定多中心 union 的额外 open-space 暴露
→ OOS F1 在 Known coverage 与 OOS rejection 之间取得当前最平衡的工作点。

当前仍不能说 representation 单独解释全部分数，因为 Gate 形式也有独立作用；同一 Trainable 表示上的 detector control 说明 boundary/detector 仍是必要组成。

## S2C vs ADB / DA-ADB

### S2C vs ADB：CLINC150、Banking77 与 StackOverflow 不是一种机制

![S2C 与 ADB 的跨数据集/KIR 机制热力图](../../figures/archive/analysis/trainable_vs_adb_mechanism_summary_v1/dataset_kir_mechanism_heatmaps.png)

#### 观察

当前 45/45 BERT/TextOIR ADB 单元与 MiniLM S2C 的 OOS F1 差值如下，数字来自同 dataset/KIR 的五 seed paired summary：

| Dataset | KIR=.25 | KIR=.50 | KIR=.75 | 主要可见 error-budget |
|---|---:|---:|---:|---|
| CLINC150 | +3.57pp | +1.05pp | −1.91pp | S2C Known rejection +16.4～17.4pp；OOS FA −10.3～−13.5pp |
| Banking77 | +8.85pp | +8.60pp | +2.39pp | S2C Known rejection +5.6～7.9pp；OOS FA −16.1～−17.9pp |
| StackOverflow | +2.34pp | +0.46pp | +2.60pp | S2C Known rejection −0.8～−3.3pp；KIR=.50/.75 OOS FA 略高 |

OOS Precision/Recall 分解图显示：CLINC150 的 S2C OOS Recall 增加 10.3～13.5pp，但 OOS Precision 下降 3.8～14.3pp；Banking77 的 Recall 增加 16.1～17.9pp，但 Precision 下降 0.9～8.6pp；StackOverflow 在 KIR=.50/.75 的 Precision 分别增加 1.8/4.1pp，而 Recall 只小幅下降。

#### 竞争性假设

H1：S2C 的 representation/score ranking 真的使困难 OOS 更靠近“应拒绝”的一侧。

H2：S2C 只是采用更严格的 rejection operating point，所以 OOS Recall 上升，但 Known Recall/acceptance 被牺牲；OOS F1 的提高主要来自工作点，而非排序。

#### 图像证据与判断

- CLINC150、Banking77：跨 KIR 的热力图和 OOS Precision/Recall 图都呈现“Recall 增、Precision 降、Known rejection 增”的同向结构。H2 得到较强支持；不能仅凭这些图宣称 H1 已被证明。
- StackOverflow：同样本 state transition 图是 StackOverflow/KIR=.50、protocol_v2、seed 42/87/100 的 18,000 行 aggregate。S2C 正确而 ADB 被拒的 Known 样本为 434，反向的 S2C 被拒而 ADB Known 正确为 209；OOS 中 ADB False Acceptance 被 S2C 修正为正确拒绝的为 439，反向 S2C False Acceptance 而 ADB 正确拒绝的为 380。这个方向不是“只拒绝更多”：S2C 同时恢复 Known 和修正部分 ADB 的 OOS 错误。
- 但 state transition 只给出状态计数，不给原始文本、intent 语义或 per-sample score scatter；它不能证明这些样本是 Near-OOS、局部多模态还是某一类 lexical pattern。

因此，对 ADB 的结论是：

- CLINC150/Banking77：OOS F1 优势主要由 OOS Recall 驱动，并伴随 Known coverage 牺牲和 OOS Precision 损失；“更保守/更强拒绝”的解释较强。
- StackOverflow：S2C 的 OOS 优势更接近错误排序和 coverage 重新分配，尤其 KIR=.50/.75；“单纯阈值更严格”的解释被同样本转移部分削弱。
- 由于 ADB 是 BERT/TextOIR、S2C 是 MiniLM，representation、训练目标、score 语义和边界仍然混在一起。现有证据是机制支持，不是算法组件的因果隔离。

![S2C/ADB OOS Precision、Recall 与 Known acceptance 的分解](../../figures/archive/analysis/trainable_vs_adb_oos_decomposition_v1/oos_precision_recall_decomposition_heatmaps.png)

![StackOverflow/KIR=.50 的 S2C/ADB 同样本状态转移](../../figures/archive/analysis/trainable_vs_adb_error_budget_v1/state_transition_heatmap.png)

### 为什么高 KIR 后优势缩小甚至反转

这不是所有数据集都同样发生：

- CLINC150：S2C 的 OOS F1 差值从 +3.57pp、+1.05pp 变为 −1.91pp；KIR=.75 时 OOS Precision 差值已经是 −14.3pp。S2C 的 OOS Recall 仍高于 ADB，但 Precision 损失成为 OOS F1 的瓶颈。
- Banking77：优势从 +8.85/+8.60pp 缩至 +2.39pp；主要变化也是 Precision 差值从 −0.9pp 扩大到 −8.6pp，而 Recall 优势仍约 +16pp。
- StackOverflow：没有同样的反转；KIR=.50/.75 的 Precision 差值为正，Known acceptance 也高于 ADB，优势保持在 +0.46/+2.60pp。

所以目前能支持的机制是：KIR 增大以后，某些数据集上 S2C 默认 boundary 的 OOS Precision/False Acceptance 变成主导瓶颈，Recall 优势无法补偿；这比“高 KIR 一定更差”更准确。原因究竟是 OOS 语义难度、Known/OOS 局部密度变化，还是 score calibration 失配，现有图不能单独区分。

### S2C vs DA-ADB

![DA-ADB 当前协议的 Known/OOS 工作点](../../figures/da_adb_current_protocol_summary_v1/current_protocol_tradeoff.png)

#### 观察

StackOverflow/KIR=.50、当前 protocol_v2、BERT/TextOIR DA-ADB 三 seed 的 OOS F1 均值为 72.48±6.24%，False Acceptance 均值为 29.07±11.19%；同 seed 的 S2C-Trainable-K1 为 88.21±1.72%，False Acceptance 为 8.18±3.36%。当前图上 DA-ADB 的三个点同时位于更高 False Acceptance、较低 OOS F1 的区域。

历史兼容单格的 DA-ADB OOS F1=90.90% 与当前 protocol_v2 seed=42 的 70.82% 也不在同一合同：Known list、split/seed、运行环境和当前 protocol 都不同。

#### 竞争性假设

H1：DA-ADB 算法自身的 boundary/ranking 在当前数据上更容易吸收 OOS。

H2：差异主要来自 BERT/TextOIR 训练、当前 Known split/list、seed、依赖或协议迁移；当前图只看到最终工作点，不能把这些因素拆开。

#### 判断

“DA-ADB 只是更严格阈值”的解释被图否定，因为当前 DA-ADB 的 False Acceptance 反而更高。可以说当前 DA-ADB 落在不利的 OOS operating regime；不能说已经证明 DA-ADB 的算法机制比 S2C 差。缺少 matched Known Recall frontier、same-sample score scatter 和 error transition，结论强度为：工作点差异已验证；算法归因无法判断。

## S2C vs MOGB

### 先区分两个失败模式

![S2C-Trainable-K1 与 MOGB-Fair 的 paired delta heatmaps](../../figures/s2c_vs_mogb_mechanism_dashboard_v1/paired_delta_heatmaps.png)

同一 protocol_v2、MiniLM、5 seeds 的 MOGB-Fair 组件对照中：

- S2C 的 Known Recall 在九格都更高，差值约 +36.5～+62.9pp；
- S2C 的 OOS Precision 在九格都更高，但 OOS Recall 更低；
- S2C 的 False Acceptance 更高，说明 S2C 并不是靠拒绝更多 OOS 获胜；
- F1-All 和 OOS F1 仍然更高，说明恢复 Known 的收益与更高 OOS Precision 超过了 OOS Recall 损失。

这正是 MOGB local under-coverage 的形状：MOGB-Fair 的低 False Acceptance 与低 Known Recall 同时出现。它不能被解释成“拒绝 OOS 特别好”，因为 OOS F1 是 Precision/Recall 平衡，且大量 Known 被拒绝。

![MOGB-Fair 到 S2C 的 Known/OOS error-budget transfer](../../figures/s2c_vs_mogb_mechanism_dashboard_v1/error_budget_arrows.png)

图中的 square 到 circle 显示：三数据集、三个 KIR 下，MOGB 的 Known false rejection 大幅下降，但 OOS false acceptance 会上升。该方向与“boundary under-coverage → 放宽/改进 boundary 后恢复 Known，同时付出 open-space cost”一致。

### 竞争性假设：boundary 还是 score ranking

H1：MOGB 只是默认半径/边界太保守；重新选阈值后可以追上 S2C。

H2：S2C 的 representation/score ranking 更好；即使在 matched Known Recall 或 MOGB oracle workpoint 下，S2C 仍位于更优 OOS frontier。

现有图给出三个互相独立的判断：

- matched-known-recall-frontier：在相同 Known Recall 上，S2C 曲线整体高于 MOGB-Fair；这减少了默认阈值差异的解释空间。
- threshold-free delta heatmap：S2C 相对 MOGB-Fair 的 AUROC、AUPR-OOS 和 standardized score separation 在 45 个 paired cells 都为正；这直接支持排序/分离能力差异。
- default-versus-oracle OOS F1：MOGB 从 default 到 oracle 的改善仍不能追平 S2C，说明阈值不是全部原因。

结论强度：S2C 相对 MOGB-Fair 的 boundary coverage 差异已较强支持；score ranking 差异也有较强支持；representation 与 boundary 的相对因果权重仍不能由单一图完全分离。

![同一 Known Recall 下 S2C/MOGB 的 OOS F1 frontier](../../figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/matched_known_recall_frontier.png)

![S2C 与 MOGB-Fair 的 threshold-free 指标差值](../../figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/threshold_free_delta_heatmap.png)

### 固定多中心与 MOGB local 的相反失败

固定 K=2 的 boundary expansion waterfall 显示：增加第二中心后，Known 恢复量有限，但新增接受的 OOS 明显更多；在 StackOverflow 上，这与 K=2 的 OOS F1 下降和 False Acceptance 上升一致。这是 acceptance-region union 的 open-space risk。

MOGB-Fair 则是另一个方向：默认 workpoint 的 False Acceptance 很低，但 Known False Rejection 极高。MOGB calibration workpoints 把 Known coverage 往上推时，False Acceptance 很快上升、OOS F1 下降；这支持 local balls/radius 的 under-coverage，而不是 fixed-K union over-coverage。

![MOGB workpoint 的 False Acceptance/False Rejection 风险](../../figures/archive/analysis/mechanism_closure_v1/mogb_risk_workpoints.png)

逐球风险图进一步显示，前 10% 高风险 balls 集中了约 0.9～0.98 的 OOS False Acceptance；support quartile 图又显示风险不只由 tiny balls 决定，较大 support quartile 也可能贡献主要错误。因此“某些 tiny ball 有问题”是局部事实，不是全局充分解释。

### 组件桥接的边界

MOGB partition + S2C boundary 比 MOGB-Fair 有改善，但仍落后 S2C-Trainable-K1；S2C partition + MOGB boundary 也不能恢复 S2C 水平。该组件交换结果支持 representation/adaptation 与 boundary 都在起作用，但它们不是完整论文方法的公平替代。当前所有结论应写成 MOGB-Fair 或 fair component，不写成“完整 MOGB 被 S2C 超过”。

## S2C vs KNNCL

现有证据不足以回答用户要求的“local kNN density 与 class centroid + boundary 谁在多模态 intent 上更好”。

可见证据只有两层：

1. historical fulltex heatmap 中有 KNNCL 的历史 OOS F1，但它属于旧 Cascade/旧 split/旧骨干合同，不能解释当前 S2C 与 KNNCL 的样本差异。
2. native detector control 在同一 Trainable MiniLM 表示下包含 kNN，StackOverflow/KIR=.50 中 kNN 的工作点低于 Gate；这只能说明当前 Gate 在该表示上的 detector control 更好，不能代表完整 KNNCL。

没有找到可审计的当前同协议、同 seed、同 backbone KNNCL final metrics，也没有 intent multimodality × S2C–KNNCL gap 图。因此当前结论是：证据缺失，不能声称 S2C 在单峰 intent 上一定占优，也不能声称 KNNCL 在多模态 intent 上已经占优。

## S2C vs DCLOOS

![外部监督条件参照图](../../figures/archive/analysis/experiment_comparison_overview_v2/external_supervision_reference.png)

这张图明确标出 DCLOOS reduced 使用 BERT、pseudo-OOS 与外部 SQuAD OOS，并且 KIR/seed 与 S2C、ADB 不同。它是实验合同图，不是机制解释图。

#### 竞争性假设

H1：DCLOOS 的优势来自 external OOS 与测试 OOS 的语义覆盖，尤其覆盖 near-OOS 或高密度混淆样本。

H2：优势来自 BERT representation、pseudo-OOS 训练方式、boundary 或其它训练细节，与 external OOS 覆盖无关。

当前图只能验证“监督条件不同”，不能区分 H1/H2。仓库中没有测试 OOS 与 external OOS 的语义相似度、nearest-external distance、external-similarity bin × OOS Recall 的配对图，也没有同协议 DCLOOS 完整 final metrics。故不能写“DCLOOS 的优势具体帮助了哪些测试 OOS”，只能写“存在更强 OOS 监督条件，当前不可公平归因”。

## 当前 S2C vs 历史 Ours

![历史 full Cascade 的 OOS F1 heatmap](../../figures/archive/analysis/historical_sota_comparison_v1/historical_oos_f1_heatmap.png)

![历史 fulltex 与当前 Trainable 的 protocol-mismatch reference](../../figures/archive/analysis/minilm_trainable_5seed_fair_v1/trainable_vs_fulltex_reference.png)

#### 观察

历史 fulltex Ours 是完整 Gate→Router→Expert Cascade；当前 S2C-Trainable-K1 是 Known-only MiniLM + K=1 Gate。两者不是同一个方法。

当前五 seed protocol_v2 与 historical_full_cascade 的逐格差异：

| Dataset/KIR | Historical Ours OOS F1 | Current S2C OOS F1 | Current − historical |
|---|---:|---:|---:|
| Banking77/.25 | 93.99 | 91.65 | −2.34pp |
| Banking77/.50 | 88.23 | 83.56 | −4.67pp |
| Banking77/.75 | 86.49 | 68.65 | −17.84pp |
| CLINC150/.25 | 95.01 | 95.16 | +0.15pp |
| CLINC150/.50 | 91.96 | 90.44 | −1.52pp |
| CLINC150/.75 | 87.10 | 82.85 | −4.25pp |
| StackOverflow/.25 | 94.47 | 95.48 | +1.01pp |
| StackOverflow/.50 | 89.71 | 87.67 | −2.04pp |
| StackOverflow/.75 | 75.57 | 76.30 | +0.73pp |

Historical数字来自 fulltex.tex；当前数字来自 current protocol_v2、5 seed paired reference。该表只回答“差距发生在哪里”。它不能回答“为什么”。

#### 竞争性假设

H1：representation 变化造成 Banking77 高 KIR 的 OOS 分离下降。

H2：当前 gate boundary/threshold 与历史 boundary 不同；representation 本身未必是主因。

H3：历史 Router/Expert Cascade、Known split/list 或 protocol/数据流水线改变，导致当前 Gate-only 结果不可直接复现。

现有图没有 Frozen representation + Historical boundary、Frozen representation + Current boundary、Trainable representation + Historical boundary、Trainable representation + Current boundary 四个 cell，所以 H1/H2/H3 没有被排除。当前 Known F1/Recall 或 Accuracy 更好也不能抵消 OOS F1 在 Banking77/KIR=.75 的 17.84pp 下降；该下降必须被明确视为问题。

结论强度：差距定位已验证；机制解释无法判断。

## 图片分析卡片

下表只列真正改变机制判断的图片；普通 OOS F1 bar chart 或只有结果排名的图没有列入。

| 图片名称 | 比较方法 | Dataset/KIR | 图实际展示 | 能解释的问题 | 支持的机制 | 竞争解释 | 证据强度 | 论文正文 | 当前不足 |
|---|---|---|---|---|---|---|---|---|---|
| dataset_kir_mechanism_heatmaps.png | S2C vs ADB | 3 dataset × 3 KIR；5 seeds；BERT/MiniLM | CLINC/Banking 的 Known rejection 增、OOS FA 降；SO 的 Known rejection 不增 | OOS F1 差值来自哪一类错误 | dataset-specific rejection/coverage trade-off | backbone 与训练合同混杂 | 中 | 是，需标合同 | 没有 score frontier |
| oos_precision_recall_decomposition_heatmaps.png | S2C vs ADB | 3 dataset × 3 KIR；5 seeds | CLINC/Banking Recall 增而 Precision 降；SO 高 KIR Precision 增 | OOS F1 是 Precision 还是 Recall 驱动 | CLINC/Banking 偏 Recall，SO 偏 Precision | 仍可能只是阈值点 | 中 | 是 | 缺 matched recall |
| state_transition_heatmap.png | S2C vs ADB | StackOverflow/.50；3 seeds | 同样本状态的双向转移；S2C 修正一部分 ADB FA，也恢复一部分 Known | S2C 正确、ADB 错误的是哪类状态 | coverage/error redistribution，不是单向更严 | 只有计数，没有文本/intent | 中偏强 | 补充材料 | 未公开 sample text/score |
| intent_error_budget_scatter.png | S2C vs ADB | intent-level，多 dataset/KIR | intent 点分散在 Known rejection 与 OOS acceptance 变化空间 | 是否由少数 intent 驱动 | 差异是跨 intent 分布的 | 仍不能判多模态 | 弱-中 | 否/补充 | 没有 intent 语义难度 |
| score_gap_false_accept.png | Frozen vs Trainable S2C | 3 dataset；KIR=.50 | Trainable score gap 更大、FA 更低 | 当前 S2C 的 ranking 是否改善 | representation separation | detector/boundary 也改变 | 中 | 可作机制图 | 非外部方法对照 |
| paired_delta_heatmaps.png | S2C vs MOGB-Fair | 3 dataset × 3 KIR；5 seeds | S2C Known Recall/F1-All/OOS Precision 高，OOS Recall 低、FA 高 | MOGB 的低 FA 是否只是好拒绝 | MOGB local under-coverage | MOGB 实现/组件差异 | 强 | 是，标 fair component | 非完整论文 MOGB |
| error_budget_arrows.png | MOGB-Fair → S2C | 3 dataset × 3 KIR | S2C 恢复 Known，同时 OOS FA 上升 | Known recovery 的代价是什么 | coverage–open-space transfer | 默认 threshold 不同 | 强 | 是 | 没有文本级样本 |
| matched_known_recall_frontier.png | S2C vs MOGB-Fair | 3 dataset；多工作点 | 同 Known Recall 下 S2C OOS F1 曲线在上 | 是否只是默认阈值 | frontier/ranking advantage | score calibration 仍可能不同 | 强 | 是 | 只覆盖 fair component |
| threshold_free_delta_heatmap.png | S2C vs MOGB-Fair | 3 dataset × 3 KIR；5 seeds | AUROC/AUPR/score separation 差值为正 | 阈值无关排序是否更好 | ranking/representation | metric distribution 非 causal | 强 | 是 | 不能拆 representation/boundary |
| default_vs_oracle_oos_f1.png | S2C vs MOGB-Fair | 3 dataset × 3 KIR | MOGB oracle 仍低于 S2C；MOGB default-to-oracle gap 更大 | 阈值能否解释差距 | default boundary 不是全部原因 | oracle 受 evaluator 定义影响 | 强 | 是 | 仍非 full MOGB |
| mogb_risk_workpoints.png | MOGB-Fair 及 boundary swap | 3 dataset/KIR | MOGB FA 低但 FR 高；boundary swap 提高 coverage 并引入 FA | local under-coverage 与 open-space cost | radius/ball workpoint trade-off | radius 不是唯一缺陷 | 中 | 补充 | 缺球级语义 |
| top10pct_ball_error_concentration.png | MOGB ball risk | 3 dataset × 3 KIR | 少数高风险 balls 集中大部分 FA | open-space risk 是否集中 | structural risk concentration | support/tiny-ball 混合因素 | 中 | 否/补充 | 不能说明因果方向 |
| current_protocol_tradeoff.png | S2C vs DA-ADB | StackOverflow/.50；3 seeds | DA-ADB 三点同时有更高 FA、更低 OOS F1 | DA-ADB 是否只是更严 | 否定“更严阈值”简单解释 | BERT/训练/contract 混杂 | 中 | 是，标 external | 无 frontier/transition |
| external_supervision_reference.png | S2C/ADB/DCLOOS | 不同 KIR、seed、backbone、监督 | DCLOOS 条件含 pseudo/external OOS | 为什么不能直接排名 | supervision contract difference | 不能说明 external coverage | 强（合同） | 是，作为限制图 | 不是机制图 |
| historical_oos_f1_heatmap.png | Historical Ours vs historical baselines | fulltex；3 dataset × 3 KIR | 历史 Ours 在旧表内的 OOS F1 格子 | 历史结果差距范围 | historical Cascade 旧合同优势 | 不能迁移到当前 S2C | 强（历史事实） | 仅对照 | 无当前同合同控制 |
| trainable_vs_fulltex_reference.png | Current S2C vs historical Ours | 3 dataset × 3 KIR；合同不匹配 | Banking 高 KIR 差距最大，SO 差距小 | 差距发生在哪里 | dataset/KIR localization | representation/boundary/Cascade 都可能 | 中 | 否/补充 | 不能解释原因 |
| kir050_tradeoff_and_errors.png | Gate vs native MSP/Energy/kNN/LOF | Trainable representation；KIR=.50 | Gate 在 FA/OOS F1 上优于 native detectors | Gate 是否有独立贡献 | representation + geometric gate | 不是完整外部 KNNCL | 中 | 可作控制图 | 缺 full KNNCL |

## 现有可视化能够回答什么

| 研究问题 | 当前是否有足够可视化证据 | 当前结论 | 还缺什么 |
|---|---|---|---|
| 为什么 S2C > ADB | 部分足够 | CLINC/Banking 更像 OOS Recall 换 Known coverage；SO 更像 coverage/ranking 重新分配；不能做同骨干因果 | same-sample score scatter、matched Known Recall frontier |
| 为什么 S2C > DA-ADB | 不足 | 当前 DA-ADB 工作点明显更差，且不是简单更严格阈值；算法原因无法判断 | 同骨干/同 split score frontier、paired transition |
| 为什么 S2C 与 MOGB 不同 | 对 MOGB-Fair 组件足够 | MOGB local under-coverage；S2C 有更好的 ranking/frontier，并以有限 OOS Recall/FA 代价恢复 Known | 完整 BERT MOGB 同合同多 seed |
| 为什么 S2C 与 KNNCL 不同 | 不足 | 只有历史 KNNCL 和 native kNN control；没有当前 full KNNCL 机制证据 | current KNNCL final metrics、intent multimodality 分层 |
| 为什么 S2C 与 DCLOOS 不同 | 不足 | 只能确认 supervision/backbone/contract 不同；不能说明 external OOS 覆盖了哪些测试样本 | external/test semantic similarity × OOS Recall |
| 为什么当前 S2C 低于历史 Ours | 不足 | Banking77/.75 的下降已定位并必须承认，但 representation/boundary/Cascade/protocol 未拆开 | historical/current 2×2 representation-boundary + Cascade control |
| 为什么高 KIR 退化 | 部分足够 | CLINC/Banking 的 OOS Precision/FA 成为瓶颈，SO 机制不同；不是普遍规律 | difficulty bins、near-OOS 分层、matched operating curves |

## 现有可视化还不能回答什么

| 未回答的问题 | 为什么当前图不够 |
|---|---|
| S2C 正确、ADB 错误的具体文本/intent 是什么 | state transition 只有 aggregate counts，且没有公开原始文本 |
| S2C 与 DA-ADB 的优势是否由少数 OOS 样本或某类语义组成 | 没有 DA-ADB 同样本 error transition |
| KNNCL 是否在多中心/多模态 intent 上真正占优 | 没有 full KNNCL 当前协议结果与 intent multimodality 指标 |
| DCLOOS 是否主要覆盖与外部 SQuAD 相似的测试 OOS | 没有 external-to-test similarity 或 matched OOS recall |
| 历史 Ours 的优势来自 representation 还是 historical boundary/Cascade | 缺少 2×2 控制，当前 reference 图只能定位差距 |
| MOGB 的每个球为何吸收某些 OOS | ball risk 图有 support/radius/风险，但没有文本语义和严格结构因果 |
| 高 KIR 退化究竟是 distribution shift、score calibration 还是 near-OOS 比例变化 | 当前图有 KIR 工作点，但没有 difficulty-bin 和 per-sample score |

目前最大的分析缺口是：跨外部方法的完整同一样本、同分数语义、带原始 intent/text 的 error set。现有 aggregate transition 已经能支持错误预算判断，但还不能回答“哪一批样本”。

## 下一步最值得补的 6 张机制图

### 1. ADB/DA-ADB same-sample score scatter + matched Known Recall

- 需要数据：同一 protocol view 下 S2C、ADB、DA-ADB 的 per-sample score、true label、predicted_oos、seed；如果 score 不能直接比较，先做每个方法内部 percentile normalization。
- x 轴：外部方法的 normalized OOS score 或 percentile。
- y 轴：S2C 的 normalized OOS score；点颜色为 OOS/known，形状为两方法共同正确、S2C-only、external-only、共同错误。
- 方法：S2C、ADB、DA-ADB；优先 StackOverflow/.50，再扩展 CLINC/Banking。
- H1 成立时：在 matched Known Recall 下，S2C-only 正确 OOS 集中在外部方法的低分/混淆区，且 score ordering 有系统性分离。
- H2 成立时：两方法排序高度一致，只是阈值线不同；把阈值移动到相同 Known Recall 后 OOS F1 差距消失。
- 解释：前者支持 ranking/representation；后者支持 operating point。若 DA-ADB 没有完整分数，则该方法点保持缺失，不用中间预测替代。

### 2. Cross-method Error UpSet / transition matrix

- 需要数据：同一 sample_id 对齐的 S2C、ADB、DA-ADB、MOGB-Fair、KNNCL、DCLOOS final predictions；每行带 dataset/KIR/seed 和五态标签。
- x 轴：错误集合交集，例如 only-S2C-error、only-ADB-error、S2C+ADB-error、all-method-error。
- y 轴：样本数或 OOS F1 贡献，Known 与 OOS 分面。
- 方法：所有能通过最终 metrics 和 sample alignment 审计的方法；blocked 方法不画。
- H1 成立时：S2C-only-error 少，且 S2C-only-correct 集中在特定 difficulty/intent 区域。
- H2 成立时：方法错误集合高度重叠，差异主要是同一批样本的 threshold state flip。
- 解释：这是回答“是哪批样本”的最直接图；当前仓库还没有完整版本。

### 3. OOS difficulty bins × method gap

- 需要数据：每个测试 OOS 的 nearest-known distance、nearest-second-centroid margin、class score、是否近邻/局部密度；外部方法还需能对齐到同一 sample。
- x 轴：按 nearest-known distance 或 margin 的 quantile bins，从 Near-OOS 到 Far-OOS。
- y 轴：OOS Precision、OOS Recall、False Acceptance、OOS F1，或 S2C−baseline 差值。
- 方法：S2C、ADB、DA-ADB、MOGB-Fair、KNNCL（若 final metrics 可用）。
- H1 成立时：S2C 优势集中在 near-OOS，且随 difficulty 呈系统曲线。
- H2 成立时：各 bin 的 score ordering 相近，差异只在整体 threshold/workpoint。
- 解释：把“更强 rejection”具体化为哪类 OOS，而不是只看平均 OOS F1。

### 4. Intent multimodality × S2C–KNNCL gap

- 需要数据：每个 Known intent 的 embedding cluster count、silhouette、centroid-to-within variance、local density entropy；同一 intent 的 S2C/KNNCL OOS F1 或 error rate。
- x 轴：intent multimodality 指标，例如 cluster count 或 within/between variance ratio。
- y 轴：S2C−KNNCL 的 OOS F1、Known Recall 或 FA。
- 方法：当前 full KNNCL 与 S2C；不能用 native kNN 代替。
- H1 成立时：S2C 优势随紧致单峰结构增大，在高 multimodality 区域缩小或反转。
- H2 成立时：gap 与 multimodality 无关，主要由整体 calibration/threshold 决定。
- 解释：这是验证 centroid Gate 与 local-density 方法机制差异的必要图；目前完全缺失。

### 5. MOGB structural gain vs open-space cost

- 需要数据：每个 ball 的 support、radius、selected class、purity、OOS assignment、Known false rejection、S2C rescue 状态和 dataset/KIR/seed。
- x 轴：Known coverage gain 或 ball radius/support；可分面 tiny/large support。
- y 轴：新增 OOS False Acceptance、OOS F1 change 或 Known-correct gain。
- 方法：MOGB-Fair、MOGB partition + S2C boundary、S2C；固定 K=2 作为独立 union-risk reference。
- H1（local under-coverage）成立时：扩大/恢复少数保守球会带来明显 Known recovery，且 cost 由少数结构性球承担。
- H2（纯 threshold）成立时：所有 balls/数据点沿相同单调曲线移动，结构属性不再解释 residual。
- 解释：现有 ball risk 图已接近，但需要把“rescue gain”和“新增 open-space cost”放在同一 ball/cell 坐标上。

### 6. Historical/current 2×2 representation–boundary causal decomposition

- 需要数据：同一 historical split、Known list、checkpoint/evaluator 下的 Frozen representation、Trainable representation、Historical boundary、Current K=1 boundary 四组合；再加有/无 Router/Expert Cascade 的 control。
- x 轴：四个 representation × boundary cell。
- y 轴：OOS F1，同时显示 OOS Precision/Recall、Known Recall、FA。
- 方法：historical Ours components、current S2C components；只在合同完全闭合后绘制。
- H1（representation 主导）成立时：换 representation 的两条边同时显著改变 OOS F1，换 boundary 影响较小。
- H2（boundary 主导）成立时：同一 representation 换 boundary 就产生主要差距。
- H3（Cascade/protocol 主导）成立时：Gate-only 2×2 解释不了差距，加入 Router/Expert 后差距才出现。
- 解释：这是回答当前 S2C 为什么低于历史 Ours 的最小充分实验，不应再用单独的 OOS/Accuracy bar chart 代替。

## 最终判断

现有图已经给出一个可写进论文的、但有边界的机制故事：

- 对当前 Known-only MiniLM fair component，S2C-Trainable-K1 的优势不是单纯阈值；它有更好的 score ranking，并以 K=1 控制 acceptance union。
- 对 ADB，S2C 在 CLINC150/Banking77 主要赢在 OOS Recall/FA trade-off，在 StackOverflow 更像 coverage recovery 与 error redistribution；外部 BERT/MiniLM 合同使因果结论保持中等强度。
- 对 MOGB-Fair，现有证据已较强支持“local under-coverage + ranking/frontier 差异”，并明确区别于固定多中心的 open-space over-coverage。
- 对 DA-ADB、KNNCL、DCLOOS，当前主要是合同/证据缺口，不能用最终分数反推机制。
- 当前 S2C 低于历史 Ours 的 Banking77/KIR=.75 下降是真问题，现有图尚未解释原因；2×2 representation–boundary–Cascade 拆解是最高优先级。
