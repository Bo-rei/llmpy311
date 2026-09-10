# S2C 统一对比实验补充执行计划 V1

更新时间：2026-08-11
前置审计：[`PLAN_ALIGNMENT_AUDIT_V1.md`](PLAN_ALIGNMENT_AUDIT_V1.md)

## 目标和边界

本计划只补足附件要求与当前证据之间的缺口，不增加元学习、自适应 K、椭球、新 loss 或复杂
多中心聚合；不重跑已闭合 E0–E3、Trainable/Frozen/MOGB-Fair 主矩阵。`S2C-Trainable-K1`
继续是唯一自有主方法：Known-only MiniLM 适配、K=1 单中心 Gate。

所有新增工作都必须落在四个合同层之一：

1. `same-protocol fair`：同 split、Known list、registry、seed、evaluator 和 calibration-only 选择；
2. `native_backbone_control`：仍可比较，但明确表示和训练合同；
3. `different_supervision`：显式记录 pseudo-OOS/external-OOS 监督和成本；
4. `historical`：只作历史参照，不能回填当前逐样本结果。

### 合同层命名映射

计划中的语义名称与当前机器可读结果的 `contract_layer` 对应如下；`layered`、
`different_system_level` 和 `mechanism` 是旧图谱的可视化分组/图类型，不是新的公平方法层。

| 计划语义 | 当前机器 label | 用途 |
|---|---|---|
| `same-protocol fair` | `same_protocol_fair` | 当前 MiniLM Known-only 公平主层 |
| `native_backbone_control` | `same_protocol_native_backbone_control` | 同一表示上的 native detector 控制 |
| 外部 BERT/TextOIR 参照 | `external_backbone` | ADB/DA-ADB 等外部 backbone，禁止并入 fair 排名 |
| `different_supervision` | `different_supervision` | DCLOOS 的 pseudo-OOS/external-OOS 监督层 |
| `historical` | `historical`；旧图谱中的 `different_system_level` | fulltex/Cascade 等历史系统级参照 |

## 执行门和依赖顺序

### G0：每次实验前的可执行性门

先运行 `check_research_state.py`，核对 ledger 的 `do_not_repeat`，确认输入 registry/canonical/source
manifest 和 artifact 根没有变化。外部 BERT 方法必须先通过：

```text
import torch → BERT forward → 单批 CPU/GPU smoke → final-metrics 导出 smoke
```

没有通过 G0 不启动训练；保留 `runtime_blocked_no_metrics` 或具体失败状态，不把失败解释为算法性能。

### G1：扩展 P0 合同而非填空

对每个方法建立一份最终状态登记，至少包含：

```text
method, contract_layer, dataset, kir, seed, split, known_list,
source_manifest_sha256, registry_sha256, canonical_manifest_sha256,
final_metrics_available, test_used_for_selection, prediction_path
```

只有 `final_metrics_available=true`、预测有限、sample_id 顺序/集合和 Known labels 审计通过的 cell
才能进入 `prediction_analysis_table`。DA-ADB、DCLOOS 或传统 detector 如果只有 summary、smoke、
timeout 或中间 prediction，继续放在 `blocked_methods.csv`。

### W1：补齐 baseline，先单格后矩阵

按上一版计划的顺序执行，且每一步都经过 G0/G1：

1. StackOverflow / KIR=.50 / seed=42：KNNCL；
2. 同一工作点：先只读审计已有 DA-ADB seed=42 有效 cell 的 final-metrics/逐样本合同；不重复已经完成的训练，只有在确有缺失 adapter 且 G0 通过时才新增 smoke；
3. 同一工作点：OpenMax、DOC、DeepUnk；
4. 同一工作点：DCLOOS fixed-registry converged cell。

每个方法的 smoke 验收包括：完整 final metrics、有限的逐样本预测、11 个标签/正确 Known list、
sample_id 对齐、calibration-only 选择声明和可复现运行 manifest。任何一项失败就停止该方法的
3×3×5 扩展，并更新 `failed_or_blocked_methods.csv`。

只有 W1 单格通过的方法才允许扩展到 3 datasets × 3 KIR × 5 seeds。ADB 现有外部结果不重复；
DA-ADB 只有在跨数据集/多 KIR 的同一外部合同闭合后才扩展。传统 TextOIR 方法的结果始终保留
`native_backbone` 或 `historical` 标签，不与 MiniLM fair 行混排。

### W2：Backbone × Method 控制

先做 StackOverflow / KIR=.50 / seed=`42,87,100`：

| 表示 | S2C-K1 | ADB/对应 detector |
|---|---|---|
| MiniLM | 现有 Trainable-K1 | 只有在 ADB-MiniLM 适配真实可用时运行 |
| BERT | 新增 S2C-BERT-K1 控制 | 现有 ADB BERT 外部参照 |

四个格必须固定 split、Known list、评价器、阈值选择规则和 seed；不增加新的 loss 或边界规则。
如果 ADB-MiniLM 或 S2C-BERT 的实现不支持，记录为 `not_supported/blocked`，不得用不同的
方法替代缺格。

### W3：性能层统一图

对所有通过 W1/W2 的方法重新生成统一源表和 manifest：

- Dataset × KIR OOS F1/F1-All/F1-K/F1-Known Recall 热图；
- OOS F1–Known Recall Pareto；
- Known Recall–false acceptance 工作点曲线；
- KIR 曲线、seed 分布/森林图、AUROC/AUPR 曲线；
- 统一的 mean ± std、95% paired bootstrap CI、win/tie/loss、effect size。

主结果点只取 Known calibration；test OOS 最优阈值只能作为 post-hoc 曲线，并在图例和 manifest
中显式标注。没有 final-metrics 的方法显示为 blocked，不用 0、空值或旧论文数字补齐。

### W4：机制层补足

按“性能现象 → 错误转移 → score → 表示 → 边界”的顺序补图：

1. **Score**：Known/OOS 密度、阈值、AUROC/AUPR、Overlap Coefficient；统一 score 方向和分母。
2. **Geometry**：预注册 `C_y`（类内离散）、`M_y`（最近类间距离）、`R_y=M_y/C_y`，并附
   effective rank、kNN neighborhood preservation、intent-level Spearman；不同 backbone 不做
   无条件 pooled correlation。
3. **Decision-aware UMAP**：只从 hash-registered local embedding artifact 生成，标记
   Correct Known、False Rejected Known、Correct OOS、False Accepted OOS，并叠加 centroid/ball。
   不能取得合法 embedding 时，保留 `blocked_no_embedding_artifact`，不以 score 聚合替代。
4. **Near-OOS**：用 Frozen/reference 或 calibration-only 固定边界定义 Very Near/Near/Medium/Far，
   测试只应用边界；无 validation OOS 时标为 exploratory，不能作为正式成功标准。

### W5：MOGB、DCLOOS 与错误集合

- MOGB：不重跑主矩阵，只把已有 ball count/radius/support/risk、Known rejected/OOS accepted、
  component swap、selected-ball 缺类/tiny-ball、Trainable–MOGB transition 合入一个 manifest。
- DCLOOS：先完成 fixed-registry 的一个 converged cell；再按 `pseudo-only`、`external-only`、
  `pseudo+external` 和可行的 `none/reference` 建 supervision-performance frontier，同时列出
  外部数据、训练时长和资源成本。没有 converged final metrics 不画 frontier。
- Error sets：对 aligned final methods 输出 Frozen→Trainable、MOGB→S2C、ADB→S2C、DCLOOS→S2C
  的五状态 transition、UpSet/集合交集和 intent-level false-acceptance heatmap。DCLOOS 只有在同一
  sample_id/test split 合同成立后才进入 paired transition。

### W6：鲁棒性和成本

只有 W1 核心 baseline 闭合后才执行：

- low-resource：10%、25%、50%、75%、100%，固定 subsampling、Known calibration 比例和 seed；
  优先 S2C-Trainable-K1、Frozen-K1、KNNCL、ADB、MOGB-Fair；
- threshold robustness：统一 threshold grid，输出 threshold/OOS F1/F1-All/Known Recall/FA/FR；
- efficiency：trainable/total parameter count、training time、peak memory、embedding throughput、
  Gate latency、total inference time；缺测写 `missing`，不估算。

### W7：最终 8 图和报告验收

主文八图只有在每张图同时拥有 source CSV、manifest、contract layer、selection audit 和可读图像时
才标记 `complete`：

1. fair heatmap；
2. OOS F1–Known Recall Pareto；
3. Known Recall–FA operating curve；
4. Known/OOS score distribution；
5. geometry + error-aware UMAP；
6. near-OOS difficulty curves；
7. MOGB mechanism dashboard；
8. DCLOOS supervision-performance frontier。

报告固定按“性能现象 → 样本错误转移 → score 分布 → 表示几何 → 边界机制 → 监督差异 → 代价与失败
场景”组织，并在每个结论旁写清 `same-protocol fair`、`native backbone`、`different supervision`
或 `historical`。

## 停止条件

出现以下任一项就停止扩展，不伪造排名：

- final metrics、sample_id、Known list 或 split/hash 不完整；
- test OOS 参与选参、早停、阈值或分桶；
- runtime 无法通过最小 BERT/CUDA smoke；
- MOGB 只有兼容单格或无法提供 ball/center/radius 字段；
- DCLOOS 只有 reduced、timeout、smoke 或中间预测；
- 图没有可审计源 CSV/manifest；
- 方法只能靠不同监督合同比较，却被写成同协议公平结论。

## 完成定义

本补充计划完成不等于“所有实验数字已存在”；它要求：每个附件要求都有当前状态、证据路径、下一步
entry/exit gate，并且未完成项不会进入主结论。只有 W1–W7 的相应验收门全部通过，才能把附件中的
最终假设升级为正式结论；在此之前只保留当前 fair 层的限定性解释。
