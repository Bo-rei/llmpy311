# DEVELOPMENT_LOG

此文件是 s2c 的追加式开发记录。不要改写历史条目；发现错误时新增
`Correction` 条目说明修正内容与原因。

## 2026-07-22 — protocol_v2 数据独立化启动

- Base commit: `fcd9df5249a7e5388080277795c81afa49ed6f7d` (`main`)。
- TEXTOIR snapshot: `dffe2b1b848a069a6808f8089b4cb9bd16e2062b`，工作树干净。
- 开始前已有、且与本轮直接相关的未跟踪内容：
  `results/data_audit/` 与 `tools/maintenance/audit_dataset_provenance.py`。
- 数据影响：尚未复制、清洗、去重或删除任何数据；`artifacts/s2c` 保持原样。
- 本轮目标：把 TEXTOIR 的三个固定数据快照复制为 `s2c/data` 的独立来源，
  从该来源构建 protocol_v2，并禁止实验运行时读取 `textoir/data`。
- 预检：父仓库无已跟踪未提交改动；`textoir` 工作树干净；
  `artifacts/s2c` 当前约 27GB；尚未运行训练或 embedding。
- 风险：旧 v19/历史结果的数据协议不得与 protocol_v2 混合；任何正式模型实验
  必须等待数据、registry、views、export 与运行时独立性门禁全部通过。
- 下一步：迁移完整审计明细至 artifacts，仅保留轻量公共裁决，然后建立独立数据层。

## 2026-07-22 — protocol_v2 数据层、导出与 Gate 基础设施

- Base commit: `fcd9df5249a7e5388080277795c81afa49ed6f7d`；TEXTOIR snapshot:
  `dffe2b1b848a069a6808f8089b4cb9bd16e2062b`。
- 修改范围：新增 `src/s2c/` 的 runtime、data、evaluation、tracking 与 fixed-boundary Gate
  experiment 核心；新增稳定 data/experiment 薄脚本、声明式 protocol_v2 配置、测试、Makefile
  和数据/迁移/结果契约文档；未改动 legacy v19 运行时路径。
- 数据影响：逐文件字节复制 TEXTOIR `oos`、`banking`、`stackoverflow` 到被 Git 忽略的
  `data/sources/textoir/<commit>/`；构建 canonical（CLINC150 23,700、Banking77 13,083、
  StackOverflow 20,000 行），未改文本、标签、split 或重复行。构建 429 个固定 registry；
  fixed views 正在物化，完成前不启动模型实验。
- 审计迁移：完整逐样本来源审计移至
  `../artifacts/s2c/reports/data_provenance_audit/2026-07-22_textoir_dffe2b1b/`；Git 内仅保留
  `docs/audits/data_provenance/` 的轻量裁决。没有删除历史 artifact 或 v19 结果。
- 运行时独立性：临时将 `../textoir` 重命名为 `textoir.disabled` 后，protocol validation、
  s2c/TEXTOIR-format export 与 36-cell Gate dry-run 均通过；随后恢复目录，并确认 TEXTOIR
  工作树仍干净。
- 已验证：editable installation、`compileall`、15 个 protocol_v2 unit/integration 测试、
  `check_data_tracking.py` 与 `check_development_log.py` 通过；smoke exports 已为三个数据集、
  KIR 0.25/0.50/0.75、seed 42 建立。
- 风险：dense Gate 尚未开始；边界、表示、外部 baseline 与完整 pipeline 仅有配置和 exporter
  接口，不能写成完成的实验。正式 E1 必须等待全量 views/exports、完整测试和协议核验全部通过。
- 下一步：完成 429 views 与统一 exports，复核运行时独立性，然后执行 36-cell Gate smoke。

## 2026-07-22 — protocol_v2 E4 外部 Baseline 可运行性适配

- Base commit: `fcd9df5249a7e5388080277795c81afa49ed6f7d`；本条只新增
  `src/s2c/experiments/external_baselines.py`、稳定薄脚本、E4 配置字段、最小测试与运行文档，
  不修改 E1/E2/E3 Gate runner、历史 v19 路径或 TEXTOIR 工作树。
- 方法边界：`MSP`、`Energy`、`kNN`、`LOF` 只在本地可用的
  `frozen MiniLM + scikit-learn` 依赖满足时作为 Known-only Gate control 执行；训练不含 OOS，
  阈值由 Known calibration 的有限样本上分位数选择，测试集不参与选择。
- 外部方法边界：`DOC`、`ADB`、`DA-ADB` 只接受 protocol_v2 的 TEXTOIR-format export；
  `MOGB` 只接受 MOGB-format export；未建立单独、审计过的上游环境与预测导入器时写
  `blocked` manifest。`(K+1)-way` 在当前 Known-only train/dev 协议下写 `unsupported`，
  不引入真实或合成 OOS 训练样本，也不伪造性能数字。
- 数据与 artifact 影响：adapter 在运行前核对 export manifest 与 registry SHA256；实际 E4
  结果仅会写入新的 `../artifacts/s2c/runs/protocol_v2/external_baselines/`，不会读取
  `textoir/data`、不会覆盖 E1–E3 或历史 artifact。本条实现期间未运行模型训练或大规模实验。
- 测试：新增 synthetic fixed-split test，覆盖矩阵解析、方法状态、s2c export 绑定、native MSP
  的 Known-only threshold/resume，以及 ADB blocked manifest。后续在 E3 资源空闲后可运行
  一个显式的 CLINC150/KIR=0.50/seed=0 native smoke。
- 风险与下一步：upstream DOC/ADB/DA-ADB/MOGB 不应因 export 存在而被视为已经复现；如需运行，
  先建立隔离环境、固定版本和 predictions importer，再以相同 registry 做 smoke。

## 2026-07-22 — Correction：三方数据来源裁决收紧并冻结候选实验

- 触发原因：新的来源裁决要求同时核验官方/原始来源、固定 TEXTOIR commit 与实际被历史 s2c
  实验读取的 `assets/datasets/s2c/prepared/data/multidataset/v19`；不能仅凭样本总数或规范化
  文本相同宣布数据一致。
- 审计：`tools/audit/audit_dataset_provenance.py` 现在将原始文本-标签精确匹配与 NFKC/大小写/空白
  规范化匹配分开统计，并输出缺失、额外、标签、split、许可证和历史输入引用证据。完整结果写入
  `../artifacts/s2c/reports/data_provenance_audit/2026-07-22_three_way_verification/`，未修改原始数据。
- 裁决：CLINC150 与 Banking77 都是 `reconstructed_from_official`，分别需从官方
  `data_full.json` 和 Banking raw train/test 重建；Banking 的 TEXTOIR snapshot 只有
  13,080/13,083 原文精确匹配，不能作为 raw canonical source。StackOverflow 与历史
  BANKING77-OOS 都是 `blocked_unverified`；前者缺少可核验的上游数据许可证，后者没有可追溯的
  官方 OOS 扩展来源。
- 影响：已写入的 `s2c/data/`、registry、views、E1/E2/E3 候选 run 保留为不可覆盖的审计历史，
  不得再称为正式 protocol_v2 结果。`configs/data/protocol_v2_admission.json` 现为 `blocked`；
  Gate runner 和外部 baseline 的 `--execute` 会 fail closed，干跑仍可用于输入结构检查。
- 验证：审计未启动训练、embedding、MOGB/DCL 或 TEXTOIR 方法；完成前不得恢复候选 sweep。下一步
  仅限于按官方 raw source 重建 CLINC150/Banking77，再生成新的 dataset_version、registry、views
  和独立 run root；StackOverflow 保持排除状态直至许可证证明完备。

## 2026-07-22 — 官方 raw 重建版本与按数据集准入

- 新版本：新增 protocol_v2_official_v1，所有 canonical、registry、view、export、run 和 embedding
  cache 路径均由 dataset_version 隔离；默认 protocol_v2 路径保持原候选数据不变。
- 来源：CLINC150 从 clinc/oos-eval 固定 commit 的 data_full.json 导入；Banking77 从 PolyAI 固定
  commit 的 train.csv/test.csv 导入。导入逐文件 SHA256 校验、拒绝 symlink/hardlink，并将许可证
  SHA、上游 commit 和 source format 写入 versioned manifest；不再把 TEXTOIR 当 raw source。
- split：CLINC 保留 train/val/test/oos_train/oos_val/oos_test 的原始名称，并另存 view_role。
  Banking 保留官方 train/test；由于官方没有 dev，以 class-stratified SHA256 rank 从 raw train
  独立选择 1,000 条 calibration，sample-id 列表与 derivation hash 可复算且不等同于 TextOIR dev。
- 物化：已生成两数据集 canonical、全部正式 KIR/seed registry，以及 seed=0/KIR=0.50 的 view/export
  验证样本；其余 view/export 按需生成。未运行 encoder、embedding、训练、Gate、MOGB/DCL 或任何
  新实验。
- 准入：admission 改为 partially_admitted，仅 protocol_v2_official_v1 下的 CLINC150 与 Banking77
  可在 materialized inputs 存在时执行；StackOverflow、BANKING77-OOS 和 TEXTOIR candidate 继续拒绝。
- 验证：官方导入/解析/派生 split 单元测试、canonical/registry/view/export validation、compileall
  均通过。后续开始正式 run 前仍需按目标 seed/KIR materialize inputs 并重跑完整测试。

## 2026-07-22 — Correction：官方数据不变量与审计回写

- validate_protocol 现在对 source_name=official 额外核对许可证 manifest、CLINC150 的
  23,700/150/1,200 不变量、Banking77 的 13,083/77/0 不变量，以及 1,000 条 calibration
  derivation 的 sample-id 标记一致性；小型 fixture 和被阻断 candidate 不受这些固定计数约束。
- 三方审计增加 official raw 与 protocol_v2_official_v1 canonical 的反向逐样本比较。CLINC150 和
  Banking77 的 exact record、normalized text、label、split、intent-set 与 per-class count
  match rate 均为 1.0；完整 CSV/JSON 仍只写在 artifacts 审计根目录。
- audit_experiment_registry.py 已发现并登记新增的 protocol 核心入口，但在大规模历史文件系统
  扫描阶段出现宿主机 I/O 等待；这不影响数据层验证，后续应在 I/O 空闲时重新完成该只读审计。

## 2026-07-24 — Correction：implementation audit 跟随 dataset version

- 触发原因：旧的轻量 `protocol_v2 implementation report` 固定遍历全部 schema 数据集，且沿用
  TEXTOIR candidate 的叙事；这会把已封锁的 StackOverflow 混入官方重建版本的项目状态说明。
- 修改范围：`tools/audit/generate_protocol_v2_implementation_report.py` 改为只读取当前
  `S2C_DATASET_VERSION` 下同时具备 source/canonical manifest 的数据集，并将 admission 裁决、
  实际 source revision 与数据版本写入报告；增加对应回归测试。
- 数据与 artifact 影响：只重写 Git 可跟踪的轻量 audit CSV/Markdown/JSON；不改 raw source、
  canonical、registry、view、export、embedding、run 或历史 artifact。
- 验证计划：使用 `protocol_v2_official_v1` 重新生成 audit，并运行报告单测、官方数据验证、
  data tracking 与 development log 检查。未运行训练、embedding 或模型实验。

### Correction：audit 磁盘统计保持轻量

- 发现：旧 generator 递归统计整个 `../artifacts/s2c`，会扫描约 27GB 的不可变原始实验输出，
  与“轻量 Git audit”职责冲突，并可能在宿主文件系统 I/O 等待。
- 修正：报告继续统计项目与 data 树，但对 artifacts 记录 `not_scanned`，明确不以不完整数值冒充
  磁盘测量。原始 artifacts 未修改、未删除，且这不是实验执行。

### 验证结果

- 通过：官方 source/canonical/registry/view/export 的定向 pytest（9 passed）、报告 generator
  回归测试、Ruff、compile、官方版本 `validate_protocol`、data tracking 与 development-log 检查。
- 未完成：`pytest -q tests/unit` 在 90 秒内没有输出，已安全中断；这与先前全量历史文件系统扫描的
  宿主 I/O 等待一致。不得将本条写成全量 unit suite 已通过。

## 2026-07-24 — Correction：protocol Gate runner 的 src-layout 导入

- 触发原因：官方 CLINC150 E1 预运行完成 MiniLM 加载后，四个单元均因
  `ModuleNotFoundError: src` 安全失败。原因是新 `s2c` 包从 `src/` 安装时，runner 仍使用
  checkout-only 的 `src.gate...` 模块路径。
- 修改范围：runner 改为导入公开的 top-level `gate` 包；`gate/__init__.py` 改用相对导入。
  新回归测试从没有项目根目录的子进程执行，确保不再依赖隐式 `src` namespace。
- 数据与 artifact 影响：此前四个 run 只有 failed state，未生成 metrics、predictions 或 checkpoint；
  本修复不改 canonical、registry、view、export 或历史 artifacts。随后将用 `--resume` 重试同一
  immutable run id，成功时只会写此前不存在的完整 run 目录。
- 验证计划：先运行新的 src-layout import 测试与定向 Gate 测试，再重试 CLINC150/KIR=.50/seed=42。

### Correction：verify/summarize 与 runner 使用同一分片选择

- 发现：runner 支持 dataset/seed/KIR 分片，但 verify/summarize 只能读取完整 YAML；对已准入的
  official 两数据集 E1 子矩阵会错误报告候选 config 中尚未启动的 StackOverflow 单元为缺失。
- 修正：新增 `filter_gate_specs`，并让 runner、verify、summarize 共用它；verify/summarize 也接受
  `--dataset`、`--seed`、`--kir`，summarize 的计划文件可用 `--shard-name` 隔离。
- 数据与 artifact 影响：不改变任何 run id、模型、数据或已有输出；仅允许对明确选择的已准入矩阵
  作一致的检查与轻量汇总。

### 官方 E1 Gate-only 子矩阵执行

- 前置：CLINC150、Banking77 的 official raw canonical/registry/view/export 在
  `protocol_v2_official_v1` 下通过验证。TextOIR 临时重命名后，官方数据验证与 CLINC Gate dry-run
  仍只读取 `s2c/data`；随后已恢复 TextOIR，commit 仍为 `dffe2b1…` 且工作树干净。
- 执行：运行 CLINC150、Banking77 × KIR `{.25,.50,.75}` × seed 42 × K `{1,2}` ×
  `{euclidean, mahalanobis_diag}` 的 frozen-MiniLM fixed-boundary Gate。24/24 完成、0 failed、
  verification 为 complete=24/missing=0/invalid=0，primary metrics 均为有限值。
- 数据与 artifact 影响：写入新的 `../artifacts/s2c/runs/protocol_v2_official_v1/` run、cache、
  plan/state 和 summary；没有覆盖 historical/candidate artifact，也没有训练表示、Router 或 Expert。
  StackOverflow、BANKING77-OOS、MOGB/DCL/ADB/DA-ADB 与完整 Cascade 均未运行。
- 公开证据：将 `official_e1_admitted.csv` 加入 `public_results.yaml` 白名单，导出后由
  `results/MANIFEST.csv` 记录其 SHA256。该单 seed Gate-only smoke 不是论文主表或完整系统结论。
- 下一步：先完成文档/公开快照验证；只有在 StackOverflow 来源许可证得到独立核验后，才可讨论完整
  三数据集 E1 或后续 dense/boundary/representation 任务。

### Correction：公开快照与官方 E1 结论收口

- 发现：`results/protocol_v2/` 中保留了未跟踪的候选协议汇总，既不在现行公开白名单，也会使结果
  校验将来源已冻结的 candidate 误当作活动 GitHub 证据。
- 修正：候选快照完整保留到
  `docs/archive/historical_repro_bundle/protocol_v2_candidate_results/`，活动 `results/` 只保留
  `configs/public_results.yaml` 白名单文件。`results` 导出/校验现为 48 个轻量 CSV/JSON、905,687 bytes，
  全部 SHA256 一致；没有移动或改写任何 `../artifacts` 文件。
- 审计文案：implementation report 不再要求一个与官方 admission 不兼容的三数据集 36-cell E1。
  它明确记录现有 24 个 completed Gate run 仅覆盖 admitted 的 CLINC150/Banking77；StackOverflow 和
  BANKING77-OOS 继续是 blocked 数据，不能被旧 candidate 结果补齐。
- 验证：公开结果 `--verify` 通过；随后重新生成 version-aware implementation audit。该步骤未运行
  训练、表示适配、Router/Expert、MOGB/DCL 或任何新的实验单元。

### Correction：implementation audit 的 embedding 与执行 provenance

- 触发原因：官方 E1 的 completed run manifest 已记录 frozen MiniLM embedding cache，但旧
  `audit_manifest.json` 将 `embedding_generation` 固定写为 `false`，同时缺少固定 TEXTOIR commit、
  执行命令和公开 candidate 快照的归档移动记录。这会低估已发生的数据处理，也不满足审计所需的
  provenance 粒度。
- 修改范围：`tools/audit/generate_protocol_v2_implementation_report.py` 现在从 completed run
  manifest 推导 embedding cache 使用情况，记录生成起止时间、TextOIR commit（仅审计参考，非运行时
  依赖）、执行命令、移动文件、工作树已跟踪修改以及 artifact 写入状态；报告新增 materialized
  canonical inventory。对应回归测试覆盖 blocked StackOverflow 文案和新 manifest 字段。
- 数据与 artifact 影响：只重写轻量 audit CSV/JSON/Markdown；已确认 `training_run=false`、
  `embedding_generation=true`、`artifacts_deleted=false`。没有复制、删除或改写 source/canonical/view/
  export/历史 artifact，也没有启动新模型实验。
- 验证：定向 protocol/runner/public-results pytest、Ruff、py_compile、官方 views/exports 验证、
  24-cell Gate verify、公开结果 SHA256 verify 与 `git diff --check` 已重跑；完整 unit suite 仍因宿主
  文件系统 I/O 等待而不在本条声称通过。

### Correction：将三数据集协议停止条件独立成审计证据

- 触发原因：implementation report 已提到 StackOverflow blocked，但原目标同时要求发生停止条件时生成
  blocker report。若只在正文段落中保留该限制，后续容易将二数据集 Gate 子矩阵误解为原定 E1/E2 的
  可扩展完成状态。
- 修改范围：audit generator 现在生成
  `docs/audits/protocol_v2_implementation/blocker_report.md`。它列出 admitted/blocked dataset、
  被影响的 36-cell E1 与 3,300-cell E2、禁止的替代来源，以及解除 StackOverflow 阻断所需的 source/
  license/20-label/20,000-row/三方比较证据。
- 数据与 artifact 影响：仅新增轻量审计 Markdown；不改变 admission、原始数据、embedding、run、
  历史 artifact 或公开结果。该报告不是新的实验结果。
- 验证：generator 回归测试、Ruff 与 py_compile 通过；重新生成 audit manifest，blocker report 已纳入
  SHA256 generated-files 列表。完整 unit suite 的宿主 I/O 限制仍保持如实记录。

### Correction：StackOverflow 来源链复核与有界历史审计刷新

- 触发原因：StackOverflow 的 blocked 状态需要可复查的正向溯源证据，而不应只写“许可证不清楚”。
  固定 upstream commit 的 README 已复核：它声明 20,000 titles/20 labels、要求致谢 Kaggle，但未给出
  Kaggle dataset、data-dump revision、post metadata 或许可证；固定 revision 的 `LICENSE` endpoint 返回
  HTTP 404。Stack Overflow 的通用许可随原帖日期变化，而当前语料没有 post ID/date，因此不能逐条赋予
  可验证许可。
- 修改范围：新增 `docs/audits/data_provenance/stackoverflow/source_trace.md`，并将 README SHA256、
  LICENSE 缺失状态与 license assessment 写入 public/full `dataset_decision.json` 及 full
  `source_license_report.csv`。`audit_dataset_provenance.py` 同步固化这项证据，新增
  `--reuse-historical-audit`：source/license 刷新复用已存在的 historical prepared-input inventory，
  不再递归遍历大型 immutable artifact 树。
- 数据与 artifact 影响：完整三方审计使用 reuse 模式重新生成，决定仍为 CLINC150/Banking77
  `reconstructed_from_official`、StackOverflow/BANKING77-OOS `blocked_unverified`；只更新 audit
  CSV/JSON，不修改数据、模型、embedding、run 或历史实验输出。
- 验证与风险：静态检查、编译与一次 reuse audit 成功；新增 reuse fail-closed 单测。该单测的后续
  pytest 复跑与全量 pytest 均可因宿主文件系统 I/O 进入 D 状态，已终止卡住进程；因此不把这部分写成
  全量 suite 通过。未来解除阻断仍必须提供 immutable raw source、20-label/20,000-row mapping、逐条
  post metadata 与许可证链，不能以 TEXTOIR 或 Kaggle 名称替代。

### Correction：官方 implementation audit 固定 dataset version 并增加需求矩阵

- 触发原因：`ProtocolV2Paths` 为兼容历史 candidate，默认版本仍是 `protocol_v2`。一次未显式传入
  dataset version 的 audit 命令会读取 candidate 的 4,584 个历史运行，从而错误地把它们展示在
  official audit 中；该报告已立即使用 `protocol_v2_official_v1` 重建，未据此启动实验或修改 artifact。
- 修改范围：implementation-audit CLI 新增 `--dataset-version`，且默认安全指向
  `protocol_v2_official_v1`；candidate 只有在调用方显式指定时才会被审计。新增
  `requirement_matrix.csv`，逐项区分已完成的 CLINC150/Banking77 官方范围、按需物化的数据视图、
  24 个 admitted-scope Gate run，与因 StackOverflow/BANKING77-OOS 阻断而不得宣称完成的原三数据集
  E1、E2、表示/基线/Cascade 矩阵。
- 数据与 artifact 影响：仅重写轻量 audit CSV/JSON/Markdown，并新增 requirement matrix；没有修改
  source/canonical/registry/view/export/embedding/run、历史实验输出或 TEXTOIR。`audit_manifest.json`
  明确记录 `protocol_v2_official_v1`、2 个 materialized dataset、24 个 completed Gate-only run 和
  `embedding_generation=true`。
- 验证：`python -m py_compile ...`、`ruff check ...` 与
  `pytest -q tests/unit/data/test_protocol_implementation_report.py` 通过（5 passed）。完整 suite 仍不在
  本条声称通过，原因是先前观察到宿主文件系统 I/O 可阻塞全量收集。

### Correction：公开三方裁决快照与完整审计保持同一字段语义

- 触发原因：轻量 `docs/audits/data_provenance/summary.csv` 与单数据集 decision JSON 是较早生成的
  快照；它把官方→official reconstruction 的 `split_match` 与 TEXTOIR/历史 split 差异混在一起，
  因而会与完整 audit 的当前字段产生表面矛盾。
- 修改范围：公开 decision JSON 现在镜像完整 audit 的字段（StackOverflow 保留额外的 public
  `source_trace.md` 引用）；`summary.csv` 明确改为
  `official_reconstruction_split_match_rate`，新增轻量 `source_license_report.csv`，README 说明该
  split 指标不代表 TEXTOIR 或历史 prepared split 一致。public manifest 同步列出这些证据文件。
- 数据与 artifact 影响：只更新 Git 可读的裁决快照和说明，未改动完整逐样本比较、canonical 数据、
  registry/view/export、embedding、run 或历史 artifacts；四项裁决不变。
- 验证：使用 `csv.DictReader` 校验两个公开 CSV，逐项比较 public decision 与完整 audit decision
  （除 public StackOverflow trace 指针外完全一致），并重新通过 development-log 检查与
  `git diff --check`。

### Correction：ProtocolV2Paths 默认指向已准入官方数据版本

- 触发原因：尽管 `require_experiment_admission()` 会拒绝 legacy `protocol_v2` candidate，路径解析器
  本身仍将它作为默认版本；普通 dry-run、plan 或 audit 调用因此可能先读取不应作为正式依据的候选
  data tree。
- 修改范围：`ProtocolV2Paths` 的 dataclass 与环境变量 fallback 现在默认
  `protocol_v2_official_v1`。历史 candidate 仍完整保留，但只能通过显式
  `S2C_DATASET_VERSION=protocol_v2` 选择；对应 admission test 同时锁定“默认 official 可用、显式
  candidate 被拒绝”的行为。
- 数据与 artifact 影响：不移动、不删除、不重写任一 data/artifact。执行一次 CLINC150/Banking77、
  KIR=.50、seed=42 的 8-cell dry-run，仅写可再生 plan/state 元数据；所有 required input 均来自
  `data/exports/protocol_v2_official_v1`，`uses_textoir_data=false`，未载入模型或生成 embedding。
- 验证：`py_compile`、Ruff 以及 admission/runner-import/implementation-report 定向 pytest 通过
  （9 passed）；实际 bare CLI 默认 scope 打印为 `protocol_v2_official_v1`。
