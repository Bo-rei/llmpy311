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

### Correction：建立唯一活动的 TEXTOIR-compatible 三数据集协议

- Base commit：`ea210083b331c489059f275edcc2e0c3241cfba7`。
- 触发原因：此前 StackOverflow 的 `blocked_unverified` 将“公开重新分发完整语料”的高标准错误地
  用作“固定 benchmark snapshot 的本地科研使用”前置条件，导致三数据集 TEXTOIR-compatible 实验
  无法启动。该限制已按当前研究范围收缩：本地使用与公开再分发分开管理。
- 修改范围：新增活动 `protocol_v2_textoir_v1`；`ProtocolV2Paths` 默认切换到该版本；准入状态新增
  `admitted_official`、`admitted_benchmark_local_only`、`blocked_content_unverified` 与 `legacy_only`
  的兼容处理。`protocol_v2_official_v1` 保留为 frozen audit，legacy `protocol_v2` 保持拒绝。
- 数据影响：从干净的 TEXTOIR commit `dffe2b1b848a069a6808f8089b4cb9bd16e2062b` 字节复制
  `oos`、`banking`、`stackoverflow` 到 `data/sources/textoir/<commit>/`；不改文本、标签、split 或
  重复行。StackOverflow 保留 20,000 条/20 标签，标记 `local_research_only=true` 与
  `redistribution_by_s2c=false`，完整语料继续被 Git 忽略。
- 协议影响：canonical 记录冻结 TEXTOIR `benchmark_labels` 的标签顺序，并使用
  `numpy.random.seed(seed)`/无放回 choice 生成 KIR registry，兼容 TEXTOIR 当前 Known-class
  选择语义。新增 ADB/DA-ADB compatible TSV export；它们只是固定输入格式，不是方法复现结果。
- E0：完成三份 canonical（CLINC150 23,700/150/1,200 native OOS；Banking77 13,083/77；
  StackOverflow 20,000/20）、165 registry、165 views 与 990 exports。临时将 `../textoir` 改名后，
  全量 validate、36-cell dry-run 和三数据集 Gate view loading 均通过；恢复后 TEXTOIR 工作树仍干净。
- E1：执行 `configs/experiments/protocol_v2_textoir_v1/smoke_gate.yaml`，36/36 frozen-MiniLM Gate
  单元完成、0 失败、关键指标无 NaN。结果仅写入
  `../artifacts/s2c/runs/protocol_v2_textoir_v1/`，不覆盖历史 artifact。
- E2：生成 1,650 单元计划并按 `--resume --shard-name e2_core` 启动；状态写入同一 run root 的
  `plans/gate_core_dense.e2_core.state.json`。E3--E7 未启动。
- 修改文件：`src/s2c/runtime/paths.py`、`src/s2c/data/*` 的协议/导出/验证路径、Gate matrix/runner/
  summary、活动 configs、数据与运行文档、测试与 implementation audit generator。
- 验证：19 个针对 data/admission/registry/export/runner 的单元与集成测试通过；`py_compile` 通过；
  E0 全量验证返回 3 datasets / 165 registries / 165 views / 990 exports；E1 verify 返回 36 complete /
  0 missing / 0 invalid；data tracking check 通过。
- 风险与下一步：E2 是长时间可恢复 sweep，完成前不得解释 dense-grid 结论或启动 E3--E7；公开 Git
  只能跟踪 manifest/轻量汇总，绝不能包含 StackOverflow 完整文本、embedding、checkpoint、Parquet
  或逐样本输出。完成 E2 后先生成 summary 和机制分析，再决定后续 boundary/baseline/representation/Pipeline 阶段。

### Validation update：活动协议测试与审计收口

- 验证：`pytest tests/unit -q`（213 passed）、`pytest tests/integration -q`（8 passed）、
  `pytest tests/smoke -q`（3 passed）以及 `python -m compileall src scripts tools` 均通过；公开结果
  SHA256 verify、data-tracking check、registry audit 和 `git diff --check` 也通过。
- 审计影响：implementation audit 现在分别报告 E1 的固定 36 单元和 E2 的实时可恢复进度，避免把
  E2 已完成单元错误计入 E1。此更新不改数据、历史 artifact 或 E2 的配置；E2 继续只写入 ignored
  artifact run root，E3--E7 仍未启动。

### E2 closeout：冻结三数据集 dense Gate sweep 并完成配对分析

- Base commit：`ea210083b331c489059f275edcc2e0c3241cfba7`；E2 代码身份由
  `artifacts/s2c/runs/protocol_v2_textoir_v1/E2_CODE_SNAPSHOT.patch` 及其 SHA256
  `127c80ecb2ae57e96e51dc3f146d5c8083dcc8653acad6940d6d42840a020b6f` 绑定。closeout 只新增派生汇总和文档，未改写任何 E2 run、配置、canonical、registry 或 embedding。
- 目标：审计 `dataset × 11 KIR × 5 seed × 5 K × 2 distance = 1,650` 个
  `protocol_v2_textoir_v1/e2_gate_core_dense` 单元，并以
  `dataset × KIR × seed × distance` 为配对单位比较 K=2--5 与 K=1。
- 完成状态：`1,650/1,650`，失败 `0`，缺失 `0`，重复 `0`，无效 `0`；所有 run 的
  protocol、canonical manifest、registry、resolved config、MiniLM encoder 文件哈希通过审计。
- 输出：派生证据位于
  `artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e2_closeout/`，包括完整性报告、
  dataset/KIR/K 汇总、配对 K 效应、距离比较、Known/OOS trade-off、K 选择边界和无效 run 清单。
  核心指标使用固定 RNG seed `20260725` 的 10,000 次 paired percentile bootstrap；计时和簇规模字段仅作描述性 mean/std。
- 结论边界：E2 使用固定 Known-only `mean_std` 边界，没有 per-K validation 选择，因此
  `oracle_test_best_k` 只作测试敏感性上限，`validation_selected_k` 不可从本 sweep 推出。
  E2 之后没有启动 E3--E7，也没有重新训练或生成新表示。
- 风险与下一步：CLINC150、Banking77 和 StackOverflow 的多中心效应分别呈现条件性、条件性和明显有害信号；
后续若继续，只能先根据配对区间决定是否做 KMeans/random-balanced、tiny-cluster 和 Known-only reliability 分析。

### E3 closeout：多中心机制诊断完成

- Base commit：`1f299d33bee949d934a74cadbf6adb1962d620ea`；活动协议为
  `protocol_v2_textoir_v1`。E2 保持只读，E3 使用独立根目录
  `artifacts/s2c/runs/protocol_v2_textoir_v1/e3_mechanisms/`。
- Provenance：`E3_PROVENANCE_SNAPSHOT.json` SHA256 为
  `58478682570e78afa5f35e903c33fae51d3af42eaad42cb3da5f0be34faec93c`；
  `E3_CODE_SNAPSHOT.patch` SHA256 为
  `ac1ecb5c9813c5e70665a967ab822abe4929fe3c01bda8772fc9971cb789d7ce`。快照记录
  E2 closeout、canonical/registry、MiniLM 和 Python/NumPy/SciPy/scikit-learn 版本哈希。
- 目标：在不改变 legacy detector、E2 配置或 embedding 的前提下，比较 KMeans 与
  random-balanced 分簇，并使用 train/calibration-only 信号诊断稳定性、tiny cluster、覆盖和
  reliability；不定义最终 adaptive-K。
- 执行：E3-A `720/720` 完成、失败 `0`；E3-B/C `180/180` 诊断组完成、失败 `0`，每组
  40 个 partition/seed/distance 组合，共 `7,200` 行诊断。KMeans seed=42 与 E2 固定参考单元
  的指标逐项相等（`max_abs_delta=0.0`）。E2 run 未被写入或修改，E4--E7、ADB、DA-ADB、
  MOGB、表示学习和完整 Pipeline 未启动。
- 结果摘要：`../artifacts/s2c/runs/protocol_v2_textoir_v1/e3_mechanisms/summaries/`，包括
  `E3_partition_paired_effects.csv`、`E3_cluster_stability.csv`、
  `E3_known_coverage_analysis.csv`、`E3_reliability_features.csv`、
  `E3_dataset_mechanism_decision.md` 和 `E3_integrity_report.md`。Combined OOS 的 KMeans−random
  OOS F1 平均差约为 Banking77 `+0.0437`、CLINC150 `-0.0137`、StackOverflow `-0.1543`；
  这支持“数据集条件性”而非统一多中心收益。Known-only reliability 关联为探索性证据，不能
  作为测试集选 K 或最终 adaptive-K 声明。
- 验证：E3 verifier（含 E2 等价性）、逐组 40 行结构审计、Ruff 新增模块检查和定向 E3 测试
  已通过；完整 unit/integration/smoke、compileall、data tracking、development-log 和 registry
  audit 在最终交付前复跑。风险：诊断文件保留全局 tiny-cluster 字段，同时 intent-level features
  提供按意图阈值的信号；解释时应优先使用后者，避免将全局阈值误读为每意图碎片化。
- 下一步：停止在 E3；先完成全套回归验证和结果审计，再决定是否另行批准 E4--E7。

### E3 mechanism diagnostics：开始独立多中心机制层

- Base commit：`1f299d33bee949d934a74cadbf6adb1962d620ea`；E2 保持冻结，E3 使用独立的
  `artifacts/s2c/runs/protocol_v2_textoir_v1/e3_mechanisms/` 根目录，不覆盖或 resume E2。
- 目标：在固定 Frozen MiniLM、mean+std 边界和相同 E2 embedding cache 下，比较 KMeans 与
  random-balanced 分簇，并用 train/calibration-only 特征诊断稳定性、tiny cluster、覆盖风险和
  reliability signal；本阶段不实现最终 adaptive-K，也不启动 E4--E7。
- 修改范围：新增 `src/s2c/experiments/partitions.py`、`mechanism_runner.py`、`mechanism_summary.py`、
  `mechanism_verify.py`，E3 配置与 `scripts/experiments/run_e3_*`、摘要/验证入口，以及对应单元测试。
  分簇适配通过注入 legacy detector 的中心和标签实现，未修改 E2 detector 行为。
- 计划：E3-A 为 720 个 Gate 单元；E3-B/C 为 180 个诊断组、每组 2 种分簇 × 10 个初始化 × 2
  个距离；K=1 只读引用 E2。E3 formal run 要求先冻结 `E3_PROVENANCE_SNAPSHOT.json` 和
  `E3_CODE_SNAPSHOT.patch`。
- 验证：E3-A/E3-B/C 计划分别解析为 720/180；random-balanced、seed 可复现、分簇尺寸和
  KMeans(seed=42) 注入适配器单测通过；真实 `clinc150/KIR=.50/seed=42/K=2/euclidean`
  的适配器指标与 E2 对应单元逐项相等（浮点容差 `1e-12`）。尚未启动正式 E3 run。
- 风险与下一步：E2 仍需保持原始 hash；先完成 E3 provenance 冻结，再运行 E3-A 并做完整性审计，
  随后运行 train/calibration-only 稳定性诊断与汇总。E4--E7、ADB、DA-ADB、MOGB、表示学习和
  完整 Cascade 在 E3 收口前保持未启动。

### R1 Geometry-Preserving CE-Recon pilot：收口（2026-07-28）

- Base commit：`1f299d33bee949d934a74cadbf6adb1962d620ea`；活动协议为
  `protocol_v2_textoir_v1`。E2/E3 run、配置、canonical、registry、views、exports 和 embedding
  保持只读；R1 使用独立根目录
  `artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation/`。
- 方法：在 Known-only CE-Recon 上加入 batch 内 pairwise cosine relation preservation；teacher
  MiniLM 冻结，训练只读取 Known train，checkpoint 只使用 Known calibration macro-F1 选择。
  beta 候选 `0.1/0.5/1.0` 由三个数据集 seed=42 的 Known-only 目标统一选择，结果为 `beta=1.0`。
- 规模：9 个 beta 候选训练、9 个 CE-Recon 正式训练、6 个非 seed-42 Geometry checkpoint
  训练；3 个 seed-42 Geometry 行复用 beta-selection checkpoint，共 24 次实际训练；Gate
  `3 datasets × 3 seeds × 3 representations × 2 K × 2 distance = 108/108`，失败 0。
- 结果：相对 CE-Recon，K=1 OOS F1 在 CLINC150/Banking77/StackOverflow 分别变化
  `+0.0035/+0.0215/+0.0255`，平均 ID Recall `-0.0054`；effective rank、pairwise relation
  correlation、kNN preservation 均提高，collision 平均下降 `0.0131`。Banking77 near-OOS
  下降 `-0.0291`，StackOverflow K=2 仍严重退化 `-0.5908`，因此结论为条件性表示层证据，
  不宣称普遍解决多中心问题。
- Provenance：`R1_PROVENANCE_SNAPSHOT.json`、`R1_CODE_SNAPSHOT.patch`、配置 hash 和 E2 closeout
  hash 已保存；`R1_CLOSEOUT.md`、`R1_method_decision.md`、`R1_gate_summary.csv`、
  `R1_geometry_analysis.csv` 和 `R1_pilot_effects.csv` 是收口证据。
- 验证：R1 单元测试、编译和 Ruff 定向检查通过；E2/E3 没有被写入；ADB、DA-ADB、MOGB、R1_full
  和完整 Pipeline 未启动。
- 决策：`pilot_success_conditionally_r1_full_candidate`。下一步只能生成 R1_full 计划并进行
  论文 claim 审阅，不能自动扩展 KIR、运行外部 baseline 或接入完整 Pipeline。

## 2026-07-28：R1_full 计划与预检

- 目标：将 R1 pilot 的条件性表示证据扩展到 KIR `0.25/0.50/0.75` 和五个正式 seed，仍以 K=1 为主、K=2 为结构诊断。
- 新增：`configs/experiments/protocol_v2_textoir_v1/r1_geometry_preserving_full.yaml`、
  `src/s2c/experiments/r1_full_runner.py`、`scripts/experiments/plan_r1_full.py`。
- 范围：135 个表示 cell、270 个 Gate 单元；冻结 `beta=1.0`，距离按 pilot 选择为 `mahalanobis_diag`。
- 数据影响：只读取 protocol_v2_textoir_v1 的 canonical/views 和 E2 embedding cache，不修改 E2/E3 artifact。
- 状态：仅完成 plan/dry-run 与语法检查，未开始训练；R1_full 的 provenance 在正式启动前冻结。
- 下一步：冻结 provenance 后按 `--run` 执行并支持断点恢复；不启动 ADB、DA-ADB、MOGB 或完整 Pipeline。

## 2026-07-28：R1_full 受控运行启动

- provenance：`../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation_full/R1_FULL_PROVENANCE_SNAPSHOT.json`。
- 规模：135 个表示 cell、270 个 Gate 单元；三数据集、KIR `0.25/0.50/0.75`、5 seed、Frozen/CE-Recon/Geometry。
- 固定项：`beta=1.0`（R1 pilot Known-only 选择）、`mahalanobis_diag`、`mean_std`、K `{1,2}`。
- 隔离：不修改 E2/E3，不使用 OOS 训练或 test 选择，不启动外部 baseline/Pipeline。
- 状态：已启动，支持按 cell 断点恢复；完成后生成 R1_full integrity/closeout。

## 2026-07-28：R1_full 收口

- 完成：135/135 表示 cell、270/270 Gate 单元，0 failed、0 invalid；摘要位于 `../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation_full/summaries/`。
- 结果：K=1 OOS F1 相对 CE-Recon 三数据集均为正；near-OOS 仅 CLINC150 小幅提升，Banking77 与 StackOverflow 下降；K=2 在 Banking77 条件性有效，StackOverflow 仍严重退化。
- 研究决策：将 R1 定位为条件性单中心表示适配证据，不宣称普遍多中心或完整 Pipeline 改善。
- 风险记录：`R1_full_geometry_analysis.csv` 中历史几何函数的 intra/inter 字段仅作 teacher-reference 诊断，closeout 不使用它们；论文主结论使用 effective rank、relation correlation、kNN preservation、collision 和 Gate 指标。
- 下一步：claim 审阅与外部直接 baseline 规划；不启动 E4--E7。

## 2026-07-28：R1 K=1/K=2 指标审计

- 目标：核对 StackOverflow `-0.5908` 是否由指标列、聚合顺序或协议混用造成。
- 方法：仅读取 R1 pilot/R1_full 配对 CSV、R1 原始 metrics 和历史 Frozen v19 汇总；没有重跑实验，
  没有修改 E2/E3/R1 原始结果。
- 结果：pilot 的 `-0.5908` 为 Geometry CE-Recon combined OOS F1 的 K=2−K=1 配对均值；
  R1_full Geometry 15 个单元均值为 `-0.4852`，Frozen MiniLM 为 `-0.0915`。历史 v19
  Frozen KIR50 对角马氏结果为约 `-0.0622`，协议和表示不同。
- 决策：保留该差异作为表示依赖的机制证据，后续论文表格必须显式包含 metric、representation、KIR、
  seed 和 distance，避免将 near-OOS 单元差值写成 combined OOS 均值。
- 证据：`R1_FULL_K1_K2_AUDIT.md`。
