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

## 2026-07-28：R1 contract repair pilot 收口

- Base commit：`5880a339c809a3dada72b1a21f92c4a9ece42676`；活动协议为
  `protocol_v2_textoir_v1`；新阶段为 `r1_contract_repair_v1`。
- 目标：隔离并修复三项契约问题：classifier 使用 pooled/normalized pooled 不明确；student
  intra/inter 几何统计误用 teacher distance；near/medium/far 使用 test OOS quantile。
- 修改文件：`src/s2c/experiments/geometry_preserving.py`、`src/s2c/experiments/r1_contract_repair.py`、
  `scripts/experiments/run_r1_contract_repair.py`、`configs/experiments/protocol_v2_textoir_v1/r1_contract_repair.yaml`、
  `tests/unit/test_r1_geometry.py`；更新研究台账、决策和 claim 审计。
- 数据影响：只读取 StackOverflow/KIR50/seed `{42,87,100}` 的 protocol_v2_textoir_v1 E2 cache 和
  Known train/calibration；不读取 `textoir/data`，不改变 canonical、registry、views、exports 或旧 artifacts。
- 方法：12 个 trainable checkpoints（4 个 trainable representation × 3 seeds）和 30 个 Gate 单元；
  explicit `pooled/pooled_norm/teacher_pooled/teacher_norm`；默认 `classifier_input=pooled`、
  `geometry_input=normalized_pooled`、`gate_embedding=normalized_pooled`；geometry loss 固定 beta=1.0。
- 分桶：当前 calibration 是 Known-only，没有合法 validation OOS，因此 30 个 Gate 行均为
  `exploratory_unavailable_validation_oos`，q20/q80 为空，未使用 test OOS 选桶或调参。
- 结果：pooled vs normalized head 的 K1 OOS F1 均值为 `0.8840` vs `0.8842`，K2 为 `0.3002` vs
  `0.2444`；pooled-head Geometry vs CE-Recon 的 K1 变化 `+0.0009`，K2 仍为结构性退化；
  student/teacher intra/inter 和 relation correlation 已分离记录。
- 产物：`../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_contract_repair_v1/`，包含 provenance、
  code patch、12 个 checkpoint manifest、五份 Gate/geometry/near/K1-K2 CSV、integrity、decision 和 closeout。
- 测试：R1 geometry unit tests `6 passed`（包含不同 teacher/student geometry 和 validation-only bucket）；
  pilot `12/12` checkpoints、`30/30` Gate、`0` failures；E2/E3/R1 legacy artifacts 修改状态为 false。
- 研究决策：旧 R1 标记 `completed_but_superseded_by_contract_audit`；旧 Gate prediction 保留，旧
  geometry 标记 `invalid_metric_implementation`，旧 near-OOS 标记 `exploratory_test_defined_bucket`。
  corrected R1_full 未获授权；不启动外部 baseline、ADB、DA-ADB、MOGB 或完整 Pipeline。
- 风险与下一步：near-OOS 正式结论仍缺 validation OOS 契约；下一步只能是 contract-repair claim
  审阅和是否另行批准 corrected R1_full 的决策。

## 2026-07-28：StackOverflow 多中心边界归因收口

- Base commit：`bca13b51221a5c327fa0197229e783c42f57bba7`；活动协议为
  `protocol_v2_textoir_v1`，阶段为 `multicenter_boundary_attribution`。
- 目标：在不训练 encoder、不修改 legacy detector 和父实验 artifacts 的前提下，判断 K=2 崩溃
  主要来自 per-cluster covariance、原始距离选球、半径归一化选球还是 Known-only 半径。
- 修改：新增独立边界归因模块、CLI、60 单元固定配置与单元测试；研究状态检查器不再硬编码 R1
  阶段名。没有修改 E2/E3/R1 的 run、checkpoint、canonical、registry、view 或 export。
- 实验：StackOverflow/KIR50/seed `{42,87,100}`，Frozen、CE-Recon pooled-head、Geometry
  pooled-head，K `{1,2}`；完成 `60/60`，失败/缺失/无效均为 `0`。6 个 adapted checkpoint 只做
  一次确定性编码并冻结缓存，没有 encoder 训练。
- 结果：shared-intent diagonal covariance 是唯一一致改善 K=2 的组件，但三种表示均未通过
  预注册安全门。最佳缓解方向 Frozen shared covariance 的 K2−K1 OOS F1 仍为 `-0.0331`，
  false acceptance 增加 `+0.0356`；CE-Recon/Geometry 分别为 `-0.3551/-0.3404`。归一化选球
  与 q95 半径继续扩大 false acceptance。
- 决策：`stop_fixed_kmeans_multicenter_rescue`。不再通过更多损失、K、半径或 selector 救活
  StackOverflow 固定 KMeans 多中心；下一步仅准备统一协议的最小外部 Baseline pilot，完整
  Pipeline 继续暂缓。
- 产物：`../artifacts/s2c/runs/protocol_v2_textoir_v1/multicenter_boundary_attribution/`；
  初始 scoring patch SHA256 为
  `16ecffdea31faded6305b9a9d5d5d165ac3a59a008315fdc4c1158f1edcca0d7`，closeout 另记录分析源码
  SHA256。
- 验证：60 run manifests 完整且 run root 与 E2/E3/R1 隔离；`pytest tests/unit -q`
  为 `237 passed`，integration/smoke 为 `8/3 passed`；Ruff、compileall、research-state、
  data-tracking、development-log、registry、public-results SHA256 和 Git whitespace 检查均通过。

## 2026-07-28：源码 namespace 与目录边界整理

- Base commit：`bca13b51221a5c327fa0197229e783c42f57bba7`；本轮不运行训练、不修改
  `../artifacts`、canonical、registry、views、exports 或历史结果，也不执行 Git commit/push。
- 目标：移除令人混淆的 `src/s2c/` 嵌套包，建立唯一且可检查的 active/legacy 源码边界。
- 当前活动包：`src/protocol_v2/`，使用 `protocol_v2.*` namespace；包含当前 data、evaluation、
  experiments、active Gate、runtime 和 tracking。
- 历史兼容包：`src/legacy/`，使用 `legacy.*` namespace；包含 v19 Gate、Router、Expert、pipeline、
  严格 SVDD、旧 runtime 和兼容 CLI。旧多球 import 路径保留为转发 wrapper，实际实现只保留在
  `src/protocol_v2/gate/multi_sphere_oos_detector.py`。
- 规范：新增 `docs/CODE_LAYOUT.md`，同步 `AGENTS.md`、`PROJECT.md`、`STRUCTURE.md`、RUNBOOK、
  reproducibility 和数据入口命令；`pyproject.toml` 的 package discovery/console entrypoint 已切换
  到 `protocol_v2*` 与 `legacy*`。
- 迁移：源码、测试和工具的 import 已统一；registry/entrypoint audit 已重新生成，配置不再引用
  `src/s2c`、`src/gate` 等旧物理路径。artifact 目录名和历史 provenance 文档中的旧事实未改写。
- 验证：布局/CLI 回归 `15 passed`；协议 runner、E3 partition、admission、research-state 回归
  `14 passed`；完整 `pytest tests/unit -q` 为 `242 passed`，integration 为 `8 passed`，smoke 为
  `3 passed`；compileall、Ruff、registry audit、data tracking、development-log check、import help、
  active/legacy detector identity 和 `pip install -e . --no-deps` 均通过。
- 状态：`find src -maxdepth 1` 现在只剩 `protocol_v2` 与 `legacy` 两个源码 namespace；无
  `src/s2c`、无 `src/__init__.py`，工作树仍包含本轮未提交迁移和用户既有改动。
- 下一步：由用户审阅 `docs/CODE_LAYOUT.md` 和迁移 diff 后决定是否提交；本轮不自动 commit/push。

## 2026-07-28：MiniLM training and StackOverflow repair pilot

- Base commit：`bca13b51221a5c327fa0197229e783c42f57bba7`；工作树含既有源码布局整理，
  本阶段不执行 Git commit/push，不修改 E2/E3/R1 原始 artifacts。
- 目标：在 `protocol_v2_textoir_v1` 下逐样本审计 StackOverflow Frozen K=1/K=2 路径，
  并在完全相同的 Known train/calibration/test 和 Gate 契约中比较 Frozen、Head-only CE、
  Full CE、SupCon、CE-Recon。
- 新增源码：`src/protocol_v2/experiments/minilm_training.py`；薄入口位于
  `scripts/experiments/audit_stackoverflow_k1_k2.py`、`run_minilm_training_pilot.py`、
  `summarize_minilm_training_pilot.py`、`verify_minilm_training_pilot.py`；配置为
  `configs/experiments/protocol_v2_textoir_v1/minilm_training_and_stackoverflow_repair.yaml`。
- 数据与选择：CLINC150、Banking77、StackOverflow；KIR=0.50；seed `{42,87,100}`；
  K `{1,2}`；Euclidean/diagonal-Mahalanobis；训练和 checkpoint 选择只读取 Known train 与
  Known calibration，未使用 OOS 或 test。
- 规模：36 个 trainable checkpoint、180/180 Gate cells；StackOverflow 逐样本和子簇审计写入
  `../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_training_and_stackoverflow_repair_v1/`。
- 审计结果：E2 cache/view 的 sample ID、embedding bytes、评分、协方差、半径契约通过；Frozen
  K=1/K=2 指标复现 E2。新 OOS false-accept 样本与每个子簇的样本数、方差、半径和贡献已导出。
- 方法结果：Full CE/CE-Recon 在部分数据集提升 K=1，但 StackOverflow K=2 仍大幅退化；SupCon
  也未把 K=2 退化降至预注册安全门内。阶段决策为停止通过表示训练救活固定后处理多中心，
  保留最强单中心对照和 StackOverflow boundary-union failure 证据。
- 状态：`EXPERIMENT_LEDGER.csv` 已将未执行的旧 M1 planned row 标记 superseded，并登记本阶段
  为 `do_not_repeat`；`RESEARCH_STATUS.md`、`DECISION_LOG.md`、`PAPER_CLAIM_AUDIT.md`、
  `PROJECT.md`、`EXPERIMENTS.md` 已同步。未启动 R1_full、ADB、DA-ADB、MOGB 或完整 Pipeline。
- 后处理校正：首次 closeout 生成时将每个 distance 的最后一个 seed 误标为“均值”；已仅修正
  汇总器并重新生成 closeout，改为三 seed 的均值。180 个 Gate 结果、paired delta、checkpoint
  和 StackOverflow 逐样本审计均未修改；校正后的 closeout 明确记录该事实。
# 2026-07-29 — MOGB baseline integration and StackOverflow smoke

- Base commit: `294d6f2`；branch: `main`；本轮未 commit/push。
- 目标：隔离引入 MOGB 作为直接多中心基线，统一读取当前 `protocol_v2_textoir_v1` 数据、registry、
  frozen MiniLM cache 和评价器，不修改历史 E2/E3/R1/M1 artifacts。
- 修改文件：`src/protocol_v2/experiments/mogb.py`、`scripts/experiments/run_mogb_fair.py`、
  `scripts/experiments/reproduce_mogb_original.py`、`scripts/experiments/aggregate_mogb_results.py`、
  `configs/baselines/mogb_fair.yaml`、MOGB 审计文档、粒球单元测试、研究台账。
- 第三方来源：`third_party/mogb_official` pinned at
  `5b689e2a03de0d86ec41212825e5db8d7f0e5c02`；未将其文件导入 active package。
- 数据影响：无 canonical/registry/view/export 变更；StackOverflow 只使用现有本地 benchmark snapshot。
- Artifact：`artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_baseline_v1/`；包含官方 preflight、30 个
  StackOverflow KIR=0.50 三 seed + 三数据集 seed42 pilot run、ball statistics、predictions 和轻量 summary。
- 执行：`reproduce_mogb_original.py`；StackOverflow seed 42/87/100，6 methods/seed；
  `aggregate_mogb_results.py`。
- 测试：MOGB 粒球单元测试 `3 passed`；官方 preflight 明确 blocked；pilot `30/30` 完成。
- 初步结果：MOGB official-style mean radius 在 frozen MiniLM 上明显牺牲 Known Recall；组件混合结果不能
  被写成官方复现或 s2c SOTA。
- 风险：上游仓库无 LICENSE、依赖旧 BERT/TextOIR 格式并缺 `utils`；当前没有官方论文数字复现。
- 下一步：审阅 smoke provenance 与方法表，确认是否值得登记三数据集/三 KIR/五 seed 扩展；不启动完整矩阵。
- 收口：补充 `run_mogb_sweep.py` 的确定性 dry-run/resume 入口，并记录正式协议的 seeds 为
  `{13,42,87,100,123}`；用户原计划中的 seed 0/1/2 不存在于当前 protocol_v2_textoir_v1 views，
  未重建数据或静默替换 split。全矩阵仅登记为 `planned_not_started`，不因有入口而自动启动。
- 扩展：pilot 在 StackOverflow 三 seed 上保持低波动但显示稳定的 Known 覆盖损失，故按已登记
  270-cell 计划启动 MiniLM 公平矩阵；已完成的 30 个重叠单元通过 `--resume` 复用，实际新增
  240 个评分单元。官方 BERT/TextOIR 路径仍不启动。
- 收口：270/270 完成、0 失败；聚合器验证每个 dataset×KIR×seed×method 恰有 5 个 seed。
  270 个结果仍是冻结 MiniLM 的统一协议组件比较，不是官方 MOGB BERT 复现；官方路径继续
  保持 `audited_not_reproduced`。

## 2026-07-31：CLMSG Milestone 1--3 启动

- Base commit：`294d6f2`；branch `main`；工作树已有未提交 MOGB 接入，本阶段不
  commit/push，也不修改 E0--E3、R1、M1 或 MOGB artifacts。
- 目标：用训练支持样本的局部尺度和 Known-only split conformal 替代中心--半径覆盖；先只实现
  KNN-only、local-scale KNN 与 local-scale conformal，不提前加入 manifold 或 label entropy。
- 数据：固定 `protocol_v2_textoir_v1`；现有 `train_known` 作为 proper-train，现有 Known-only
  `calibration_known` 作为校准，`test_combined` 只用于冻结方法后的评价；不重新划分 Known intents。
- 修改：新增 `src/protocol_v2/gate/clmsg.py`、薄 CLI `scripts/experiments/run_clmsg.py`、
  `configs/gates/clmsg.yaml`、单元测试和 `docs/clmsg/CLMSG_PIPELINE_AUDIT.md`。
- 计划：StackOverflow/KIR50/seed13 smoke；只有数值有效且 Version C 不被 Single centroid 明显
  支配时才补 seeds42/87。当前计划共 18 个方法/alpha 输出单元，独立 artifact root 为
  `../artifacts/s2c/runs/protocol_v2_textoir_v1/clmsg_v1/`。
- 初始验证：CLMSG 单元测试 `8 passed`，Ruff、py_compile 和 seed13 dry-run 通过；正式 smoke
  尚未启动。
- 风险：local-scale exact search 计算量高于 centroid Gate；global conformal 只控制整体 Known
  coverage，不保证每意图 coverage；Version C 未通过前不得实现后续复杂模块。

## 2026-07-31：CLMSG Milestone 1--3 停止门收口

- 完成：补齐附件要求的 global、class-conditional、hybrid `{0.25,0.50,0.75}` 支持模式；
  `tests/unit/test_clmsg.py` 增至 10 个测试。正式 seed13 输出为 26/26，0 失败/无效。
- 旧预检：最初 global-only 输出保留在 `clmsg_v1/stackoverflow/.../seed_13`，不覆盖、不纳入正式
  计数；补齐模式后的正式结果位于 `clmsg_v1/support_modes_v1/stackoverflow/.../seed_13`。
- 结果：primary class-conditional conformal alpha=0.05 的 OOS F1 `0.3336`、Known Recall
  `0.9493`；Single-centroid 为 `0.6957/0.8820`。所有 Version C 固定 alpha/mode 均未超过
  Single-centroid；最佳描述性行 OOS F1 `0.4975`。
- 机制：共形 Known FR 接近 target alpha，但 local-scale 将普通 KNN AUPR `0.8431` 降至约
  `0.67--0.70`，导致大量 OOS 被接受。该问题不是 NaN、cache 错位或 calibration 泄漏。
- 决策：`stop_after_seed13`；不运行 seeds42/87，不实现 manifold/entropy/cross-conformal，
  不启动三数据集 full sweep。报告见 `docs/clmsg/CLMSG_RESEARCH_REPORT.md`，artifact closeout 见
  `../artifacts/s2c/runs/protocol_v2_textoir_v1/clmsg_v1/summary/`。
- 验证：CLMSG verifier 通过（26 个授权输出、156,000 条预测、1,000 条校准分数、无 split
  交叉或 test selection）；完整 unit/integration/smoke 分别为 `258/8/3 passed`；Ruff、
  compileall、research-state、data-tracking、development-log、registry、public-results SHA256
  和 `git diff --check` 均通过。
- Git：本轮未 add/commit/push；GitHub 仍落后于包含 MOGB 与 CLMSG 的本地工作树。

## 2026-07-30：CLMSG seed42/87 确认性扩展启动

- 原因：用户明确要求按协议尽可能补充实验；为避免把单 seed 负结果误写成稳定结论，新增独立
  `CLMSG_M4_CONFIRMATION` ledger 行，而不改写已冻结的 seed13 pilot。
- 计划：StackOverflow、KIR50、seed `{42,87}`，每 seed 26 个固定输出，共 52 个；复用现有
  MiniLM cache、Known-only calibration、配置和 evaluator，不训练模型、不修改数据。
- 约束：不使用 test 选 alpha、k 或支持模式；不启动 manifold、entropy、cross-conformal 或 full
  sweep；运行结束后独立汇总并将 ledger 从 `in_progress` 更新为最终状态。

## 2026-07-30：普通 KNN 全协议确认矩阵启动

- CLMSG 三 seed 收口：seed42/87 的 52 个固定输出完成，三 seed 共 78 个；primary Version C 的
  平均 OOS F1 为 `0.3604`，相对同 seed Single-centroid 平均下降 `0.3924`，局部尺度路线停止。
- 新问题：保留未做局部尺度归一化的普通 KNN，固定 `k=10`、Known calibration
  `alpha=0.05`，覆盖 3 数据集、KIR `{0.25,0.50,0.75}`、5 seeds，共 45 个输出。
- 工程：runner 现在严格遵守 config 的 `methods` 白名单；新增定向回归测试，保证 KNN-only
  配置不会生成 local-scale/conformal 结果。新 variant 与 CLMSG 三 seed artifacts 隔离。
- 边界：该矩阵是预声明 operating point 的公平基线，不用 test 选 k/alpha；完成前不启动
  manifold、entropy 或完整 Pipeline。

## 2026-07-30：KNN 邻居数敏感性启动

- k=10 结果：45/45 完整，九个 dataset×KIR 组对同 split Single-centroid 均为 `0/0/5`
  win/tie/loss；因此不能把单个 k=10 结果解释成 KNN 家族结论。
- 扩展：固定其他合同，仅增加 `k={5,20,30}`，共 135 个新评分单元；与 k=10 合并后形成
  `4 k × 3 datasets × 3 KIR × 5 seeds` 的 180-cell 敏感性。
- 实现：runner 增加显式 `--k-neighbors`、`--primary-alpha` 和 `--variant` 覆盖；每个 resolved
  config 都进入 run manifest/hash，variant 目录互相隔离。
- 约束：完整报告全部 k，不通过 test OOS 选择默认 k；不启动 manifold/entropy/Pipeline。

## 2026-07-31：KNN k 敏感性收口

- 新增脚本：`scripts/experiments/summarize_knn_k_sensitivity.py`、`verify_knn_k_sensitivity.py`，
  产出 `summary/knn_k_sensitivity_v1/` 下的 `all_runs.csv`、`k_overview.csv`、
  `dataset_kir_summary.csv`、`paired_vs_single.csv`、`significance.csv`、`integrity.json`、
  `KNN_K_SENSITIVITY_CLOSEOUT.md` 和 `KNN_K_SENSITIVITY_PROVENANCE.json`。
- 验证：`completed_cells=180`、`new_k_cells=135`、`existing_k10_cells=45`、`reused_k10_cells=3`、`unique_cells=180`、
  `missing=[]`、`invalid=[]`、`test_used_for_selection=false`。
- 结论：`k=5` 的整体 mean OOS F1 为 `0.6492`，`k=10` 为 `0.6335`，`k=20` 为 `0.5957`，
  `k=30` 为 `0.5565`；四个 k 均落后于同 split Single-centroid，最优也只是“最不差”而非可推广
  主方法。

## 2026-07-31：MOGB frozen-MiniLM 组件消融启动

- Base commit：`294d6f2`；活动协议 `protocol_v2_textoir_v1`；不修改或覆盖已冻结的 270-cell
  `mogb_baseline_v1`、E2/E3/R1/M1/CLMSG/KNN artifacts，也不执行 commit/push。
- 目标：通过 OFAT 隔离粒球 purity、minimum support 与 boundary distance/radius 的贡献；不训练
  BERT/MiniLM，不重新划分 Known intents，不使用 test 选参。
- 计划：3 datasets × 3 KIR × 5 seeds × 12 非默认配置 = 540 个新评分单元；默认配置的 45 个
  reference cells 直接读取 `mogb_baseline_v1/mogb_minilm`，不重复计算。
- Artifact：`../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_ablation_v1/`；实现和 dry-run 通过后
  冻结 config/plan hash，再把 ledger 状态改为 in_progress。
- 冻结：config SHA256 `db644d47fd38923a1b51a519fc0a53e24fca1a2c01cd6b1340863e1b54bb287b`；
  540-cell plan SHA256 `b97c4348cad00111c6cc8018383e200f76c1697ed819609d6c0bc56789f23dfd`。
  Unit `4 passed`、Ruff、py_compile、StackOverflow/KIR50/seed42 cache dry-run 均通过；正式矩阵开始前
  不再修改 config、runner 或 sweep plan。

## 2026-07-31：MOGB frozen-MiniLM 组件消融收口

- Base commit：`294d6f2`；活动协议 `protocol_v2_textoir_v1`；未修改 canonical、registry、view、
  embedding 或已冻结的 MOGB 270-cell fair matrix。
- 执行：完成 3 datasets × 3 KIR × 5 seeds × 12 OFAT variants，共 540/540 新单元；默认 MOGB、
  Single-centroid、MOGB-partition/ours-boundary 各复用 45 个控制单元。
- 工程修复：sweep 按 dataset×KIR×seed 只加载一次 immutable embedding；修正 variant 到独立
  artifact directory 的映射。粒球最近 seed 计算改为数学等价的二维平方欧氏形式，真实单元的
  predictions/balls/metrics 完全一致，峰值内存降至 355,296 KiB。
- 并发事件：重复 runner 曾对同一目标触发 atomic rename 冲突；重复进程终止后用单一 `--resume`
  收口，最终 540 个计划组合全部唯一、完整。三个隐藏 atomic temp directory 不进入正式统计。
- 汇总：`all_runs.csv` 540 行，baseline 135 行，四组合 boundary component 180 行，配对效应
  648+648 行，significance 1,296 行；verifier 返回 0 missing/invalid/duplicate。
- 结果：最佳 partition-only `purity_get=0.90` 相对默认 MOGB `+0.0135` OOS F1，仍比单中心
  `-0.0391`。`Euclidean+mean_std` 相对默认 MOGB `+0.0526`，但比单中心低 `0.1048` F1-All 和
  `0.2593` Known Recall；更多粒球与 diagonal Mahalanobis 均未改善该权衡。
- 决策：ledger 更新为 complete/do_not_repeat；不再扩 nearby purity/support/distance/radius。
  唯一下一步是官方 hierarchical representation-learning 的单格可运行性与复现审计。
- Artifact：`../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_ablation_v1/summary/`。
- Git：未 add/commit/push；GitHub 仍落后于本地。

## 2026-07-31：MOGB fixed-K mean-radius 对照启动

- Base commit：`294d6f2`；活动协议 `protocol_v2_textoir_v1`；540-cell OFAT 与 270-cell fair
  matrix 均保持只读。
- 目标：隔离动态粒球数量本身的作用，在相同 frozen MiniLM、Euclidean distance、mean-radius
  和 nearest-ball 推理下比较 adaptive MOGB 与 per-intent fixed K=1/2/3/4。
- 计划：3 datasets × 3 KIR × 5 seeds × fixed K={1,3,4} = 135 个新单元；fixed K=2 的
  45 个既有单元只读复用，合并分析共 180 格。
- 约束：不 test-select K，不重新编码，不修改数据/registry/cache，不启动 official BERT 训练或
 完整 Pipeline，不 commit/push。

## 2026-07-31：MOGB fixed-K mean-radius 对照收口

- Base commit：`294d6f2`；协议、canonical、registry、Frozen MiniLM cache、270-cell fair matrix
  与 540-cell OFAT 均保持只读。
- 执行：完成 fixed K=1/3/4 的 135/135 新评分单元；K=2 的 45 格来自
  `ours_partition_mogb_boundary`，真实 StackOverflow/KIR50/seed42 的 6,000 条逐样本预测零差异，
  后续 45 格 reference 的 manifest 与输入 hash 均通过验证。
- 审查修复：不改已有 run 数值，只将配置设为 K 列表的唯一来源、在 sweep 中强制验证 K2
  reference，并把 summary 的 partition 统一为 `per_intent_kmeans`、保留 source partition。
- 汇总：180/180 fixed-K、45/45 adaptive、45/45 single reference；0 missing/invalid/input mismatch；
  固定 10,000 次 bootstrap，所有 K 完整报告，没有 test-selected K。
- 结果：K1/K2/K3/K4 的总体 OOS F1 为 `0.7808/0.7590/0.7477/0.7482`；adaptive 为
  `0.7339`。K1 相对 adaptive 的 45 格 W/T/L 为 `45/0/0`，但其 F1-All 仍低于 s2c
  mean+std single-centroid，证明 boundary rule 仍是重要混杂因素。
- Artifact：`../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_fixed_k_mean_ablation_v1/`；
  closeout 为 `summary/MOGB_FIXED_K_MEAN_CLOSEOUT.md`。
- 风险：本阶段只隔离 Frozen MiniLM 下的 partition granularity，不包含官方 BERT/nearest
  sub-centroid 联合表示训练；不得据此写成官方 MOGB 失败或 SOTA 对比。
- 下一步：隔离的 official-logic 单格训练 smoke；不自动扩完整矩阵，不 commit/push。

## 2026-07-31：MOGB official-logic modernized smoke 收口

- Base commit：`09a956d`；活动协议 `mogb_official_logic_textoir_v1`；未修改 pinned
  `third_party/mogb_official` checkout，也未覆盖既有 270/540/180 fair artifacts。
- 实现：新增 device-aware 的运行时粒球适配、MPLCONFIGDIR 隔离、可记录 interrupted 状态以及
  smoke-only train/eval batch-size 覆盖；官方 stale-graph 修复仍通过外部 two-pass runtime 完成。
- StackOverflow/KIR50/seed0：1 epoch 工程 smoke 完整走通 BERT CE、epoch-end feature bank、GBNR
  粒球、nearest-ball evaluation 和结果写入；输出位于
  `../artifacts/s2c/external/mogb_official_modernized_smoke_v1/stackoverflow/kir_0.50/seed_0/`，
  结果只作执行链证据，不进入论文主表。
- Banking77/KIR50/seed0：设备修复后 CPU 尝试进入运行，但当前环境模型/特征 IO 耗时过长，未形成
  可报告指标，最终以 `returncode=130` 收口；准确 blocker 保存在同一 run manifest 和 closeout。
- 测试：MOGB compatibility/runtime unit `6 passed`；官方 checkout 未修改；未执行 commit/push。
- 风险：one-epoch modernized smoke 不是严格官方复现，Banking77 未形成收敛结果；不能据此宣称
  MOGB SOTA 或公平优越性。下一步只做文档/provenance/全仓回归收口，除非获得独立可复现 GPU/legacy 环境。

## 2026-08-01 — MOGB 官方逻辑收敛、BRAK pilot 与 DCLOOS preflight

- Base commit: `2ff028e` (`main`)，父仓库未自动 commit/push；MOGB checkout 保持 pinned
  `5b689e2a03de0d86ec41212825e5db8d7f0e5c02` 且未修改。
- 目标：在不重复 E0--E3/R1/MiniLM/MOGB fair/OFAT/fixed-K 的前提下，完成官方 MOGB 兼容层
  收敛尝试、Known-only BRAK pilot，并审计 DCLOOS 官方/统一入口的必要数据。
- 代码与配置：新增 `src/protocol_v2/experiments/brak.py`、`scripts/experiments/run_brak_pilot.py`、
  `scripts/experiments/run_dcloos_preflight.py`、官方收敛配置与最终汇总脚本；第三方源码分置于
  `third_party/dcloos_official/` 和 `third_party/dcloos_source/`，未覆盖上游文件。
- MOGB：StackOverflow 与 Banking77 各 5 seed，共 10/10 GPU 单元完成；结果写入
  `../artifacts/s2c/external/mogb_official_converged_v1/`，官方格式 F1-All 均值为 40.7243/19.2843。
  这是 modernized compatibility evidence，不是严格论文复现，未与 MiniLM-fair 主表混合。
- BRAK：StackOverflow/KIR50/seeds 42,87,100 计划 21 个 summary cells 全部完成，30 个 Known intent
  均选择 K=1；K>1 的 calibration risk 上升，未通过 expansion gate。输出位于
  `../artifacts/s2c/runs/protocol_v2_textoir_v1/brak_v1/`，不启动全量扩展。
- DCLOOS：preflight 返回预期 blocker；source 编译通过但缺失 `squad_placeh.tsv` 等官方
  open-domain negative corpus，未训练、未生成伪指标。报告位于 `docs/dcloos/` 和对应 external artifact。
- 记录：更新 `EXPERIMENT_LEDGER.csv`、`configs/experiment_registry.yaml`、`RESEARCH_STATUS.md`、
  `DECISION_LOG.md`、`PAPER_CLAIM_AUDIT.md`；生成 `results/final_baselines/summary.csv`（27 行，
  含同协议 fair 组件、BRAK 控制、官方兼容结果以及 ADB/DA-ADB/DCLOOS 的明确 not-run/blocked 状态）。
- 测试/验证：BRAK unit `3 passed`；官方 10/10 returncode=0；DCLOOS preflight 以预期 blocker 退出；
  下一步仅做全仓回归、manifest/hash/台账一致性检查，不自动 commit/push。
- 风险：官方结果不能写成 SOTA 或严格论文复现；BRAK 是负控制而非新方法；DCLOOS 需要外部语料和
  许可证后才能启动。任何后续基线必须新建 ledger 行并遵守 `do_not_repeat`。
## 2026-08-01 — 重新纳入 DCLOOS 端到端基线

- 原因：前一版外部基线计划错误地把重点缩成 MOGB、BRAK 和 ADB/DA-ADB，遗漏审稿人点名的
  DCLOOS fully end-to-end 方法。
- 当前状态：`blocked_missing_external_negative_data`。官方代码仍保留，源码可编译；缺失
  `squad_placeh.tsv` 外部 open-domain negative corpus，因此不伪造指标，也不使用 protocol test OOS 替代。
- 计划边界：MOGB 官方完整收敛、MOGB 表示上的 BRAK 和 DCLOOS 数据审计分别记录；语料可得时区分
  `DCLOOS-official` 与 `DCLOOS-adapted`，ADB/DA-ADB 只作为边界基线，不替代 DCLOOS。

## 2026-08-01 — MOGB 严格单格、BRAK 表示迁移与基线边界最终收口

- Base commit：`a51f97494afdcfca30dd3d94b54a6acdad8a41cc`；父仓库保持 dirty，未执行 add/commit/push；
  TEXTOIR 工作树仍干净，MOGB pinned checkout 未修改。
- MOGB 严格单格：新增 `configs/baselines/mogb_exact_reproduction_v1.yaml` 和
  `scripts/experiments/run_mogb_exact_reproduction.py`；StackOverflow 20,000 行、KIR50、seed0、
  BERT-base、官方 ball/nearest-ball/early-stop 流程的 `official_fixed` 与 `unified_zero` 共 2/2
  完成。两者 checkpoint SHA 相同，Acc/F1-All/F1-U/F1-K 为 `75.1667/68.3502/79.9676/67.1884`，
  相对论文参考不复现；完整证据在 `../artifacts/s2c/external/mogb_exact_reproduction_v1/`。
- BRAK 表示迁移：新增 `scripts/experiments/run_brak_mogb_representations.py`，复用同一 MOGB
  checkpoint 和 split，在 Frozen MiniLM、initial BERT、trained BERT 上完成 18/18 固定 K/BRAK
  汇总；trained BERT 仅 2/10 intent 选 K2，绝对指标仍低，停止扩大该路线。
- ADB/DA-ADB：新增 `docs/mogb_integration/ADB_DAADB_AUDIT.md`；隔离 preflight 两项均在训练前
  因当前 Transformers 缺失顶层 `AdamW` 导入而阻塞，没有生成性能数字。
- DCLOOS：保持 `blocked_missing_external_negative_data`；官方所需 `squad_placeh.tsv` 不存在，
  不使用 protocol test OOS 替代，不把 ADB/DA-ADB 当作端到端替代。
- 汇总与记录：更新 `MOGB_REPRODUCTION_REPORT.md`、`results/final_baselines/summary.csv`、
  `EXPERIMENT_LEDGER.csv`、`RESEARCH_STATUS.md`、`DECISION_LOG.md`；新增严格 MOGB、BRAK 表示
  迁移和 DCLOOS re-audit 行，旧 E0--E3/R1/MiniLM/MOGB fair artifacts 未覆盖。
- 测试/验证：严格 MOGB 两模式已完成并通过 manifest/hash 审计；ADB/DA-ADB preflight 结果按预期
  阻塞；待本条记录完成后运行定向 compile、Ruff、registry/data/research-state/development-log
  检查和 `git diff --check`。下一步只保留 DCLOOS 语料/许可证获取和未来隔离环境登记。

## 2026-08-01 — 外部基线最终验证与公开结果同步

- 变更：修正 `configs/experiment_registry.yaml` 中外部基线相对 artifact 根路径、manifest
  清单和计数来源；登记严格 MOGB 与 BRAK-on-MOGB 表示结果；保留 DCLOOS
  `blocked_missing_external_negative_data` 状态。
- 公开结果：将严格 MOGB/BRAK 的轻量 CSV/JSON 纳入 `configs/public_results.yaml` 白名单；未导出
  checkpoint、模型、原始文本或日志。`export_public_results.py --dry-run/--execute/--verify`
  均通过，79 个文件、1,797,774 bytes。
- 验证：实验注册表 `--check-only` 通过（19 entries）；研究状态、数据跟踪、开发日志、diff
  检查通过；unit `284 passed, 3 warnings`，integration `8 passed`，smoke `3 passed`。
- 结论：严格 MOGB 仍为 `not_reproduced_strict`，BRAK 表示迁移仍为负控制，ADB/DA-ADB 没有
  性能数字，DCLOOS 仍是必须保留但缺少外部负样本的端到端基线；不启动新矩阵。
- Git：父仓库仍 dirty，未执行 add/commit/push；第三方源码和完整 artifacts 保持本地审计边界。
- 交付补充：新增 `docs/mogb_integration/MOGB_EXACT_REPRODUCTION_REPORT.md`，逐项回答数据契约、
  收敛、论文差距、兼容性诊断、BRAK、扩展门和 DCLOOS/ADB/DA-ADB 边界；不修改任何已有 run。

## 2026-08-01 — 外部基线单格最终收口

- Base commit：`a51f97494afdcfca30dd3d94b54a6acdad8a41cc`；父仓库保持 dirty，未执行
  `git add/commit/push`；TEXTOIR 工作树仍干净。
- 目标：补齐 MOGB Banking77/KIR=.75/seed=0 严格单格，保留 DCLOOS 端到端基线，完成
  ADB/DA-ADB 隔离兼容单格，不扩大矩阵。
- MOGB：Banking77 13,083 行、77 标签、KIR=.75/seed=0 完成 1/1；Acc `57.0779`、
  F1-All `59.1627`、F1-U `53.1049`、F1-K `59.2671`，相对论文参考仍为
  `not_reproduced_strict`。
- DCLOOS：官方 Drive `squad.tsv`（SHA256
  `f6bf61866c86d3b11565826c3ca1faa00e31f196e0ad9bfd000ec45575fd426e`）复制为上游要求的
  `squad_placeh.tsv`，训练约三小时后按声明上限停止；artifact 标记 `timeout_incomplete`，
  无 final metrics，中间 `predictions.npz` 不进入汇总。
- ADB/DA-ADB：在 `textoir-py39` + torch-native AdamW overlay + isolated BERT pytorch 权重副本
  下完成 StackOverflow/KIR=.50/seed=0，各 1/1；ADB `Acc/F1-All/F1-open=88.53/87.6272/89.4712`，
  DA-ADB `90.07/89.2256/90.8978`。均标注 modernized compatibility，不称 strict reproduction。
- 代码/记录：新增 DCLOOS 中断 provenance、更新 ADB/DA-ADB 审计、研究状态/决策日志/ledger、
  registry 与 `results/final_baselines/summary.csv`；不修改既有 E0--E3/R1/MiniLM/MOGB artifacts。
- 风险与下一步：DCLOOS 仍缺最终可报告指标，且 ADB/DA-ADB 是单 seed 兼容参考；完成回归和
  公开结果 SHA 校验后冻结本轮，不启动 MOGB/BRAK/adaptive-K 大矩阵或完整 Pipeline。

## 2026-08-01 — DCLOOS reduced-budget recovery

- Base commit：`a51f97494afdcfca30dd3d94b54a6acdad8a41cc`；未执行 add/commit/push。
- 目标：在不扩大 DCLOOS 矩阵的前提下，验证已定位的官方外部 SQuAD 负样本能否完成一个可审计的端到端单格。
- 执行：`dcloos_official_oos_kir75_seed888_reduced_v2`，KIR `.75`、seed `888`、`max_epochs=100`、`patient=10`。
- 数据影响：使用同一 `squad.tsv` SHA256 `f6bf61866c86d3b11565826c3ca1faa00e31f196e0ad9bfd000ec45575fd426e`；原始 corpus 仍只在本地 artifact。
- 结果：上游 test evaluation 输出 5,700 条预测；raw JSON 因 overlay 缺失 `json` import 失败，恢复指标为 Accuracy `88.6842`、F1-All `90.2629`、F1-U `87.0527`、F1-K `90.2916`、OOS F1 `87.0527`。
- 记录：保留原 `failed` run manifest，新增 `recovery_manifest.json` 和 `recovery_metrics.json`；该结果是 reduced compatibility evidence，不是严格论文复现。
- 代码：补充 overlay 的 `json` import 和无 test-local 变量时的最终测试兜底；第三方 checkout 未改动。
- 测试：待本轮 registry、summary、public export、unit/integration/smoke 和 lint 回归完成后冻结。
- 风险与下一步：不再扩大 DCLOOS；继续保持 MOGB/BRAK/adaptive-K/完整 Pipeline 停止，先完成本轮验证。
# 2026-08-02 — workspace consolidation and adaptive-K feasibility audit

- Base commit: `7c9008d7e85c637334139783a91a80841725d628`.
- Scope: moved obsolete active-document entrances into `docs/archive/`; added the
  four active docs (`METHOD.md`, `CURRENT_STATUS.md`, `EXPERIMENTS.md`,
  `REPRODUCIBILITY.md`); updated research-state and registry paths.
- Data impact: none. `../artifacts/s2c/` was read-only and no training or old
  `do_not_repeat` matrix was started.
- New diagnostics: read-only fixed-K audit under
  `results/diagnostics/adaptive_k/`; MOGB four-combination preflight under
  `results/diagnostics/mogb_diff/`.
- New code: `tools/analysis/audit_adaptive_k.py`,
  `tools/analysis/diagnose_mogb_diff.py`, and the implementation-only
  `src/protocol_v2/experiments/adaptive_split_merge.py` with dry-run entry and
  unit tests.
- Risk: intent-level adaptive-K rows are test-sensitivity diagnostics, not a
  validation selector; MOGB A/C remain blocked because original accompanying
  data are unavailable.
- Next: run the validation suite, review the two diagnostic summaries, and
  register a pilot only if Known-only evidence justifies it.

## 2026-08-02 — consolidation closeout verification

- Base commit: `7c9008d7e85c637334139783a91a80841725d628`; working tree remains dirty and no commit/push was performed.
- Evidence refreshed: all 1,650 immutable E2 run directories were re-audited into
  `results/diagnostics/adaptive_k/`; the MOGB four-group diagnosis remains
  `public_code_not_reproduced_under_available_materials` because original accompanying data is absent
  and the pinned source stops at the missing `utils` import.
- New code scope: only the Known-only split–merge prototype, its dry-run entry point, tests, and read-only
  diagnostics; no model training, embedding generation, or writes under `../artifacts`.
- Verification: `pytest tests/unit -q` (289 passed), `pytest tests/integration -q` (8 passed), compileall,
  Ruff, experiment registry audit, data-tracking check, development-log check, research-state check,
  and `git diff --check` all passed.
- Decision: keep fixed-K/MOGB historical results frozen; do not promote test-oracle adaptive-K rows to a
  selection rule. Any future pilot must first include lambda sensitivity and an explicit Known/OOS leakage audit.

## 2026-08-02 — Correction: adaptive-K intent metric naming

- The read-only audit now names the intent-level class-F1 delta
  `delta_intent_class_f1_vs_k1`; the global `delta_f1_all_vs_k1` remains dataset-level only.
- Dataset rows additionally report the number of intent×KIR×distance groups with any K>1 test-oracle
  winner on at least 3/5 seeds. This is descriptive sensitivity evidence, not a validation selector.
- No model, embedding, registry, view, export, or original artifact was changed.

## 2026-08-02 — lambda leakage audit and sensitivity closeout

- Base commit: `03a6e26ed373747baccabbbb459d0a355af935ae`; branch:
  `experiment/lambda-leakage-audit`. The frozen E0--E3/R1/M1 and external-baseline
  artifacts were not modified, and no write occurred under `../artifacts/s2c`.
- Added the read-only audit/summary tools
  `tools/analysis/run_lambda_leakage_audit.py`,
  `tools/analysis/summarize_lambda_sensitivity.py`, and the focused unit test
  `tests/unit/test_lambda_leakage_audit.py`.
- Added the append-only ledger row `lambda_leakage_audit_v1` and diagnostic outputs under
  `results/diagnostics/lambda_leakage/` and `results/diagnostics/lambda_sensitivity/`.
  The audit covers 9 dataset/seed split contracts and 216 scoring rows from 18 unique detector
  fits; `paper_default_k` is an explicit alias of K=2 and does not create another fit.
- Evidence: all split IDs are disjoint; the active E2 contract uses fixed `lambda=1.0` without
  test-OOS selection; the 18 lambda=1 comparisons reproduce frozen E2 metrics and predictions
  exactly. Historical validation-OOS lambda searches remain excluded from the primary evidence.
- Decision: Known-only evidence does not satisfy the pre-registered adaptive-K gates. Banking77
  remains conditional and misses the strict F1-All/intent-level requirements; StackOverflow is a
  negative control with strong K=2 false-acceptance growth; no split--merge pilot is authorized.
- Targeted verification passed (`py_compile` for both tools and the two new unit tests). Full
  repository validation remains the final closeout step. Next: review this audit in the paper's
  parameter/leakage section; if adaptive-K is revisited, first establish an independent validation
  OOS or a strict Known-only intent-level selection contract.

## 2026-08-02 — lambda audit verification closeout

- Public whitelist now includes the aggregate adaptive-K/MOGB diagnostics and the lambda audit;
  `export_public_results.py --verify` passed with 102 records and 13,820,620 bytes. No raw text,
  embedding, checkpoint or sample-level score was added.
- Verification passed: 291 unit tests, 8 integration tests, 3 smoke tests, compileall, Ruff,
  data-tracking, development-log, research-state (`ledger_rows=39`), registry `--check-only`,
  public-result SHA/size verification, and `git diff --check`.
- The full registry hash pass was not used for this closeout because its checkpoint glob walks the
  large local artifact tree; the check-only audit passed and no source/artifact content changed.
- Decision and next step are unchanged: keep the lambda result as leakage/parameter evidence, do not
  launch adaptive-K or any new training stage until a legal validation-OOS or Known-only intent-level
  selection contract exists.

## 2026-08-04 — URCSG Known-only pilot closeout

- Base commit: `4eec67da592b580867afa8529024e85541adfa7d`; the worktree retained the pre-existing
  nested `third_party/mogb_official` change and no old artifact directory was modified. New files are
  `src/protocol_v2/experiments/urcsg.py`, `scripts/experiments/{run_urcsg_pilot.py,summarize_urcsg.py,verify_urcsg.py}`,
  `configs/experiments/urcsg_pilot_v1.yaml`, and focused unit/integration tests.
- Registered `urcsg_pilot_v1` under `protocol_v2_textoir_v1`. The pilot reused frozen all-MiniLM
  train/calibration/test caches for Banking77 and StackOverflow at KIR=.50, seeds 13/42/87, K=1..5,
  diagonal Mahalanobis, mean+std radius with lambda=1.0; no encoder training or implicit encoding.
- The selector estimates target-intent union risk from leave-one-known-intent-out Known calibration,
  records Wilson UCB95 and coverage deltas, and includes a shuffled episode negative control. Test OOS
  is not used for selection; `oracle_test_k` is descriptive only.
- Result: 6/6 cells, 54 metric rows and 720 candidate-intent rows, zero failures/missing/duplicates.
  URCSG-primary vs single-centroid was Banking77 OOS F1 -0.39 pp / F1-All -0.35 pp and StackOverflow
  OOS F1 -6.10 pp / F1-All -2.67 pp with false acceptance +10.93 pp. Both pre-registered dataset gates
  failed, so the full matrix was not authorized. The pilot is retained as a negative result: this
  union-risk proxy did not safely identify multicenter headroom in the frozen representation.
- Artifacts: `../artifacts/s2c/runs/protocol_v2_textoir_v1/urcsg_pilot_v1/`; lightweight summaries:
  `results/diagnostics/urcsg/`; the five aggregate files are now in the public-results whitelist and
  SHA/size verified. Validation included focused tests, pilot integrity/verify, registry
  check-only, and no changes to prior E2/E3/BRAK results. Next step: stop this selector and register
  a different experiment before any further multicenter expansion.

## 2026-08-04 — URCSG output-contract serialization repair

- Scope: serialization only; no URCSG cell, detector fit, embedding, test score, or frozen artifact was
  rerun or changed. Completed per-intent CSVs now expose the required generic `selected_k`,
  `selection_reason`, `ineligible`, and `skip_reason` fields while retaining the strategy-specific
  audit columns.
- Evidence: `summarize_urcsg.py` performs an atomic schema migration before aggregation; the six
  existing run manifests still report 6/6 and the pilot decision remains `stop` with unchanged deltas.
- Validation: focused URCSG tests, schema verification, research-state check, registry check-only,
  data-tracking check, public-results SHA/size verification, and `git diff --check` passed. The full
  unit suite was rerun after the migration.

## 2026-08-04 — CCSG competition-calibrated support pilot

- Base commit: recorded in `../artifacts/s2c/runs/protocol_v2_textoir_v1/ccsg_pilot_v1/CCSG_PROVENANCE.json`;
  no prior E2/E3/URCSG/BRAK artifact was modified and no encoder was trained.
- Registered `ccsg_pilot_v1` under `protocol_v2_textoir_v1`. The pilot reused frozen all-MiniLM caches
  for CLINC150, Banking77 and StackOverflow at KIR=.50 and seeds 13/42/87. It fitted K=1 and K=2
  detectors once per cell and compared current union, mixture-support, margin-only, CCSG joint and
  independent-AND scoring.
- CCSG uses a class-level log-mixture support score and top-two competition margin. A single joint
  threshold is calibrated from Known calibration only with target false rejection 5%; test OOS is never
  used for threshold or configuration selection.
- Result: 9/9 cells, 72 metric rows, zero failures/missing/duplicates/invalid. CCSG-K2 versus CCSG-K1
  F1-All changed by -1.26pp (Banking77), -3.68pp (CLINC150) and -0.39pp (StackOverflow); StackOverflow
  false acceptance increased 3.00pp. All pre-registered dataset gates failed, so no full matrix was
  authorized. Decision: `stop_ccsg_pilot`.
- Artifacts: `../artifacts/s2c/runs/protocol_v2_textoir_v1/ccsg_pilot_v1/`; lightweight summaries:
  `results/diagnostics/ccsg/`. Validation included focused CCSG tests, pilot integrity/verify, and
  Known-only split contract checks. Next step: do not tune CCSG or expand K; register a separate
  strong-baseline or representation-boundary experiment if needed.

## 2026-08-04 — RC-AMBL adaptive_v1 StackOverflow pilot

- 目标：在不重跑 E2/E3/URCSG/CCSG/BRAK/MOGB 的前提下，验证 Risk-Calibrated Adaptive Multi-centre Boundary Learning 是否能在 StackOverflow/KIR=.50 上安全地决定是否增加局部中心。
- 范围：冻结 `protocol_v2_textoir_v1`、StackOverflow、KIR=.50、正式 seed 13/42/87；每个 seed 运行 `RC-AMBL-KnownOnly` 和 `RC-AMBL-ProxyOOS`，计划 6 个评价单元，实际完成 6/6。
- 新增代码：`src/protocol_v2/experiments/adaptive_v1/`、`scripts/experiments/adaptive_v1/`、`configs/experiments/protocol_v2_textoir_v1/adaptive_v1.yaml`；旧 E2/E3/R1/BRAK/MOGB 文件未修改。
- 方法：每个 intent 从 K=1 父边界开始，使用 PCA median split、收缩 diagonal covariance、类级加权 evidence、父级边界保护、top-two margin 和 Known-only 安全门；不枚举 K，不读取 test OOS 选择中心或阈值。
- 防泄漏：训练 6000、calibration 1000（SeedSequence 的 select/threshold 两部分）、test 6000；每个 run 的 registry/view/export/cache hash 和 `test_used_for_selection=false` 均写入 provenance。
- 运行与验证：dry-run、3 seed runner、verify、summarize、diagnose、plot 均完成；专项回归测试与 E3 partition 测试通过。`ADAPTIVE_V1_VERIFY.json` 为 3/3 seed、6 行指标、status=pass。
- 结果：三次候选分裂均被拒绝，bootstrap median ARI 为 0.7051--0.7712，10 个 Known intent 最终均为 `K_y=1`；RC-AMBL 两模式 OOS F1 均值 `0.5785±0.0926`，相对 E2 K=1 下降 `19.44pp`，false acceptance 增加 `29.14pp`。
- 决策：`stop_adaptive_v1_pilot`。这是安全回退和 evidence/阈值失败诊断，不是新方法成功，不授权扩展 CLINC150、Banking77、其他 KIR、K 网格或完整 Pipeline。
- 产物：`../artifacts/s2c/runs/protocol_v2_textoir_v1/adaptive_v1/contract_repair5/`、`results/diagnostics/adaptive_v1/`、`docs/adaptive_v1/ADAPTIVE_V1_REPORT.md`、`docs/adaptive_v1/REPRODUCE_ADAPTIVE_V1.md`。
- 风险：当前 RC-AMBL evidence 阈值不是 E2 nearest-sphere 逐值复现；MOGB 主表行仍是兼容组件参考，不能称官方 BERT 复现或 SOTA。下一步只允许先做同合同 K=1 calibration control，若仍有高 false acceptance，则保留 E2 K=1 并转向已登记强基线/端到端对比。

## 2026-08-05 — RACAL-v1 第一阶段 Trainable K=1

- Base commit：记录于 ../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1/RACAL_PROVENANCE.json；第三方 third_party/mogb_official 保持只读 dirty 状态，未修改或清理。
- 目标：建立独立 RACAL-v1 框架，先精确复现 E2 StackOverflow/KIR=.50/K=1，再验证最后两层 MiniLM 加 384D 残差投影的 Known-only 单中心训练。
- 新增文件：src/protocol_v2/experiments/racal_v1/、scripts/experiments/racal_v1/、configs/experiments/protocol_v2_textoir_v1/racal_v1.yaml、tests/unit/test_racal_v1.py、tests/integration/test_racal_v1_integration.py、docs/racal_v1/；新 artifact 根为 ../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1/。
- 数据影响：只读取固定 StackOverflow train=6000、calibration=1000、test=6000；train/calibration 仅 Known；三个 split 的 sample-id 无交叉；test OOS 未参与训练或选模。
- E2 replay：seed 13/42/87 共 3/3；sample-id mismatch、prediction mismatch、score max delta 和 metric max delta 均为 0。
- Trainable K=1：3/3 checkpoint 和 Gate 结果完成；训练只包含 CE、类内中心紧致和最近异类中心 margin；最后两个 MiniLM block 与 projection 可训练，前四层冻结。
- 结果：Frozen OOS F1 0.7729±0.0417，Trainable OOS F1 0.8671±0.0079；F1-All 0.7860→0.8565；Known Recall 0.8371→0.8392；false acceptance 0.2654→0.1114。三个 seed 方向一致。
- 测试：RACAL targeted tests 4/4 通过；dry-run、checkpoint reload、E2 replay、verify 通过；compileall/ruff 已通过。数据 tracking 通过；development-log checker 仍报告预-existing third-party dirty 例外。
- 决策：Trainable K=1 满足晋级门，允许登记下一阶段 fixed K=2、proxy-OOS 和风险约束中心激活，但本批不自动启动。
- 风险：当前结果仍是单中心表示证据，不能宣称 RACAL 多中心已成功或达到 SOTA；下一阶段必须先运行 fixed K=2 对照，并继续保持 StackOverflow/KIR=.50。

## 2026-08-05 — RACAL-v1 阶段二 fixed K=2 边界对照

- Base commit/provenance：`../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1/stage2_fixed_k2/RACAL_STAGE2_PROVENANCE.json`；阶段一 checkpoint 只读复用，第三方 MOGB checkout 未修改。
- 目标：在已验证有效的 Trainable MiniLM 表示上，只比较同一 checkpoint 的 K=1 与每个 intent 内 KMeans-2；不引入 proxy-OOS、risk gate、adaptive K、阈值调参、K=3--5、其他数据集或 Cascade。
- 新增文件：`src/protocol_v2/experiments/racal_v1/stage2.py`、`scripts/experiments/racal_v1/run_racal_v1_stage2.py`、`verify_racal_v1_stage2.py`、`summarize_racal_v1_stage2.py`、`diagnose_racal_v1_stage2.py`、阶段二配置和测试；新 artifact 根为 `../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1/stage2_fixed_k2/`。
- 数据与防泄漏：StackOverflow/KIR=.50、seeds 13/42/87；每个 seed train=6000、calibration=1000、test=6000，split overlap 为 0；阶段一 sample-id、registry 和 canonical hash 全部复核；checkpoint 不重新训练；`test_used_for_selection=false`。
- 运行：dry-run、seed42 smoke、seeds13/87、verify、summarize、diagnose 完成；3/3 run、30 intent diagnostics、18,000 hashed sample-audit rows，失败/缺失/重复/无效为 0。
- 结果：Trainable K=1 OOS F1 `0.8671±0.0079`、false acceptance `0.1114±0.0165`；fixed K=2 OOS F1 `0.6765±0.0615`、false acceptance `0.4526±0.0782`；K=2 相对 K=1 OOS F1 `-19.06pp`、F1-All `-8.85pp`、Known Recall `+9.70pp`、false acceptance `+34.11pp`、AUROC `-2.30pp`。
- 逐样本：三个 seed 新增 OOS false acceptance 分别为 1169、753、1154；恢复 Known false rejection 分别为 298、309、285；净收益均为负。K=2 改变最近中心/意图的样本数为 5733、5747、5947。
- intent 诊断：30 行均记录 K=1 半径、K=2 子簇大小/半径/方差、bootstrap ARI、silhouette、Known Recall 变化和净收益。跨 seed 有正净收益 intent 16 行、负净收益 intent 14 行，但多数高风险 intent 在三个 seed 均负，说明存在意图异质性且总体风险由过覆盖主导。
- 判定：`A_primary_with_C_heterogeneity`。fixed K=2 明显退化，停止 K=3--5；不停止 RACAL，但下一阶段只能登记一次最小 risk-gated K=1→K=2 center activation，不能自动运行。
- 测试：阶段二专项测试 4/4；后续执行 full unit/integration/smoke、compileall、ruff、git diff --check、data tracking、research state、development log 和 public whitelist verify。

## 2026-08-05 — joint_adaptive_multicenter_v1 训练参与式自适应多中心 pilot

- Base commit：`dc2c0b59fa200a9590dcbebaadf6b244f53c84e2`；工作树 dirty；当前代码快照和配置哈希记录在
  `../artifacts/s2c/runs/protocol_v2_textoir_v1/joint_adaptive_multicenter_v1/repair6/JOINT_ADAPTIVE_PROVENANCE.json`。
- 目标：验证局部中心参与 MiniLM 表示训练、每个 intent 的第二中心由 Known-only calibration 决定时，能否避免
  StackOverflow 固定 K=2 的 false-acceptance 爆炸；不重跑 E2/E3/R1/RC-AMBL/RACAL/MOGB 历史实验。
- 新增代码：`src/protocol_v2/experiments/joint_adaptive_v1/`、`scripts/experiments/run_joint_adaptive_multicenter_v1.py`、
  `scripts/experiments/analyze_joint_adaptive_v1.py`、配置和专项单元测试；新 artifact 根为
  `../artifacts/s2c/runs/protocol_v2_textoir_v1/joint_adaptive_multicenter_v1/repair6/`。
- 方法：RACAL Trainable K=1 checkpoint 作为初始模型；MiniLM 最后两层、残差 projection 和 intent prototypes 共同训练；
  候选 split 由 Known train PCA 残差提出；结构选择只读 Known calibration，父边界保护不允许子中心扩大已知安全区域；
  `test_used_for_selection=false`、`oos_used_for_training=false`。
- 实验：StackOverflow/KIR=.50/seeds 13、42、87，计划 3，完成 3，失败 0；候选每 seed 实际训练一次，3/3 被
  `no_known_only_compactness_gain` 拒绝，最终 10 个 Known intent 均为 `K_y=1`。
- 结果：joint adaptive OOS F1 `0.8661±0.0111`、F1-All `0.8563±0.0050`、Known Recall `0.8388±0.0045`、
  false acceptance `0.1129±0.0228`；相对 RACAL Trainable K=1 的 OOS F1 `-0.10pp`、F1-All `-0.03pp`、Known Recall
  `-0.04pp`、false acceptance `+0.14pp`，没有安全的新增中心收益。
- 诊断：相同候选的无父边界 K=2 union 平均 OOS F1 `0.6777`、false acceptance `0.4513`；parent guard 后为
  `0.8663`、`0.1118`，说明 union 过覆盖是主要风险。后者仅为事后诊断，不是正式 adaptive 结果。
- 验证：专项测试 3/3、compileall、run summarize、verify、analysis 完成；repair1--repair5 仅保留为被修复的实现尝试，
  repair6 是当前有效合同（修复 NaN/Inf、父边界陈旧和候选学习率问题）。
- 决策：训练参与式链路已证实可运行，但当前 StackOverflow/KIR=.50 未证明 `K_y>1` 有益；停止扩大数据集、KIR、
  K=3--5 和新损失，下一步必须先登记新的 split objective/边界契约，再做同规模小 pilot。
