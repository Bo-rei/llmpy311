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

## 2026-08-05 — joint_adaptive_multicenter_contract_repair_v1 合同修复 pilot

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`；有效运行 attempt 为 `repair3`，provenance 为
  `../artifacts/s2c/runs/protocol_v2_textoir_v1/joint_adaptive_multicenter_contract_repair_v1/repair3/JOINT_ADAPTIVE_CONTRACT_REPAIR_PROVENANCE.json`。
- 目标：排除旧训练参与式 pilot 中父边界重估、unconstrained compactness 和缺少子中心负载/分离约束造成的合同混淆，
  不扩大数据集、KIR 或 K 值。
- 新增文件：`src/protocol_v2/experiments/joint_adaptive_contract_repair_v1.py`、对应 CLI、配置和单元测试；
  旧 `joint_adaptive_multicenter_v1/repair6`、E2/E3/R1/RACAL 产物未修改。
- 修复合同：K=1 父边界先拟合并冻结；候选 compactness 使用 parent-guarded score；候选训练加入子中心负载平衡
  和分离项；测试 OOS 不参与训练、选择或阈值。
- 实验规模：StackOverflow/KIR=.50，seed=13/42/87，候选每 seed 训练 2 epochs；计划 3、完成 3、失败 0，
  每个候选实际产生 checkpoint，最终 3/3 由 Known calibration Recall 安全门拒绝，`mean K_y=1.0`。
- 结果：OOS F1 `0.8661±0.0091`、F1-All `0.8563±0.0041`、Known Recall `0.8388±0.0037`、false acceptance
  `0.1129±0.0187`；与 RACAL Trainable K=1 相比没有安全多中心增益。
- 验证：contract-repair 单元测试 3/3、compileall、runner、summarize、verify 通过；repair2 汇总器字段错误已在
  repair3 修复并重新运行，repair2/repair3 均保留，不覆盖旧 artifact。
- 风险：本结果说明当前 StackOverflow 条件下候选训练仍不能通过 Known-only 安全门，不等于所有训练参与式自适应
  多中心都不可能；但在登记新目标函数前不得扩展到其他数据集、KIR、K=3--5 或完整 Pipeline。
- 下一步：停止本方向；如需继续，必须另行登记新的候选目标/分裂契约并保持同规模 pilot，或转向已登记强基线/端到端方法。

## 2026-08-05 — consistency_gate_v1 单中心一致性/证据冲突 pilot

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`；新阶段独立于 E2/E3/R1、RACAL、joint adaptive 和 contract repair。
- 目标：在训练参与式多中心候选全部回退后，验证 Trainable K=1 是否能通过多视图预测一致性和证据 margin 降低
  OOS false acceptance；不增加中心、不重新训练 encoder。
- 新增文件：`src/protocol_v2/experiments/consistency_gate_v1.py`、CLI、配置、单元测试和中文报告；新 artifact 根为
  `../artifacts/s2c/runs/protocol_v2_textoir_v1/consistency_gate_v1/`。
- 视图：原始 eval、两次固定种子 MC-dropout、NFKC/空白归一化；中心和半径仍为 RACAL Trainable K=1 的
  单中心 diagonal-Mahalanobis mean+std；冲突容忍度与 evidence margin 只使用 Known calibration 选择。
- 实验：StackOverflow/KIR=.50、seed=13/42/87，3/3 完成，失败/缺失/重复/无效为 0；test OOS 不参与选择或训练。
- 结果：Trainable K=1 OOS F1 `0.8671±0.0079`；evidence-margin `0.8673±0.0076`，F1-All `0.8580±0.0027`，
  Known Recall `0.8376±0.0020`，false acceptance `0.1099±0.0145`；combined Gate OOS F1 `0.8674±0.0073`，
  Known Recall `0.8319±0.0025`，false acceptance `0.1060±0.0161`。
- 判定：一致性/证据冲突是一个安全但收益很小的单中心拒识候选，不能称 SOTA，也不能用来证明多中心有效；不扩展
  KIR、数据集、更多视图或 K 网格。
- 测试：专项单元测试 3/3、compileall、3 seed runner、summarize、verify 通过；所有结果和配置哈希独立保存。
- 下一步：保持 Trainable K=1 为当前骨干，若继续实验先做同协议强基线/端到端对照或预注册更明确的拒识目标，
  不再无条件扩大多中心结构。

## 2026-08-06 — minilm_trainable_control_v1 跨数据集 K=1 控制

- Base commit：运行时记录于 `../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_control_v1/PROVENANCE.json`；工作树保持 dirty，未执行 commit/push。
- 目标：在不重复 E2/E3、不扩展 K 的前提下，验证 Known-only Trainable MiniLM 的 K=1 收益是否跨数据集成立。
- 新增文件：`configs/experiments/protocol_v2_textoir_v1/minilm_trainable_control_v1.yaml`、
  `src/protocol_v2/experiments/minilm_trainable_control_v1.py`、CLI、汇总脚本和中文报告；新 artifact 根为
  `../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_control_v1/`。
- 配置：CLINC150、Banking77 新运行；StackOverflow 复用 RACAL-v1 同协议 K=1；KIR=.50、seeds=13/42/87、
  Mahalanobis K=1、mean+std 半径；只训练 MiniLM 最后两层和 residual projection。
- 数据与防泄漏：train/calibration/test 三者无交叠；训练和 checkpoint 选择只访问 Known train/calibration；
  `test_used_for_selection=false`、`oos_used_for_training=false`；6/6 新单元完成。
- 结果：Trainable 相对同 seed E2 Frozen 的 OOS F1 为 CLINC `+1.12pp`、Banking `+5.18pp`、
  StackOverflow `+9.42pp`；Known Recall 为 `-1.33pp`、`-1.95pp`、`+0.21pp`。
- 结论：K=1 表示适配具有跨数据集正向 OOS 证据，但 CLINC/Banking 存在 Known Recall 代价；不支持无条件替代 Frozen，
  也不授权继续扩大 K。下一步先做类内距离/半径/校准诊断，再进入同协议基线小矩阵。
- 验证：317 个 unit tests 通过（3 warnings）、RACAL 相关 4 个 integration/unit tests 通过、ruff、compile、git diff --check、
  research-state checker 和 6 个 metrics/provenance 自检通过。
- 证据入口：`docs/analysis/MINILM_TRAINABLE_CONTROL_V1.md`、`results/diagnostics/minilm_trainable_control_v1/`、
  `figures/active_experiment_dashboard_v1/trainable_cross_dataset.png`。

## 2026-08-06 — minilm_trainable_kir_sweep_v1 跨 KIR 的 Trainable K=1 控制

- Base commit/provenance：运行时记录于 `../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_kir_sweep_v1/PROVENANCE.json`；本阶段未执行 commit/push，旧 E0--E3、R1、RACAL 与 baseline artifacts 保持只读。
- 目标：在同一 `protocol_v2_textoir_v1` 和同一对角马氏 Gate 合同下，检查最后两层 MiniLM+projection 的 Known-only 适配是否随 KIR、数据集变化；不增加 K、不重复旧多中心矩阵。
- 配置：CLINC150、Banking77、StackOverflow；KIR={.25,.50,.75}；seeds={13,42,87}；Trainable K=1；训练和 checkpoint 选择只使用 Known train/calibration，`test_used_for_selection=false`、`oos_used_for_training=false`。
- 实验数量：计划 27、完成 27、失败 0、缺失 0、重复 0、无效 0；每个 dataset×KIR 共用独立 stage root，未覆盖旧结果。
- 结果：相对同距离 Frozen K=1，CLINC150 OOS F1 增量为 `+0.64/+2.41/+3.59pp`（KIR=.25/.50/.75），StackOverflow 为 `+1.19/+7.69/+13.54pp`；Banking77 为 `+0.59/-0.05/-14.02pp`。Banking77 在 KIR=.75 的 Known Recall 下降约 `7.28pp`，说明 OOS F1 不能脱离 Known 指标解释。
- 判定：Trainable MiniLM 的可重复收益主要属于单中心表示/分数排序；它不是统一 SOTA，也不证明固定多中心安全。StackOverflow 固定 K=2 的 union-risk 仍由既有配对控制和 λ 诊断支持。
- 新增：`configs/experiments/protocol_v2_textoir_v1/minilm_trainable_kir_sweep_v1.yaml`、`src/protocol_v2/experiments/minilm_trainable_kir_sweep_v1.py`、runner、分析脚本、`docs/analysis/MINILM_TRAINABLE_KIR_SWEEP_V1.md`、`results/analysis/minilm_trainable_kir_sweep_v1/`、`figures/minilm_trainable_kir_sweep_v1/`。
- 风险：Trainable/Frozen 的 KIR sweep 是同距离 Gate-only 对照，不能直接与 fulltex.tex 历史 Cascade 表或使用不同监督/表示的 MOGB、ADB、DA-ADB、DCLOOS 结果混为公平排名。
- 下一步：先将本轮结果加入分层总表并完成研究状态/台账校验，再决定是否做正式同协议强 baseline/端到端条件对照；不自动扩展 R1_full、K=2--5 或完整 Pipeline。

## 2026-08-06 — minilm_trainable_k2_control_v1 K=1/K=2 跨数据集配对控制

- Base commit：运行时写入 `../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_k2_control_v1/PROVENANCE.json`；工作树保持 dirty，未执行 commit/push。
- 目标：在同一 Trainable MiniLM checkpoint、同一 split、同一距离/半径/阈值合同下，判断表示适配是否能使固定 K=2 安全；不重新训练、不扩展 KIR、不覆盖历史结果。
- 新增文件：`configs/experiments/protocol_v2_textoir_v1/minilm_trainable_k2_control_v1.yaml`、`src/protocol_v2/experiments/minilm_trainable_k2_control_v1.py`、`scripts/experiments/run_minilm_trainable_k2_control_v1.py`、`tools/analysis/build_minilm_k_interaction_report.py`；独立 artifact 根为 `../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_k2_control_v1/`。
- 实验：CLINC150/Banking77 × seeds 13/42/87 新增 6/6 个评价单元；StackOverflow 合并既有 RACAL-v1 Stage2 的 3 个只读配对行，最终跨数据集 9 行；每个 seed 的 K=1/K=2 共用同一已冻结 checkpoint。
- 数据与防泄漏：校验 E2 manifest、registry/canonical hash、train/calibration/test sample-id hash；train/calibration 只含 Known；`test_used_for_selection=false`、`oos_used_for_training=false`；CLINC/Banking 的 K=1 replay 最大差为 0。
- 结果：K=2−K=1 OOS F1 为 CLINC `-0.28pp`、Banking `+0.13pp`、StackOverflow `-19.06pp`；Known Recall 分别 `-3.26pp`、`-1.95pp`、`+9.70pp`；StackOverflow false acceptance `+34.11pp`。
- 判定：Trainable MiniLM 的 K=1 收益不能被解释为固定多中心普遍恢复；Banking77 仅呈条件性微小收益，CLINC150 更支持 K=1，StackOverflow 的并集过覆盖仍是结构性失败。
- 输出：`results/diagnostics/minilm_trainable_k2_control_v1/{per_seed.csv,mean_std.csv,delta_summary.csv}`、`docs/analysis/MINILM_TRAINABLE_K2_CONTROL_V1.md`、`figures/active_experiment_dashboard_v1/trainable_k_interaction_cross_dataset.png`。
- 验证：新模块 compile、ruff、dry-run、6/6 runner、汇总脚本和图形生成通过；历史 E2/E3/R1/RACAL artifacts 未修改。
- 下一步：保持 Trainable K=1 为当前可复用表示基线；先做同协议的表示/阈值与强 baseline 对照，不自动扩展固定 K=2 或更大 K。

## 2026-08-06 — minilm_trainable_lambda_control_v1 Known-only λ/K 交互控制

- Base commit/provenance：运行时记录于 `../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_lambda_control_v1/PROVENANCE.json`；本阶段不执行 commit/push，历史 E2/E3/R1/RACAL 产物保持只读。
- 目标：在同一 Trainable MiniLM checkpoint 上只改变 `lambda`，判断 K=2 退化是否仅由 `mean+lambda*std` 的半径系数造成；不得使用 test OOS 选择 λ、K 或 checkpoint。
- 范围：CLINC150、Banking77、StackOverflow；KIR=.50；seeds=13/42/87；K=1/2；λ={.50,.75,1.00,1.25,1.50,2.00}；Mahalanobis diagonal、mean_std、threshold=1.0；共 108/108 个评价单元完成，失败/缺失/重复/无效为 0。
- 选择合同：每个 dataset×seed×K 选择 Known calibration false-reject rate≤5% 的最小 λ；若候选网格无可行值则记录 `selection_constraint_met=false` 并保留最大 λ 作为诊断，不把它写成正式最优参数。test OOS 只在选择规则冻结后评价。
- 结果：Known-only 选择后 K=2−K=1 OOS F1 为 CLINC `+0.79pp`、Banking `+2.08pp`、StackOverflow `-9.51pp`；StackOverflow false acceptance 仍增加 `+11.83pp`。StackOverflow K=2 的退化因此不是 λ=1 单点设置造成；K=1 的 5% Known calibration 约束在三 seed 上也未稳定可行。
- 新增文件：`src/protocol_v2/experiments/minilm_trainable_lambda_control_v1.py`、CLI、配置、汇总器、`docs/analysis/MINILM_TRAINABLE_LAMBDA_CONTROL_V1.md`、`results/diagnostics/minilm_trainable_lambda_control_v1/` 和 `figures/active_experiment_dashboard_v1/trainable_lambda_k_interaction.png`。
- 决策：半径 λ 校准不能单独救活 StackOverflow 固定 K=2；停止 lambda-only rescue，保留 Trainable K=1 作为当前表示基线。下一步应做同协议强基线/端到端条件对照或明确新的开放风险目标，不再扩大 K 或新增 MiniLM 损失。
- 验证：lambda runner 108/108、汇总和图形生成通过；下一步需运行 research-state、development-log、compile/ruff 和结果完整性检查。

## 2026-08-06 — KIR=0.50 方法协议分层分析（analysis-only）

- 目标：把当前 Trainable/Frozen/MOGB fair 组件与 ADB、DA-ADB、BRAK 兼容结果放入同一份可追溯表，但不把不同表示、监督条件和 seed 数混成 SOTA 排名。
- 输入：`results/final_baselines/summary.csv`、`results/mogb/fair_matrix.csv`、`results/diagnostics/minilm_trainable_k2_control_v1/per_seed.csv`；未读取原始文本、未重跑实验。
- 输出：`docs/analysis/KIR50_METHOD_COMPARISON_V1.md`、`results/analysis/kir50_method_comparison_v1/{rows.csv,mean_std.csv}`、`figures/active_experiment_dashboard_v1/{kir50_method_layers.png,kir50_method_tradeoff.png}`。
- 结论：StackOverflow 当前协议 Trainable K=1 为 86.71%，高于 Frozen Single centroid 76.55%、MOGB-MiniLM 72.92% 和 MOGB partition+s2c boundary 79.25%；ADB/DA-ADB 为 89.47%/90.90%，但属于 BERT/不同合同兼容单格，不能直接宣称公平超越或落后。
- 验证：分析脚本 compile、ruff、图形生成和 dashboard 重建通过；历史 artifacts 未覆盖。

## 2026-08-06 — experiment_evidence_pack_v2 多 KIR/多方法统计与可视化

- 目标：把已完成的 MOGB fair matrix、固定 K 组件和 Trainable K=1 KIR sweep 汇总为可比较但不混协议的实验包；不重跑训练，不修改历史 artifacts。
- 输入：`results/mogb/fair_matrix.csv`（3 数据集×3 KIR×5 seed×6 frozen-component 方法）、`results/analysis/minilm_trainable_kir_sweep_v1/mean_std.csv`；所有 rows 保留 `scope`/representation/supervision 信息。
- 分析：以 Single centroid 为 paired reference，计算 OOS F1、F1-All、Known Recall、false acceptance 的均值差、95% paired bootstrap CI（seed=20260725、10,000 次）和 win/tie/loss。
- 输出：`tools/analysis/build_experiment_evidence_pack_v2.py`、`results/analysis/experiment_evidence_pack_v2/`（54 行 fair summary、180 行 paired effects、manifest）、`figures/experiment_evidence_pack_v2/`（21 张 PNG）、`docs/analysis/EXPERIMENT_EVIDENCE_PACK_V2.md`。
- 结论：MOGB partition+s2c boundary 在部分 KIR 提高 OOS F1，但通常伴随 25--37pp Known Recall 损失；Trainable K=1 在 CLINC150/StackOverflow 的 KIR 曲线整体更平衡，Banking77 的高 KIR 仍退化。当前自有优势应表述为 Known-only 单中心表示适配的平衡，而不是已证明自适应多中心普遍超过 MOGB。
- 验证：分析脚本 compile、ruff、运行、图像人工检查、研究状态检查、开发日志检查、数据跟踪检查和 `git diff --check` 均通过；图表使用英文标题避免缺失中文字体，中文解释保存在 Markdown。

## 2026-08-06 — 表示训练与 StackOverflow 按意图机制分析（analysis-only）

- Base commit：沿用当前工作树；未执行 commit、push 或历史 artifact 覆盖。
- 目标：将已有 Frozen/CE/SupCon 的 K=1/K=2 结果与 RACAL Stage-2 的 StackOverflow intent diagnostics
  组织成可读的机制证据，解释“表示训练改善 K=1、但固定多中心仍可能误接收 OOS”。
- 新增：`tools/analysis/build_representation_boundary_pack_v1.py` 的输出说明文档
  `docs/analysis/REPRESENTATION_BOUNDARY_PACK_V1.md`；新增脚本
  `tools/analysis/build_stackoverflow_intent_diagnostic_v1.py` 及其结果、报告和 4 张图。
- 输入与范围：表示包 18 行汇总、36 行 K=2−K=1 效应、已有几何汇总；StackOverflow 诊断包读取 30 个
  intent-seed 行，仅做聚合和 Spearman 探索性分析，不读取测试文本、不重新训练、不用 test OOS 选参。
- 结果：StackOverflow CE 的 OOS F1 由 88.13% 降至 73.44%，SupCon 由 89.63% 降至 71.90%，Known
  Recall 分别上升 8.12pp/9.73pp；诊断样本平均恢复 29.73 个 Known、却新增接受 102.53 个 OOS，
  平均 ARI 0.91。结论是稳定性不能替代 OOS 风险校准。
- 验证：脚本运行成功，输出 4 张意图诊断图；随后运行 compile/ruff、CSV 解析、研究状态、开发日志、
  数据跟踪和 git diff 检查。
- 风险：intent diagnostics 不是全部 20 个 StackOverflow 意图，near-OOS 分桶仍是机制诊断口径；不能
  将分析包写成新的 SOTA 结果或自适应 K 成功证据。
- 下一步：继续在已有结果上补充强基线/表示—边界分层分析，若启动新训练必须先单独登记并避免重复 E2/E3/RACAL。

## 2026-08-06 — 意图级 KIR 稳定性与多中心候选分析（analysis-only）

- 目标：利用已完成的 intent-level K/KIR 审计，分析多中心候选是否集中在特定数据集、KIR、距离或意图，
  不重新训练、不用 oracle best-K 作为正式选择。
- 输入：`results/diagnostics/adaptive_k/intent_level.csv`，共 13,580 行；输出 66 个
  dataset×KIR×distance 汇总、4 张图、4,646 个 intent-group seed 稳定性记录。
- 结果：oracle 口径下 K>1 候选比例 Banking77 65.3%、CLINC150 45.2%、StackOverflow 34.2%；同时满足
  OOS F1 正增益且 Known Recall 降幅不超过 1pp 的比例分别为 39.8%、35.7%、24.7%。StackOverflow
  平均 oracle OOS F1 增益约 0.99pp，远低于 Banking77 的 5.80pp。
- 风险：这些比例依赖 test-sensitivity/oracle 审计；它们只能说明意图异质性和研究空间，不能作为无泄漏
  adaptive-K 方法或 SOTA 证据。
- 验证：分析脚本运行成功，输出文件和图形存在；后续运行 compile、ruff、research-state、devlog、data tracking
  和 git diff 检查。
- 下一步：用这些候选区间指导同协议 Known-only 可靠性特征分析，不再根据 oracle 结果直接挑 K。

## 2026-08-06 — MiniLM Trainable K=1 五 seed 公平扩展与报告修正

- Base commit：运行时记录在 `../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_kir_sweep_extension_v1/PROVENANCE.json`；工作树保持 dirty，未执行 `git add/commit/push`。
- 目标：补齐现有 Trainable K=1 KIR sweep 的 seed=100/123，并用同一 E2 Frozen K=1 做逐 seed 配对，回答当前可训练 MiniLM 是否在五个正式 seed 下稳定改善单中心 Gate。
- 新增/修改：`scripts/experiments/run_minilm_trainable_kir_sweep_extension_v1.py`、扩展配置与 artifact；`tools/analysis/build_minilm_trainable_5seed_fair_report_v1.py`；五 seed 中文报告、CSV 和四张图；研究状态、实验总账与本日志。
- 实验：新增 18/18 个训练/评价单元；与原有 27 个单元合计 45/45；三个数据集、KIR=.25/.50/.75、seeds=13/42/87/100/123、Trainable K=1、diagonal Mahalanobis、mean+std、threshold=1；失败/缺失/重复/无效为 0。
- 数据与防泄漏：训练和 checkpoint 选择只使用 Known train/calibration；test OOS 不参与训练、epoch、半径、阈值或 K 选择；Frozen 逐 seed 读取相同 E2 单元；旧 E2/E3/R1/RACAL/MOGB 产物未覆盖。
- 结果：Trainable−Frozen OOS F1（KIR=.25/.50/.75）为 CLINC150 `+0.45/+1.12/+1.38pp`、Banking77 `+2.47/+4.72/+6.94pp`、StackOverflow `+5.06/+9.55/+10.50pp`；Known Recall 最大下降分别为 1.37pp、2.90pp，StackOverflow 小幅上升。固定 K=2 的安全性未因此改变。
- 报告修正：发现历史 `fulltex.tex` 参考行被误拼入 paired group mean，已修正为先完成 Trainable/Frozen 配对、再单独写历史参照；修正后的配对 CSV 与报告差值已复核。
- 验证：扩展 runner 18/18；分析脚本；后续运行 compile、ruff、CSV/ledger 解析、研究状态、开发日志、数据跟踪和 `git diff --check`。
- 风险：Trainable 是当前 Gate-only K=1 证据，不是 SOTA；MOGB 行是组件/监督条件分层上下文，fulltex 是历史 Cascade 参照，均不得合并为无条件排名。
- 下一步：完成状态校验后，优先做 Frozen/Trainable 的类内距离、半径分布和 calibration coverage 分析，再决定是否进行同协议强 baseline 小矩阵；不自动扩展 K=3--5 或完整 Pipeline。

## 2026-08-06 — MiniLM 表示—边界 score 诊断（analysis-only）

- Base commit：沿用当前工作树；未执行 commit/push，训练 artifact 与历史结果保持只读。
- 目标：解释 Trainable K=1 为什么能改善当前协议，而不把这种收益误写成固定多中心或历史 Cascade 的收益。
- 输入：五 seed Trainable K=1 与同一 E2 Frozen K=1 的 predictions/metrics；范围为三数据集、KIR=.25/.50/.75、seeds=13/42/87/100/123；不重新训练、不选择阈值、checkpoint、λ 或 K。
- 处理：读取 443,400 条 test score 记录，仅在内存按 Known/OOS 分组，导出 aggregate score quantiles、median score gap、assigned-radius gap、false acceptance/rejection 和 KIR 曲线；不输出原始文本。
- 结果：KIR=.50 的 median OOS−Known score gap Frozen→Trainable 为 CLINC `0.258→0.438`、Banking `0.259→0.334`、StackOverflow `0.116→0.434`；false acceptance 分别下降 `2.88pp`、`9.14pp`、`15.69pp`。
- 判定：Trainable 的直接收益是单中心表示/score 排序分离；它没有证明固定 K>1 安全，也没有消除 fulltex 的协议差异。
- 输出：`docs/analysis/MINILM_BOUNDARY_DIAGNOSTICS_V1.md`、`results/analysis/minilm_boundary_diagnostics_v1/`、`figures/minilm_boundary_diagnostics_v1/`、`tools/analysis/build_minilm_boundary_diagnostics_v1.py`。
- 验证：脚本运行完成 90 个 run summary、3 张图；ruff（新脚本）、py_compile、CSV 解析、research-state、development-log、data-tracking 和 `git diff --check` 通过。
- 风险与下一步：该诊断使用 test 结果做事后解释，不能作为新参数选择；下一步优先做 Known calibration coverage/半径稳定性和强基线协议分层，不盲目扩展 K。

## 2026-08-06 — MiniLM 训练动态与选模错位诊断（analysis-only）

- Base commit：沿用当前工作树；未执行 commit/push，既有训练 artifact 保持只读。
- 目标：检查 Known-only checkpoint 选择目标是否足以解释当前 Trainable 与历史 fulltex 的差距，避免在没有证据时盲目增加 epoch。
- 输入：45 个 Trainable K=1 run 的 `training_history.tsv`、`training_manifest.json`、`metrics.json`；三数据集、KIR=.25/.50/.75、五 seed。
- 结果：179 条 epoch 记录，最佳 epoch 主要集中在 3--4；选择目标为 `calibration F1-K + 0.05×Known Recall`。KIR 增大时 Known calibration 仍可稳定选模，但测试 OOS F1 在 Banking77/StackOverflow 下降。
- 判定：当前瓶颈不是简单训练不足，而是 Known-only 表示目标与 OOS 边界风险不完全对齐；这与 score 分布诊断和固定 K=2 union-risk 结论一致。
- 输出：`docs/analysis/MINILM_TRAINING_DYNAMICS_V1.md`、`results/analysis/minilm_training_dynamics_v1/`、`figures/minilm_training_dynamics_v1/`、`tools/analysis/build_minilm_training_dynamics_v1.py`。
- 验证：45 run/179 history 读取完成、3 张图生成；新脚本 ruff、py_compile、CSV 解析和研究状态检查通过。
- 下一步：优先做 Known calibration coverage/半径稳定性与同协议强 baseline 分层，不用 test OOS 改 epoch 或超参数。

## 2026-08-06 — Trainable 与 MOGB 组件五 seed 配对对比（analysis-only）

- 目标：把“自己的方法更好”拆成可核验的 OOS F1、F1-All、Known Recall 和 false acceptance 权衡，避免只按单一 OOS 指标排名。
- 输入：45 个 Trainable K=1 行、135 个 Frozen MiniLM MOGB/fixed-K 组件行；按 dataset×KIR×seed 配对；不重训、不改变 MOGB artifact。
- 结果：KIR=.50 时，Trainable 相对 MOGB partition+s2c boundary 的 OOS F1 提升 CLINC/Banking/StackOverflow `+4.88/+4.17/+8.42pp`，Known Recall 提升 `+21.10/+30.03/+33.51pp`，但 false acceptance 更高；MOGB 组件更保守，代价是大量 Known false rejection。
- 输出：`docs/analysis/TRAINABLE_VS_MOGB_COMPONENT_V1.md`、`results/analysis/trainable_vs_mogb_component_v1/`、`figures/trainable_vs_mogb_component_v1/`、`tools/analysis/build_trainable_vs_mogb_component_v1.py`。
- 统计：每个 dataset×KIR×baseline×metric 使用固定 RNG=20260725 的 10,000 次 paired bootstrap，并记录 win/tie/loss 和 effect size。
- 风险：MOGB 行是 Frozen MiniLM 组件适配，不是官方 BERT 完整复现；结果只证明工作点差异，不能写成无条件 SOTA。
- 验证：新脚本 ruff、运行、CSV 解析、三张图生成、research-state/devlog/data-tracking/git diff 检查通过。

## 2026-08-06 — 中文实验结果统一索引

- 目标：把当前大量实验、方法定义、基线协议差异和可视化证据集中到一个读者入口，避免后续将 Gate-only、完整 Cascade、MOGB 组件和 DCLOOS 外部 OOS 监督混成排名。
- 新增：`docs/analysis/EXPERIMENTAL_EVIDENCE_INDEX_V1.md`；并在 `CURRENT_STATUS.md`、`EXPERIMENTS.md` 增加入口。
- 内容：E2/E3、RACAL/Trainable、score/训练动态诊断、MOGB 组件配对、官方 MOGB/ADB/DA-ADB/DCLOOS 状态、fulltex 历史差异和全部主要图表路径。
- 约束：索引是组织证据的文档，不新增经验结果、不改变 artifact、不执行训练。

## 2026-08-06 — Trainable calibration coverage transfer 补充诊断

- 在训练动态诊断中增加 calibration Known Recall→test Known Recall 图，仍只读取现有 history/metrics，不重训、不改选模。
- KIR=.50 的平均转移差异为 CLINC150 `-0.64pp`、Banking77 `-0.21pp`、StackOverflow `+0.33pp`；因此高 KIR 的 OOS F1 下降不能简单归因为 Known coverage 崩溃。
- 新增图：`figures/minilm_training_dynamics_v1/calibration_vs_test_known_recall.png`；报告与 CSV 入口不变。

## 2026-08-06 — MiniLM score 标度与半径稳定性诊断（analysis-only）

- Base commit：沿用当前工作树；未执行 git add/commit/push，已有训练和历史 artifact 保持只读。
- 目标：解释 Trainable K=1 与 `fulltex.tex` 历史 Cascade 的差距是否主要来自 score 标度、固定 threshold 或半径估计，而不是继续训练或新增多中心。
- 新增：`tools/analysis/build_threshold_radius_stability_v1.py`；输出 `docs/analysis/THRESHOLD_RADIUS_STABILITY_V1.md`、
  `results/analysis/threshold_radius_stability_v1/` 和 `figures/threshold_radius_stability_v1/`。
- 处理：读取 45 个 Trainable 与 45 个 Frozen K=1 run，生成 810 条阈值敏感性行和 90 条半径稳定性行；阈值网格只作 test score 的事后诊断，不用于选择正式阈值、不修改 run。
- 结果：KIR=.50 的诊断性最佳 threshold（Frozen/Trainable）为 CLINC150 `1.00/1.05`、Banking77 `0.90/0.95`、StackOverflow `0.95/0.95`；半径 CV 约为 0.02--0.04，未显示 K=1 半径完全失稳。
- 判定：Trainable 已改善 K=1 score 分离，但仍是 Gate-only、固定 threshold=1、Known-only 选模；fulltex 是包含 Router/Expert、不同表示、K=2 和历史 OOS 校准合同的 Cascade，不能直接比较或归因于 MiniLM 训练不足。
- 验证：新脚本 ruff、运行、输出数量、图形生成和 CSV 解析通过；后续将运行 research-state、development-log、data-tracking 与 git diff 检查。
- 风险：诊断性 threshold 不能写成正式调参结果；未启动新训练、外部 baseline 或完整 Pipeline。
- 下一步：在保持正式 threshold=1 和协议冻结的前提下，优先把 score/边界差距与 Gate-only/Cascade/MOGB supervision 条件分层记录，不用 test oracle 追求历史数字。

## 2026-08-06 — 同协议方法权衡与可视化（analysis-only）

- Base commit：沿用当前工作树；未执行 git add/commit/push，已有训练和历史 artifact 保持只读。
- 目标：用统一的覆盖—拒识视角解释 Trainable K=1 为什么在当前协议下通常比 Frozen/MOGB 组件更平衡，而不是只看 OOS F1 排名。
- 新增：`tools/analysis/build_cross_protocol_tradeoff_v1.py`；输出 `docs/analysis/CROSS_PROTOCOL_TRADEOFF_V1.md`、
  `results/analysis/cross_protocol_tradeoff_v1/` 和 `figures/cross_protocol_tradeoff_v1/`。
- 处理：读取 315 个已完成五 seed 行，规范化 `known_recall/id_recall` 字段，生成 63 个 summary、486 个 paired bootstrap effects（RNG=20260725，10000 次）和 4 张图；历史 fulltex、官方 BERT MOGB 和 DCLOOS 外部 OOS 监督不进入 fair CSV。
- 结果：MOGB 风格组件通常具有更低 false acceptance，但伴随大量 Known false rejection；Trainable K=1 保留更高 Known Recall/F1-All。StackOverflow 固定 K=2 同时出现低 OOS F1 与高 false acceptance，支持 union-risk 解释。
- 验证：脚本 ruff、运行、图像人工检查、CSV 解析通过；后续将运行研究状态、开发日志、数据跟踪和 git diff 检查。
- 风险：`all_methods_per_seed.csv` 中 Frozen 组件使用 Euclidean/mean-radius，不能与 E2 Mahalanobis Frozen 结果混称；报告已明确标注该协议差异。
- 下一步：继续按监督条件和系统层级整理强基线小矩阵，不把 fair component 结果升级为官方 MOGB 或 DCLOOS SOTA 结论。

## 2026-08-06 — Gate→Cascade 桥接与误差分解（analysis-only）

- Base commit：沿用当前工作树；未执行 git add/commit/push，已有训练和历史 artifact 保持只读。
- 目标：验证当前 Trainable Gate-only 与系统级 Cascade 的差异是否来自 Router/Expert 和拒识错误传播，而不是把 fulltex 差距归因于一个 MiniLM checkpoint。
- 新增：`tools/analysis/build_gate_cascade_bridge_v1.py`；输出 `docs/analysis/GATE_CASCADE_BRIDGE_V1.md`、
  `results/analysis/gate_cascade_bridge_v1/` 和 `figures/gate_cascade_bridge_v1/`。
- 处理：读取当前 protocol 的 3-seed Cascade 变体和 3-seed Trainable K=1 Gate-only，共 45 行；只直接比较共享 OOS/ID 指标，Cascade 的 Accuracy/F1-K 保留为系统内部指标。
- 结果：KIR=.50 时，Trainable Gate-only OOS F1 为 CLINC/Banking/StackOverflow `90.43/84.77/86.71`，CE-Recon selected-K Cascade 为 `90.00/89.48/87.62`；误差图显示后续 Expert error 与 Gate false-reject/accept 会共同改变端到端结果。
- 验证：脚本 ruff、运行、输出数量和图形生成通过；后续运行研究状态、开发日志、数据跟踪和 git diff 检查。
- 风险：3-seed 当前 Cascade 不是历史 fulltex 主表，不能混成 SOTA 排名；该阶段只用于机制桥接。
- 下一步：继续补充统一监督/系统层级的强基线小矩阵，优先用误差分解而不是再加未注册的模型损失。

## 2026-08-06 — 原生 Frozen MiniLM baseline 矩阵（analysis/controlled baseline）

- Base commit：沿用当前工作树；未执行 git add/commit/push，历史 E2/E3/R1 artifact 只读。
- 目标：在同一 `protocol_v2_textoir_v1`、registry、Known-only calibration 和冻结 MiniLM 下补齐 MSP、Energy、kNN、LOF 原生控制，区分表示排序与阈值工作点。
- 配置：`configs/experiments/protocol_v2_textoir_v1/native_baselines_v1.yaml`；运行根为 `../artifacts/s2c/runs/protocol_v2_textoir_v1/native_baselines_v1/`。
- 数量：180/180，0 failed/blocked/unsupported；同一 dataset×KIR×seed 的方法复用一份 embedding。
- 结果：`docs/analysis/NATIVE_BASELINES_V1.md`、`results/analysis/native_baselines_v1/`；原生方法默认 α=.05 保持约95% Known Recall，Trainable 的阈值工作点更激进，故同时生成 matched-recall 诊断而不修改正式结果。
- 验证：外部 baseline 单测 5 passed，脚本 ruff/compile、dry-run、manifest 完整性和图形生成通过。
- 风险：这些方法不是官方 ADB/DA-ADB/MOGB/DCLOOS；不能与历史 fulltex 系统级数字直接排名。

## 2026-08-06 — matched Known Recall 工作点诊断（analysis-only）

- 目标：检验 Trainable 的 OOS F1 增益是否仅来自较低 Known Recall，而非 score ranking 改善。
- 输出：`docs/analysis/OPERATING_POINT_DIAGNOSTIC_V1.md`、`results/analysis/operating_point_diagnostic_v1/`、`figures/operating_point_diagnostic_v1/`；900 个回顾性 target-recall 行、3 张图。
- 处理：仅在诊断中用 test labels 对齐目标 Known Recall；不改正式 threshold，不重跑训练，不覆盖任何 baseline artifact。
- 结论：KIR=.50、名义 Known Recall=.85 时，Trainable OOS F1 仍高于可用原生 controls；但这是 post-hoc 工作点证据，不能当作无泄漏主结果。
- 下一步：继续把监督条件、工作点和历史 Cascade 分层，之后再决定是否需要统一协议的端到端闭环。

## 2026-08-06 — Trainable 与 MOGB 组件归因（analysis-only）

- Base commit：沿用当前工作树；不修改、不覆盖既有 MOGB 或 Trainable 运行。
- 目标：用相同 dataset×KIR×seed 配对，分离 MOGB 动态粒球、距离函数和半径规则的贡献，并解释 Trainable K=1 的覆盖—拒识优势。
- 输入：45 个五 seed Trainable K=1 行和 180 个 MOGB frozen-MiniLM boundary component 行。
- 输出：`docs/analysis/TRAINABLE_VS_MOGB_ABLATION_V1.md`、`results/analysis/trainable_vs_mogb_ablation_v1/`、`figures/trainable_vs_mogb_ablation_v1/`。
- 结果：324 个 paired bootstrap effects、180 个机制归因行、3 张图；KIR=.50 时 Trainable 相对 MOGB partition+s2c boundary 的 OOS F1 增量为 `+4.88/+4.17/+8.42pp`，Known Recall 增量为 `+21.10/+30.03/+33.51pp`（CLINC/Banking/StackOverflow）。
- 验证：脚本 ruff、compile 和图形生成通过；没有训练、没有 test 选择、没有官方 MOGB 复现声明。
- 下一步：继续维护同协议对比和误差分解，不把该组件分析改写为完整 MOGB 或无条件 SOTA。

## 2026-08-06 — Trainable 表示上的原生 detector 归因

- Base commit：沿用当前工作树；未执行 git add/commit/push，E2/E3/R1、Frozen native 和 MOGB artifact 只读。
- 目标：区分 Trainable MiniLM 的表示收益与当前单中心 Gate 几何/校准收益。
- 新增：`src/protocol_v2/experiments/trainable_native_baselines_v1.py`、`configs/experiments/protocol_v2_textoir_v1/native_baselines_trainable_v1.yaml`、`...native_baselines_trainable_kir50_v1.yaml`。
- 运行：StackOverflow smoke 后完成 3 数据集×KIR=.50×3 seeds×4 native detectors，共 36/36；复用每个 dataset×seed 的 Trainable checkpoint，没有新增训练。
- 输出：`../artifacts/s2c/runs/protocol_v2_textoir_v1/native_baselines_trainable_v1/`、`results/analysis/native_baselines_trainable_v1/`、`figures/native_baselines_trainable_v1/`、`docs/analysis/NATIVE_BASELINES_TRAINABLE_V1.md`。
- 统计：生成 72 个 paired effect 行（Trainable native vs Gate、Trainable native vs Frozen native），bootstrap RNG=20260725、10000 次；测试 OOS 只用于最终指标。
- 验证：ruff、compileall、smoke、36 单元、聚合脚本和图表生成通过；待补 research-state、development-log、data-tracking、registry audit、git diff check。
- 风险：该阶段仍是 KIR=.50/3 seed 归因实验，不代表完整 KIR/5 seed；native detector 不是官方 ADB/DA-ADB/MOGB/DCLOOS。
- 下一步：把该结果接入统一 evidence index，并继续以同一工作点/错误分解比较当前方法和外部基线。

## 2026-08-06 — 现有结果机制分析包 V3

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`；未执行 git add/commit/push。
- 目标：在不重新训练、不读取大型 artifact 的前提下，用已落盘的五 seed 轻量 CSV 继续做大量对比、配对统计和可视化，集中回答“Trainable 的优势来自哪里、为什么多中心仍失败”。
- 新增：`tools/analysis/build_experimental_mechanism_pack_v3.py`、`results/analysis/experimental_mechanism_pack_v3/`、`figures/experimental_mechanism_pack_v3/`、`docs/analysis/EXPERIMENTAL_MECHANISM_PACK_V3.md`。
- 输入：`all_methods_per_seed.csv` 315 行；Trainable K=1、Frozen 单/双中心、random partition 和三种 MOGB 组件按 dataset×KIR×seed 配对。
- 输出：63 个均值/标准差单元、324 个 paired bootstrap effects、63 个 Pareto 标记、4 张图；bootstrap seed=20260806、10000 次；不使用 test OOS 选参数。
- 结果：Trainable K=1 在当前同协议中通常提高 OOS F1 和/或降低 false acceptance；StackOverflow 的 fixed K>1/MOGB 组件仍显示覆盖区域组合风险，不能据此宣称自适应多中心成功。
- 验证：脚本运行成功，315/315 输入行和 7 种方法校验通过；ruff、compileall 和图像人工检查通过。
- 数据影响：无 canonical、registry、view、export 或模型数据修改；无原始文本、embedding、checkpoint 写入 Git。
- 风险：当前工作区已不存在 `../artifacts/s2c/runs/`，故本轮是轻量结果复核，不是可重跑的新训练；状态文档已明确此风险。
- 下一步：恢复并审计原始 run/checkpoint 后，才运行同 Known-only 工作点的新实验证据；外部基线继续按监督条件分层。

## 2026-08-08 — 补充实验机制包 V3 的可视化解读

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`；未执行 git add/commit/push。
- 目标：补充已有四张 V3 图的研究解释，区分“生成图表”和“完成可视化分析”。
- 修改：`docs/analysis/EXPERIMENTAL_MECHANISM_PACK_V3.md`、本日志。
- 数据影响：无；只读取既有 315 行轻量五 seed CSV，不读取原始文本、embedding 或 checkpoint。
- 关键解读：Trainable K=1 的收益主要表现为 KIR×数据集上的分数分离和覆盖—拒识平衡；固定 K=2 在 StackOverflow 的低 OOS F1/高 false acceptance 可由权衡图直接观察；MOGB 组件更保守但 Known Recall 明显较低；fulltex 对照图仅为协议不匹配的描述性参照。
- 测试：文档更新后运行研究状态、开发日志和 git diff 检查。
- 风险：当前 `../artifacts/s2c/runs` 与 checkpoint 仍缺失，可视化分析基于轻量 CSV，不能替代可重跑训练证据。
- 下一步：恢复或重新登记冻结的原始产物后，做同合同 Gate/Cascade 工作点对照；不因图表解读而新增 K 或多中心矩阵。

## 2026-08-08 — 更正原始产物状态核查

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`。
- 发现：重新核对当前 checkout 后，`../artifacts/s2c/runs/`、`../artifacts/s2c/cache/` 以及 E2、RACAL、Trainable control 目录实际存在；此前文档中的“原始产物不存在”提示已过时。
- 处理：更新 `docs/CURRENT_STATUS.md` 和 `docs/analysis/EXPERIMENTAL_MECHANISM_PACK_V3.md`，改为“目录存在但需逐阶段 provenance/manifest 验证”。
- 数据/实验影响：无；没有运行训练、没有修改或覆盖任何 artifact。
- 验证：通过目录/文件检查；后续继续运行 research-state、development-log 和 git diff 检查。
- 风险：并非所有阶段都已证明 checkpoint、registry 和 manifest 完整，不能仅凭目录存在启动新矩阵。
- 下一步：先对目标阶段执行 provenance 完整性审计，再决定是否补跑统一工作点实验。

## 2026-08-08 — 原始产物复核与主动实验面板刷新

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`；未执行 git add/commit/push。
- 目标：在发现原始运行目录实际存在后，纠正状态文档并重新读取已有输入刷新可视化面板。
- 执行：运行 `tools/analysis/build_active_experiment_dashboard_v1.py`；未启动训练，未新增实验单元。
- 结果：E2 目录可见 1,650 个 run；RACAL verifier 6/6 通过；面板刷新 12 张图和 `DASHBOARD_MANIFEST.json`，输入源 SHA256 已记录。
- 修改文件：`docs/CURRENT_STATUS.md`、本日志；面板输出仍位于既有 `results/analysis/active_experiment_dashboard_v1/` 与 `figures/active_experiment_dashboard_v1/`。
- 风险：目录存在不等于每个阶段可重跑；后续扩展实验仍需逐阶段核对 provenance、checkpoint hash 和 registry manifest。
- 下一步：用已存在 raw predictions 做同合同误差分解/可视化复核，再决定是否补跑统一 Gate/Cascade 工作点。

## 2026-08-08 — 原始逐样本 Gate 误差链与可视化（analysis-only）

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`；未执行 git add/commit/push。
- 目标：将“Trainable K=1 为什么优于固定多中心、StackOverflow K=2 为什么误接收爆炸、MOGB fair 为什么更保守”落实为 sample-aligned 的可视化证据，而不是只看汇总表。
- 新增：`tools/analysis/build_raw_gate_error_visualization_v1.py`；输出 `docs/analysis/RAW_GATE_ERROR_VISUALIZATION_V1.md`、`results/analysis/raw_gate_error_visualization_v1/`、`figures/raw_gate_error_visualization_v1/`。
- 输入：StackOverflow/KIR=.50、seed `{13,42,87}` 的 RACAL Trainable/Frozen K=1、Fixed K=2 和 MOGB frozen-MiniLM fair component raw predictions；逐 seed 校验四方法 sample_id、gold OOS 标签和样本数一致，共读取 72,000 行。
- 结果：Trainable K=1 均值 OOS F1/F1-All/Known Recall/FA 为 `0.8671/0.8565/0.8392/0.1114`；Fixed K=2 为 `0.6765/0.7681/0.9362/0.4526`；MOGB fair component 为 `0.7319/0.4515/0.2834/0.0092`。K=1→K=2 平均恢复 `297.3` 个 Known、却新增误接收 `1,025.3` 个 OOS。
- 图表：分数分布、ROC/PR、指标柱图、OOS 错误转移、接受区域扩张 waterfall、intent 风险热图；曲线点已压缩为轻量摘要，未复制原始文本/embedding/逐样本 raw score 到 Git 跟踪结果树。
- 数据影响：无 canonical、registry、view、export、checkpoint 或历史 artifact 修改；只读取本地 raw predictions。外部官方 BERT MOGB 5-seed 摘要单独保留为非同合同兼容复现。
- 验证：脚本 `py_compile`、`ruff`、实际运行、CSV 解析和图像人工检查通过；待完成 research-state、development-log、data-tracking、registry audit 和 `git diff --check`。
- 风险：该证据仍是 StackOverflow/KIR=.50/3 seed 的 Gate-only 分析；不能宣称 SOTA、不能把 MOGB fair component 当作官方 MOGB 公平排名、不能替代 DCLOOS 等不同监督条件的端到端对比。
- 下一步：继续用现有 raw artifact 生成表示几何/近邻和错误案例图，保持训练矩阵冻结；任何新模型或全量扩展需单独登记并先通过重复实验 preflight。

## 2026-08-08 — 跨数据集/KIR 可视化证据链（analysis-only）

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`；未执行 git add/commit/push。
- 目标：把已有多数据集、多 KIR、五 seed 结果组织成“性能—工作点—机制”可视化，而不是继续增加未注册模型。
- 新增：`tools/analysis/build_visual_evidence_chain_v1.py`、`docs/analysis/VISUAL_EVIDENCE_CHAIN_V1.md`、`results/analysis/visual_evidence_chain_v1/`、`figures/visual_evidence_chain_v1/`。
- 输入：63 个同协议五 seed fair summary、90 个固定 K light-sweep 行、16 个 StackOverflow intent 诊断行；外部 baseline 只复制到隔离 CSV，不参与统一排名。
- 结果：Trainable K=1 在 9 个同协议 dataset×KIR 单元中 8 次 OOS F1 第一、1 次第二；跨 KIR OOS F1 为 CLINC150/Banking77/StackOverflow `89.48/81.29/86.48%`。StackOverflow intent 平均恢复 `30.0` 个 Known、却新增 `131.4` 个 OOS 误接收，ARI 平均 `0.926`。
- 图表：性能热图、Pareto、KIR 曲线、排名热图、相对差值热图、K sweep 覆盖—拒识折中、intent 风险散点，共 7 张；不输出原始文本、embedding、checkpoint 或逐样本预测。
- 数据影响：无训练、无 canonical/registry/view/export/历史 artifact 修改；所有输入 SHA256 记录在 `results/analysis/visual_evidence_chain_v1/MANIFEST.json`。
- 验证：脚本 py_compile、ruff、实际运行、图像人工检查通过；待完成研究状态、开发日志、数据跟踪和 git diff 最终检查。
- 风险：K sweep 仅为 3-seed light 机制消融，intent-level 行为是 test-sensitivity 描述，均不能用于正式选择 K；本阶段仍不是完整 Cascade 或 SOTA 证据。
- 下一步：在现有 raw predictions 基础上补 Frozen/Trainable/MOGB 的表示几何和 near-OOS 可视化；保持训练矩阵冻结，外部 baseline 继续按监督合同分层。

## 2026-08-08 — 表示几何与 Near-OOS 可视化证据（analysis-only）

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`；未执行 git add/commit/push。
- 目标：把“Trainable K=1 的表示收益”和“固定多中心的边界风险”连接到可视化证据，补充已有性能热图和 raw 错误转移。
- 新增：`tools/analysis/build_representation_geometry_visuals_v1.py`、`docs/analysis/REPRESENTATION_GEOMETRY_VISUALS_V1.md`、`results/analysis/representation_geometry_visuals_v1/`、`figures/representation_geometry_visuals_v1/`。
- 输入：已有 Frozen/CE/SupCon 几何与 K=1/K=2 汇总、Trainable/Frozen KIR=.50 score 诊断、StackOverflow Trainable-vs-MOGB raw transitions；没有训练、调参、重建中心或覆盖历史 artifact。
- 输出：27 行几何聚合、9 行 K=2 变化、6 行 score-gap/false-accept、8 行错误四象限和 4 张图；MANIFEST 记录四个输入 SHA256。
- 关键结论：CE/SupCon 改善 K=1 几何，但 StackOverflow K=2 near-OOS 下降 `10.7pp/32.3pp`；Trainable 的 score gap 更大且 false acceptance 更低；MOGB fair 更保守，Trainable 保留更多 Known 覆盖。
- 数据影响：无 canonical、registry、view、export、checkpoint 或原始文本修改；测试标签仅用于事后可视化，不用于选择。
- 验证：脚本实际运行成功，py_compile、ruff 和图像人工检查通过；后续补充 research-state、development-log、data-tracking、registry audit 和 git diff 最终检查。
- 风险：仍是 Gate/组件级机制证据，不是完整 Cascade 或官方 MOGB/DCLOOS 公平 SOTA 比较；near-OOS 继续按 exploratory 合同解释。
- 下一步：整理同协议性能、逐样本错误、表示几何和 near-OOS 的统一 evidence bundle；不新增 K 或训练矩阵。

## 2026-08-08 — 当前实验与可视化证据总览 V1（analysis-only）

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`；未执行 git add/commit/push。
- 目标：把性能热图、StackOverflow 逐样本错误、表示几何、near-OOS 和外部基线监督合同集中到一个中文阅读入口，避免跨协议数字被误读为统一 SOTA 排名。
- 新增：`docs/analysis/EXPERIMENT_VISUAL_EVIDENCE_BUNDLE_V1.md`；该文件只链接和汇总已有报告、CSV 与图表，没有新增训练或测试选择。
- 关键结论：Trainable K=1 是当前同协议自有 Gate 的最稳定工作点；StackOverflow K=2 的新增 OOS 误接收远大于恢复 Known；CE/SupCon 的几何改善不能保证 K=2 near-OOS 安全；MOGB fair 是更保守而非同工作点的比较。
- 数据影响：无；未修改 canonical、registry、view、export、checkpoint、embedding 或历史 artifact。
- 风险：总览仍不能替代同监督条件的完整外部 baseline 主表，也不能把 Gate-only 结果写成完整 Cascade 或 SOTA。
- 下一步：以该总览作为当前分析入口，只有在发现关键同协议字段缺失时才登记最小补充实验，不扩展旧 K/训练矩阵。

## 2026-08-08 — 外部基线合同与可比性可视化（analysis-only）

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`；未执行 git add/commit/push。
- 目标：把 ADB、DA-ADB、MOGB、BRAK、DCLOOS 与当前 Trainable/Frozen fair rows 的监督、表示、seed、数据和评价合同画清楚，防止不同合同数字被误读为统一排名。
- 新增：`tools/analysis/build_baseline_contract_visuals_v1.py`、`docs/analysis/BASELINE_CONTRACT_VISUALS_V1.md`、`results/analysis/baseline_contract_visuals_v1/`、`figures/baseline_contract_visuals_v1/`。
- 结果：StackOverflow/KIR=.50 的 ADB/DA-ADB 兼容单格 OOS F1 高于 Trainable，但它们是端到端 BERT 单格；同协议 Trainable/Frozen/MOGB 组件保留五 seed fair 标签。DCLOOS reduced-budget 因不同监督/数据合同不进入同一散点。
- 数据影响：无训练、无调参、无 canonical/registry/view/export/checkpoint/embedding 或历史 artifact 修改。
- 验证：脚本实际运行、py_compile、ruff、图像人工检查通过；待最终 research-state、development-log、data-tracking、registry audit 和 git diff 检查。
- 风险：该图证明的是“合同差异和当前可比字段”，不是 SOTA 结论；ADB/DA-ADB 仍需同协议多 seed 运行后才能做正式统计比较。
- 下一步：以合同矩阵识别缺失的同协议基线字段，保持其他训练矩阵冻结。

## 2026-08-08 — ADB/DA-ADB protocol_v2 数据适配与执行预检

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`；本轮未执行 `git add`、`git commit` 或 `git push`。
- 目标：在不混合历史兼容结果的前提下，将固定 `protocol_v2_textoir_v1` 的 StackOverflow/KIR=0.50 数据清单接入 TextOIR 外部基线 runner，并验证 ADB/DA-ADB 是否能开始同协议运行。
- 新增/修改：`tools/compat/textoir/run_external_textoir.py` 支持显式 `--data-root`、`--known-labels-file`；新增 `tools/compat/textoir/build_protocol_data_root.py`；新增 `docs/analysis/BASELINE_EXECUTION_STATUS_V1.md` 及其轻量状态产物。
- 数据影响：seed=42、87、100 的 train/dev/test 与 Known labels 已从 protocol exports 物化到隔离 artifact root，复制前后 SHA256 一致；未修改 canonical、registry、views、exports、checkpoint 或 embedding。
- 执行：ADB seed=42 首次到达数据加载后因 BERT 权重格式失败；转换到隔离 `pytorch_model.bin` 后，旧版 torch/Transformers/CUDA 环境探针超时；DA-ADB 因共享运行时阻塞未启动。
- 结果：同协议适配成功但没有可用 ADB/DA-ADB 指标；旧 seed=0 兼容单格保留为 `legacy_compatibility_only`，不得进入当前 protocol_v2 主排名。
- 记录：`results/analysis/baseline_execution_status_v1/attempt_status.csv`、`protocol_data_hashes.csv`、`MANIFEST.json`；台账新增 `baseline_execution_status_v1`，状态为 `blocked_preflight`。
- 验证：适配器与状态脚本已实际运行并完成 `py_compile`；最终 research-state、development-log、data-tracking、registry audit 和 `git diff --check` 仍需收口检查。
- 风险：旧版依赖的 CUDA/torch 探针不稳定；在独立且可验证 runtime 恢复前，不能把兼容单格数字解释为同协议 SOTA。
- 下一步：仅在修复/隔离旧 runtime 后先运行 StackOverflow seed=42 ADB 与 DA-ADB，并通过完整 artifact audit 后再扩展 seed=87、100；不扩展新的 K/表示训练矩阵。

## 2026-08-08 — 配对效应与 intent-level 异质性分析（analysis-only）

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：在不重复训练的前提下，用同一 `dataset × KIR × seed` 配对统计 Trainable K=1 与 Frozen/MOGB 组件的差异，并量化逐 intent 多中心潜在收益的异质性。
- 新增：`tools/analysis/build_paired_effect_intent_heterogeneity_v1.py`、`docs/analysis/PAIRED_EFFECT_INTENT_HETEROGENEITY_V1.md`、`results/analysis/paired_effect_intent_heterogeneity_v1/`、`figures/paired_effect_intent_heterogeneity_v1/`。
- 执行：读取 315 行 fair per-seed 输入，产生 270 个有效配对行、216 个 metric summary；读取 13,580 行 intent oracle 输入，生成 18 个 dataset×KIR×distance 汇总和 88 个 best-K 分布行。
- 统计：固定 bootstrap RNG 基准与 10,000 次重采样，输出均值、标准差、95% CI、median 和 win/tie/loss；没有使用这些 test-oracle 字段进行方法选择。
- 结果：Trainable K=1 相对 Frozen K=1 的 OOS F1 在三个数据集和三个 KIR 方向一致为正；Banking77 的 safe_gain_oracle 比例整体高于 StackOverflow，StackOverflow 在 KIR=.75 的 Euclidean safe-gain 比例进一步下降。
- 数据影响：没有修改 canonical、registry、views、exports、checkpoint、embedding 或历史 artifact；外部基线仍按合同隔离。
- 验证：脚本实际运行、输出图人工检查、后续需完成 py_compile/ruff、research-state、development-log、data-tracking、registry audit 和 `git diff --check`。
- 风险：intent oracle 结果仅是后验机制证据；它不能支持正式 adaptive-K 选择或 SOTA 声明。
- 下一步：继续补同协议外部基线运行证据；在 runtime 修复前不扩展新的训练矩阵。

## 2026-08-08 — 外部 baseline runtime 二次排查

- 目标：区分 ADB/DA-ADB 的算法问题、数据合同问题和 torch/CUDA 运行时问题。
- 检查：测试项目默认环境、`bo`、`implicit_intent`、`textoir-py39` 等多个解释器，以及 CPU-only 环境变量组合；只执行最小 torch import/CUDA probe，没有启动训练。
- 结果：多个解释器在 12–15 秒内超时，历史过程中出现原生 signal 6 和 D-state 进程；`nvidia-smi` 可见 GPU 但没有稳定计算进程。
- 结论：protocol_v2 数据适配仍然正确；当前同协议 ADB/DA-ADB 缺少可用 runtime，不应把该阻断解释为方法性能或数据集失败。
- 记录：补充至 `docs/analysis/BASELINE_EXECUTION_STATUS_V1.md`；未修改数据、checkpoint 或历史 artifact，未执行 git add/commit/push。
- 下一步：仅在独立 runtime 通过 torch import、BERT forward 和 CPU/GPU smoke 后重试 seed=42；不重复启动当前已知会进入 D-state 的环境。

## 跨数据集错误归因（analysis-only，2026-08-08）

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：将 StackOverflow 的逐样本错误归因推广到 CLINC150、Banking77 和三档 KIR，检查 Trainable K=1 的 OOS/coverage 优势是否稳定，并分离 MOGB 的低误接收与高 Known 误拒绝。
- 新增：`tools/analysis/build_cross_dataset_error_attribution_v1.py`、`docs/analysis/CROSS_DATASET_ERROR_ATTRIBUTION_V1.md`、`results/analysis/cross_dataset_error_attribution_v1/`、`figures/cross_dataset_error_attribution_v1/`。
- 执行：读取 315 个已有 fair prediction run，按 `dataset × KIR × seed × sample_id` 对齐 1,890,000 条记录；重新计算 OOS F1、Known Recall、false acceptance 和 false rejection，并检查源指标一致；无训练、无测试调参、无数据或历史 artifact 修改。
- 结果：Trainable K=1 相对 Frozen K=1、Frozen K=2 和 Random K=2 在全部 9 个 dataset×KIR 单元降低 false acceptance；StackOverflow 的差异随 KIR 增大最明显。MOGB 组件在多数单元降低 false acceptance，但以更高 false rejection 为代价，不能解释为同监督整体优胜。
- 风险：MOGB 行是 Frozen MiniLM 的公平组件证据，不是官方 BERT 论文复现；ADB、DA-ADB、DCLOOS 尚未进入同监督统一表，当前 runtime blocker 仍然存在。
- 验证：脚本已实际运行并人工检查四张图；本条记录之后执行 `py_compile`、`ruff`、research-state、development-log、data-tracking 和 `git diff --check`。
- 下一步：在 runtime 修复之前不新增外部基线训练；优先维护统一 evidence bundle，随后从可验证 seed=42 单格开始恢复 ADB/DA-ADB/DCLOOS 的同协议证据。

## 跨数据集逐意图风险可视化（analysis-only，2026-08-08）

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：把三数据集×三 KIR 的总体风险拆到 intent，识别 OOS 吸收器、Known 过拒绝和错误集中度，补充性能图背后的机制证据。
- 新增：`tools/analysis/build_cross_dataset_intent_risk_visuals_v1.py`、`docs/analysis/CROSS_DATASET_INTENT_RISK_VISUALS_V1.md`、`results/analysis/cross_dataset_intent_risk_visuals_v1/`、`figures/cross_dataset_intent_risk_visuals_v1/`。
- 执行：读取 4,641 行已审计 intent summary，输出三份 CSV、MANIFEST 和四张图；没有读取原始文本、没有重跑模型、没有调整阈值或选择中心数。
- 结果：StackOverflow KIR=.50 的 Frozen K=2 前五个 OOS 吸收 intent 贡献约 64.9% 的 false acceptance；MOGB-MiniLM 的 OOS 误接收更低但 Known false rejection 更高，Trainable K=1 处于更平衡的风险位置。
- 风险：所有结果是 test prediction 的后验机制归因，不得作为正式 intent-level K 选择器或 SOTA 证据；外部 ADB/DA-ADB/DCLOOS 仍未进入统一监督表。
- 验证：脚本已运行并人工检查四张图；本条记录后执行 `py_compile`、`ruff`、research-state、development-log、data-tracking 和 `git diff --check`。
- 下一步：继续维护可审计可视化证据链；待独立 runtime 可用后再恢复同协议外部 baseline。

## 同协议五 seed 统计稳定性（analysis-only，2026-08-08）

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：用同一 `dataset × KIR × seed` 配对统计验证 Trainable K=1 与 Frozen、多中心和 MOGB 组件的差异是否稳定，并生成可读的排名和置信区间图。
- 新增：`tools/analysis/build_statistical_stability_v1.py`、`docs/analysis/STATISTICAL_STABILITY_V1.md`、`results/analysis/statistical_stability_v1/`、`figures/statistical_stability_v1/`。
- 执行：读取 315 行已完成 fair per-seed 指标，生成 486 个配对 effect 行、seed rank 汇总和三张图；固定 bootstrap seed=20260725、10,000 次重采样；没有训练、调参或修改历史 artifact。
- 结果：Trainable K=1 相对 Frozen K=1、Frozen K=2、Random K=2 在 9/9 dataset×KIR 组合中五 seed 全部胜出；按 OOS F1 平均排名在 8/9 单元第一，Banking77/KIR=.25 由 MOGB partition + ours 略高。
- 风险：五 seed 统计只能证明当前 fair 矩阵的稳定性，不能替代 ADB/DA-ADB/DCLOOS 的统一监督实验，也不能写成 SOTA 结论。
- 验证：脚本实际运行并人工检查 forest、rank heatmap 和 KIR 曲线；本条记录后执行 `py_compile`、`ruff`、research-state、development-log、data-tracking 和 `git diff --check`。
- 下一步：继续保持旧实验不可覆盖；待独立 runtime 可用后恢复外部 baseline 的最小同协议单格。

## MOGB 与 Trainable MiniLM 同工作点可视化诊断（analysis-only，2026-08-08）

- 目标：在不重跑 E2/E3/R1、不改变任何正式阈值的前提下，把 Trainable K=1、Frozen 单/双中心、随机划分和 MOGB 组件放到相同 Known Recall 工作点，直接观察 OOS F1 与 false acceptance 的权衡。
- 新增：`tools/analysis/build_mogb_operating_point_visuals_v1.py`、`docs/analysis/MOGB_OPERATING_POINT_VISUALS_V1.md`、`results/analysis/mogb_operating_point_visuals_v1/`、`figures/mogb_operating_point_visuals_v1/`。
- 规模：读取 protocol_v2_textoir_v1 的 315 个已有 fair per-seed run，生成 1,260 个工作点行（4 个目标 Known Recall），没有训练、调参或覆盖历史 artifacts。
- 结果：Known Recall≈0.85 时 Trainable MiniLM K=1 的 OOS F1/FA 为 CLINC150 0.9164/0.0715、Banking77 0.8241/0.1964、StackOverflow 0.8705/0.1132；StackOverflow 固定 K=2 为 0.6692/0.4216，MOGB-MiniLM 为 0.6860/0.3913，ours partition + MOGB boundary 为 0.4169/0.6928。
- 解释：在同一 Known coverage 下，Trainable K=1 位于更安全的 coverage–open-space tradeoff；MOGB/固定多中心的主要风险仍是 OOS 误接受，不是单纯 Known Recall 过低。
- 边界：阈值来自 test-known 分数的事后对齐，仅是机制可视化，不能用于正式模型/参数选择；外部 ADB/DA-ADB/DCLOOS 仍因 runtime blocker 未进入此表。
- 验证：脚本实际运行、4 张图人工检查；后续需运行全套 state/log/data/registry/diff 检查。

## StackOverflow 逐样本错误归因（analysis-only，2026-08-08）

- 目标：用同一 `sample_id` 对齐 StackOverflow/KIR=.50/seed={13,42,87,100,123} 的 Trainable K=1、Frozen K=1/K=2、Random K=2 和 MOGB 组件，区分 OOS 误接收与 Known 误拒绝的来源。
- 新增：`tools/analysis/build_stackoverflow_error_attribution_v2.py`、`docs/analysis/STACKOVERFLOW_ERROR_ATTRIBUTION_V2.md`、`results/analysis/stackoverflow_error_attribution_v2/`、`figures/stackoverflow_error_attribution_v2/`。
- 规模：7 方法×5 seed×6,000 条预测，共 210,000 条逐样本对齐记录；检查 sample_id 顺序、gold_intent、Known/OOS=3,000/3,000，且按 score<=1 重算的 OOS F1/Known Recall 与源 metrics 一致。
- 结果：Trainable K=1 的 OOS F1/FA/FR 为 0.8767/0.0934/0.1611；Frozen K=2 为 0.6353/0.4717/0.1311；MOGB-MiniLM 为 0.7292/0.0079/0.7291。固定 K=2 的主要误接收集中于 cocoa、sharepoint、osx、spring、scala 五个 intent。
- 解释：固定多中心的主要损失是 OOS acceptance-region 过覆盖；MOGB-MiniLM 的主要损失是过度保守、Known 大量被拒；Trainable K=1 的优势是同时降低 OOS 误接收并保持较高 Known 覆盖。
- 边界：这是对冻结预测的机制归因，不使用 test 结果调参，不证明跨数据集 SOTA，也不替代统一 ADB/DA-ADB/DCLOOS 运行。
- 验证：脚本运行、4 张图人工检查、py_compile/ruff；后续继续执行 research-state、development-log、data-tracking、registry 与 diff 检查。

## Gate→Cascade 配对桥接分析 V2（analysis-only，2026-08-08）

- Base commit：`a5a96fed4a779afdfb1586dfbe91efeb9565d541`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：将已有 Trainable Gate-only 结果与三 seed Cascade 结果配对展示，同时在 Cascade 内部比较 Frozen、CE-Recon 和受控候选，避免把不同评价层误作端到端排名。
- 新增：`tools/analysis/build_gate_cascade_paired_bridge_v2.py`、`docs/analysis/GATE_CASCADE_PAIRED_BRIDGE_V2.md`、
  `results/analysis/gate_cascade_paired_bridge_v2/`、`figures/gate_cascade_paired_bridge_v2/`。
- 执行：读取 45 行源表，固定 bootstrap seed=20260725、10,000 次重采样；生成 48 行 Gate/Cascade bridge effect、72 行 Cascade pairwise effect 和三张图；没有训练、调参或修改历史 artifact。
- 结果：Trainable Gate 相对 Frozen K=1 Cascade 的 OOS F1 差值为 CLINC150 +2.41 pp、Banking77 −0.05 pp、StackOverflow +7.69 pp；false acceptance 分别降低约 3.44、10.20、12.38 pp。Cascade 内 CE-Recon selected-K 相对 Frozen K=1 的 OOS F1 差值分别为 +1.97、+4.66、+8.60 pp。
- 风险：Trainable 行是 Gate-only，Cascade 行包含 Router/Expert；bridge 不能作为统一端到端排名，3 seed 也不足以替代五 seed 外部基线。
- 验证：人工检查三张图和两个 CSV，确认必需指标无 NaN；随后执行 py_compile、ruff、research-state、development-log、data-tracking 与 `git diff --check`。
- 下一步：保持历史结果冻结；待独立 runtime 通过 torch import、BERT forward 和 smoke 后，才开始外部 baseline 同协议单格。

## 同一 Trainable 表示下的检测器机制对照（analysis-only，2026-08-08）

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：控制 Trainable MiniLM 表示不变，只比较 K=1 Gate 与 MSP、Energy、kNN、LOF，区分表示适配与检测器/校准机制的贡献。
- 新增：`tools/analysis/build_trainable_detector_mechanism_v1.py`、`docs/analysis/TRAINABLE_DETECTOR_MECHANISM_V1.md`、`results/analysis/trainable_detector_mechanism_v1/`、`figures/trainable_detector_mechanism_v1/`。
- 执行：读取 9 个 Gate 行和 36 个原生检测器行；固定 bootstrap seed=20260808、10,000 次重采样；无训练、无测试 OOS 选参、无历史 artifact 修改。
- 结果：Gate 相对 MSP/Energy/kNN/LOF 的 OOS F1 在 12/12 比较中三 seed 全部更高；但 Known Recall 下降约 11–21 pp，false acceptance 显著降低，表明收益是边界/校准工作点与表示共同作用。
- 风险：原生检测器使用 Known-only conformal alpha=.05，Gate 使用正式 Gate 规则，校准目标不同；结果只作机制诊断，不写成 SOTA。
- 验证：人工检查三张图；执行 py_compile、ruff、research-state、development-log、data-tracking 与 `git diff --check`。
- 下一步：继续补充同协议机制图；外部 baseline 等独立 runtime smoke 通过后再运行。

## 全指标 Fair Effect Landscape（analysis-only，2026-08-08）

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：将已有五 seed fair matrix 的 OOS F1、F1-All、F1-K、Known Recall、错误率、AUROC 和
  AUPR-OOS 放入统一的 Trainable-vs-comparison 效果景观，避免只挑一个指标解释结果。
- 新增：`tools/analysis/build_fair_effect_landscape_v1.py`、`docs/analysis/FAIR_EFFECT_LANDSCAPE_V1.md`、
  `results/analysis/fair_effect_landscape_v1/`、`figures/fair_effect_landscape_v1/`，并同步
  `docs/CURRENT_STATUS.md`、`docs/analysis/EXPERIMENT_VISUAL_EVIDENCE_BUNDLE_V1.md`、
  `docs/analysis/EXPERIMENT_COMPARISON_ZH.md`。
- 执行：读取 `statistical_stability_v1/paired_effects.csv` 的 486 行源表，按预注册范围筛选
  432 行（3 数据集 × 3 KIR × 6 比较 × 8 指标）；沿用固定 RNG 的 10,000 次 paired bootstrap，
  没有训练、调参、测试选择或历史 artifact 覆盖。
- 结果：Trainable K=1 相对 Frozen K=1/K=2/Random K=2 的 OOS F1 平均优势约 +7.1/+10.2/+7.4 pp，
  9/9 dataset×KIR 单元的 CI 支持正差异；相对三个 MOGB 组件平均约 +12.4/+8.0/+9.9 pp，
  8/9 单元稳定为正。多指标图显示 MOGB 组件更保守，Known false rejection 明显更高，
  Trainable 是当前 fair matrix 内较平衡的工作点。
- 风险：该分析只证明当前同监督 fair matrix 的机制与稳定性，不能替代同协议 ADB/DA-ADB/DCLOOS，
  不能称为官方 MOGB 复现，也不能声明 SOTA。
- 验证：待本条记录后执行 `py_compile`、`ruff`、research-state、development-log、data-tracking
  和 `git diff --check`；图像已人工检查，当前图中使用英文标签以避免字体方框问题，报告和状态文档为中文。
- 下一步：恢复独立外部 baseline runtime 后，先完成 StackOverflow/KIR=.50/seed=42 的可验证单格；
  不继续盲目扩大 K、损失项或 adaptive-K 矩阵。

## KIR 敏感性分解（analysis-only，2026-08-08）

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：量化 KIR=.25→.75 时 OOS F1、Known Recall 和 false acceptance 的变化，解释固定多中心、
  MOGB 组件与 Trainable K=1 的不同退化来源。
- 新增：`tools/analysis/build_kir_sensitivity_decomposition_v1.py`、
  `docs/analysis/KIR_SENSITIVITY_DECOMPOSITION_V1.md`、
  `results/analysis/kir_sensitivity_decomposition_v1/`、
  `figures/kir_sensitivity_decomposition_v1/`，并同步状态、实验索引和对比报告。
- 执行：读取已有 `cross_protocol_tradeoff_v1/summary_mean_std.csv`；计算 3 数据集 × 7 方法 ×
  6 指标的端点变化和线性斜率；无模型运行、无测试选参、无历史 artifact 修改。
- 结果：Trainable K=1 的 OOS F1 端点下降为 CLINC150/Banking77/StackOverflow 的
  −12.3/−23.0/−19.2 pp；Frozen K=2 为 −13.0/−26.5/−43.0 pp；StackOverflow Frozen K=2
  false acceptance 增量约 +42.2 pp。MOGB 组件误接收变化小但 Known Recall 明显下降，Trainable
  的 Known Recall 基本稳定。
- 风险：该阶段是已有结果的机制分解，不是新的统计独立重复，也不构成外部 baseline 或 SOTA 排名。
- 验证：图像人工检查；本条记录后执行 py_compile、ruff、research-state、development-log、
  data-tracking 和 `git diff --check`。
- 下一步：继续等待独立外部 runtime，完成 ADB/DA-ADB/DCLOOS 的可验证同协议单格；不新增重复 K 网格。
## OOS 误接收—误拒绝预算（analysis-only，2026-08-08）

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：把已有 fair matrix 的 OOS F1 拆成 OOS precision、OOS recall、OOS→Known 误接收和 Known→OOS 误拒绝，解释 Trainable、固定多中心和 MOGB 组件的工作点差异。
- 新增：`tools/analysis/build_oos_error_budget_v1.py`、`docs/analysis/OOS_ERROR_BUDGET_V1.md`、`results/analysis/oos_error_budget_v1/` 和 `figures/oos_error_budget_v1/`。
- 数据影响：读取已完成的 315 行 `method_metrics_per_seed.csv`，没有读取原始文本、没有重训、没有重拟合中心、没有调阈值，也没有覆盖 E2/E3/R1/MOGB 历史 artifact。
- 结果：重构 315 行明细、63 行五 seed 汇总和 324 行配对效应；重构 OOS F1 与源表最大绝对差小于 `1e-9`。KIR=.50 的 StackOverflow Frozen K=2 false acceptance 为 47.17%，MOGB MiniLM 组件 Known false rejection 为 72.91%，Trainable K=1 的 FA/FR 为 9.34%/16.11%。
- 执行命令：`python tools/analysis/build_oos_error_budget_v1.py`；统计使用固定 bootstrap seed `20260808` 和 10,000 次重采样。
- 测试：脚本 `py_compile`、`ruff`、输出无 NaN/重复键、manifest 行数校验和三张图人工检查；本条记录后继续运行 research-state、development-log、data-tracking 和 `git diff --check`。
- 状态：成功；风险是外部 ADB/DA-ADB/DCLOOS 仍未形成同监督、同 runtime 的正式主表，故本阶段不能支持 SOTA 排名。下一步仍是恢复可验证外部 baseline runtime，不重复已有 K 网格。
## 2026-08-09：统一实验分析主入口

- 目标：把当前同协议五 seed 结果整理为唯一、可视化、可追溯的分析入口，避免旧报告的评价口径和过时数字继续混用。
- 修改：新增 `tools/analysis/build_experiment_analysis_master_v1.py`、`docs/analysis/EXPERIMENT_ANALYSIS_MASTER_V1.md`。
- 数据影响：仅消费 `results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv`，不修改训练结果、checkpoint、split、registry 或阈值。
- 输出：`results/analysis/experiment_analysis_master_v1/` 与 `figures/experiment_analysis_master_v1/`，共 63 行审计汇总、正确方向的 cell ranking、全局汇总、Trainable 配对差值和 5 张图。
- 结论：Trainable K=1 在当前同监督矩阵中 OOS F1 排名第一 8/9、F1-All 排名第一 9/9；StackOverflow 固定 K=2 的主要风险是 FA 由 29.71% 升至 47.17%；MOGB MiniLM 组件以高 Known false rejection 换低 FA。
- 限制：该包不构成完整 MOGB、ADB、DA-ADB 或 DCLOOS 的公平 SOTA 比较；外部基线仍需独立 runtime 和监督合同审计。
- 测试：脚本运行成功；后续执行 py_compile、ruff、研究状态、数据跟踪、开发日志和 diff 检查。
- 下一步：先补齐同协议强基线可运行单格和多 seed，不扩展新的 K、损失项或自适应多中心矩阵。

## 2026-08-09：外部 baseline runtime 再次复核

- 目标：在启动 ADB/DA-ADB 前确认独立 PyTorch/CUDA runtime 可用。
- 执行：GPU 最小探针（RTX 5070）和 CPU-only `import torch`；没有启动训练。
- 结果：GPU 探针无稳定输出并超时；CPU-only import 触发 `free(): double free detected in tcache 2`，返回码 134。
- 数据影响：无；没有新增、删除或覆盖 baseline 结果。
- 状态：`runtime_blocked_no_metrics`。这不是 ADB/DA-ADB 算法失败，也不是数据错误。
- 回归：`compileall`、分析脚本和维护检查通过；`pytest tests/unit -q` 受同一 PyTorch/runtime D-state 阻断，未产生可解释的测试结论。
- 下一步：更换/修复隔离 runtime，先通过 torch import、BERT forward 和单批 smoke，再启动 StackOverflow/KIR=.50/seed=42 单格。

## 2026-08-09：可视化证据索引

- 新增 `docs/analysis/VISUAL_ANALYSIS_INDEX_V1.md`，将现有性能热力图、Pareto、KIR 曲线、表示几何、逐样本错误、intent 风险和外部合同图组织成四层证据链。
- 该索引只引用已完成结果，不新增训练、不重算阈值、不覆盖历史 artifact；当前五 seed 权威入口仍为 `EXPERIMENT_ANALYSIS_MASTER_V1.md`。

## 2026-08-09：方法对比地图与机制证据 V2

- 目标：解决“我的方法”同时指 Trainable K=1、固定多中心和自适应 pilot 所造成的比较混乱，并补充面向机制解释的可视化证据。
- 新增：`docs/analysis/METHOD_COMPARISON_MAP_V1.md`、`docs/analysis/MECHANISM_EVIDENCE_V2.md`、`tools/analysis/build_mechanism_evidence_v2.py`、`results/analysis/mechanism_evidence_v2/` 和 `figures/mechanism_evidence_v2/`。
- 数据影响：只读取已审计的五 seed fair matrix、Trainable K=2 control 和既有表示诊断；没有训练、没有重新选择阈值/K、没有修改历史 artifact。
- 结果：输出 63 行 fair matrix、54 行 acceptance-budget、9 行 Trainable K1/K2 控制、9 行表示风险诊断和 4 张机制图。图中明确区分 S2C-Trainable-K1、冻结/固定多中心、MOGB-Fair、MOGB 组件混合和外部 baseline 合同。
- 执行命令：`python tools/analysis/build_mechanism_evidence_v2.py`。
- 测试：脚本成功生成 `MANIFEST.json` 和四张图；隔离 Python 3.11 环境曾出现 I/O 阻塞，随后使用已验证项目解释器成功完成；不将阻塞误报为算法结果。
- 状态：成功；外部 ADB/DA-ADB/DCLOOS 仍未形成同协议正式主表，当前不能宣称 SOTA。下一步仍是先恢复可验证外部 baseline runtime，避免继续把不同合同的数字混排。

## 2026-08-09：历史 SOTA 与当前方法合同分层

- 目标：回答论文中“哪个方法超过 baselines”、当前 Gate-only 候选为何不能直接与历史 SOTA 混排，以及 MOGB 复现差距应如何定位。
- 新增：`tools/analysis/build_historical_sota_comparison_v1.py`、`docs/analysis/HISTORICAL_SOTA_AND_CURRENT_COMPARISON_V1.md`、`results/analysis/historical_sota_comparison_v1/`、`figures/historical_sota_comparison_v1/`。
- 数据影响：直接解析 `fulltex.tex` 的 `tab:main_results_all`（72 个历史单元），读取当前 63 行五 seed fair summary；没有训练、调参、覆盖历史 artifact 或把合同不同的结果合并排名。
- 结果：确认历史 `Ours` 是完整 Gate–Router–Expert Cascade；历史 OOS F1 在 9/9 dataset×KIR 单元高于表内七个基线，但 Known F1/Accuracy 并非全面第一。当前 `S2C-Trainable-K1` 仍是 Gate-only 当前协议候选，不能据此宣称超过历史 Cascade、完整 MOGB、ADB、DA-ADB 或 DCLOOS。
- MOGB 结论：报告将论文差距拆成数据合同、训练合同、旧代码现代兼容、指标合同、超参数/随机性五类；当前状态保持 `official_code_not_reproduced_under_available_materials`，不把低分解释为算法本身失败。
- 执行命令：`MPLBACKEND=Agg python tools/analysis/build_historical_sota_comparison_v1.py`。
- 测试：脚本 `py_compile` 通过，输出 72 行历史 CSV、契约分层 CSV、3 张图和 manifest；matplotlib 使用 Agg 后端，中文字体缺失仅产生渲染警告，不影响数值产物。
- 状态：成功；下一步是逐阶段 MOGB discrepancy audit 和稳定 runtime 后的同协议 ADB/DA-ADB/DCLOOS 单格，而不是继续把历史数字与当前 Gate-only 数字混成 SOTA 排名。

## 2026-08-09：MOGB 复现差距定量审计 V2

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；工作树已有未提交研究文件，未执行 `git add`、`git commit` 或 `git push`。
- 目标：解释两个官方 BERT 兼容单格为何显著低于 MOGB 论文参考，并把历史 Cascade、当前 Trainable Gate、MOGB-MiniLM 组件和官方 BERT 复现合同彻底分开。
- 新增：`tools/analysis/build_mogb_reproduction_gap_analysis_v2.py`、`docs/analysis/MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md`、`results/analysis/mogb_reproduction_gap_analysis_v2/`、`figures/mogb_reproduction_gap_analysis_v2/`。
- 数据影响：只读取冻结的 2 个 exact 单格、10 个五 epoch 非严格兼容运行、pinned MOGB 源码和已有四组合诊断；没有训练、重算预测、调阈值、改 checkpoint 或覆盖历史 artifact。
- 结果：CE 与 Known dev accuracy 已收敛，排除“只是没训练够”；公开 `myloss.py` 的 L1 距离归一化将子中心 softmax 压到接近均匀概率；28/99 个最终粒球证明动态划分实际运行；OOS Recall 约 98% 而 Known Recall 仅 51.53%/43.71%，定位到子中心目标和平均半径覆盖的联合作用。
- 输出规模：2 行 exact 汇总、92 行 epoch 轨迹、127 行粒球、8 行论文差值、10 行五 seed 短跑、5 条根因判断和 5 张图；manifest SHA256 为 `e9f14feacc61cf0eacf56da75ecf54c69d3f9ff25962986012f35bce1c44ebf5`。
- 执行命令：`python -m py_compile tools/analysis/build_mogb_reproduction_gap_analysis_v2.py`；`ruff check tools/analysis/build_mogb_reproduction_gap_analysis_v2.py`；`MPLBACKEND=Agg python tools/analysis/build_mogb_reproduction_gap_analysis_v2.py`。
- 风险：作者原始数据、完整旧依赖和逐样本合同仍缺失；现代兼容层修复了 stale-graph，因此本阶段只能说明“现有材料下未复现”，不能说明论文或算法无效。
- 下一步：先把去除 L1 距离归一化/加入温度和 Known-calibration 半径覆盖注册为 adapted ablation；不得伪装成 official reproduction，也不得与当前 S2C 行直接混排。

## 2026-08-09：MOGB Known-only 校准与子中心损失归因 V1

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；工作树已有用户和前序实验的未提交文件，本阶段未执行 `git add`、`git commit`、`git push` 或 destructive reset。
- 目标：实际验证本地 MOGB 论文差距是否主要来自默认 mean-radius 工作点，并在真实 MOGB-Fair 类别距离表上测量官方子中心损失的概率/梯度动态范围。
- 新增：`tools/analysis/run_mogb_known_calibration_attribution_v1.py`、`docs/analysis/MOGB_KNOWN_CALIBRATION_ATTRIBUTION_V1.md`、`results/analysis/mogb_known_calibration_attribution_v1/` 和 `figures/mogb_known_calibration_attribution_v1/`；同步实验/可视化索引、状态和 ledger。
- 数据影响：复用冻结 MiniLM cache 和 `protocol_v2_textoir_v1` 的 train/calibration/test views；没有重新编码、训练 encoder、修改 split、registry 或历史 MOGB run。半径倍率只由 `calibration_known` 的预注册 80%/85%/90%/95% 覆盖分位数确定。
- 执行：3 数据集×3 KIR×5 seed 共 45 个粒球重拟合；每单元 5 个工作点（225 单元）；每单元 6 个损失契约（270 单元）。命令：`python tools/analysis/run_mogb_known_calibration_attribution_v1.py`，收口复用命令：`python tools/analysis/run_mogb_known_calibration_attribution_v1.py --resume`。
- 等价性：45/45 与冻结 `mogb_minilm` 参考通过；指标最大绝对误差 0，score 最大绝对误差 `4.44e-16`，accepted-label mismatch 0。
- 结果：Calibration-95 相对默认半径恢复 Banking77/CLINC150/StackOverflow Known Recall `+61.16/+62.59/+68.40pp`，但 false acceptance 增加 `+76.64/+66.62/+78.90pp`，OOS F1 下降 `-37.02/-32.59/-37.78pp`。默认工作点过窄是真问题，但放大边界无法恢复论文级开放集权衡。
- 损失诊断：官方 L1-normalized 距离损失平均真类概率/梯度范数 `0.0601/0.0502`，raw-distance/tau=.10 为 `0.6389/3.8283`；粒球筛选还会在部分单元完全丢失 Known 类，StackOverflow 最坏排除 2,400 条训练行。
- 自有方法对比：S2C Trainable K=1 相对预注册 MOGB cal-80 在 OOS F1、F1-All 上均为 45/45 胜出；该结论仅适用于当前同 split 的 MOGB-Fair 组件合同，不能冒充完整官方 BERT MOGB 或 SOTA 宣称。
- Artifact：`../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_known_calibration_attribution_v1/`；轻量 manifest SHA256=`5e2d90bf4a793fcf19af240610f04245cab505ecdf63936853ca4fd20174656b`。
- 测试：脚本 `py_compile`、`ruff`、45 单元覆盖/等价门、CSV 有限值/唯一键、图像人工检查；随后运行 research-state、development-log、data-tracking 和 `git diff --check`。
- 风险与下一步：实际 raw-distance loss 只完成数值/梯度归因，尚未在稳定 PyTorch runtime 中重新训练 BERT；后续若 runtime 恢复，只允许注册一个 corrected-loss MOGB 单格，与原 exact 单格配对，不扩大官方复现矩阵。

## 2026-08-09：MOGB selected-ball 缺类安全回补归因 V1

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；`main` 比 `origin/main` 超前1个commit且工作树已有未提交研究文件；未执行 `git add`、`git commit`、`git push` 或 destructive reset。
- 目标：量化 MOGB-Fair 的 selected-ball 筛选遗漏 Known 类是否是当前 S2C–MOGB 差距的主要原因，并区分 Known 覆盖恢复与新增开放空间风险。
- 修改：新增 `tools/analysis/run_mogb_selected_class_rescue_v1.py`、`docs/analysis/MOGB_SELECTED_CLASS_RESCUE_V1.md`、`results/analysis/mogb_selected_class_rescue_v1/`、`figures/mogb_selected_class_rescue_v1/`；同步 `CURRENT_STATUS.md`、`EXPERIMENTS.md`、实验/可视化索引、ledger 和决策日志。
- 数据影响：只读取 `protocol_v2_textoir_v1` 冻结 MiniLM cache、train/calibration/test views 和上一阶段等价参考；不读取 `textoir/data`，不训练 encoder，不修改 canonical/registry/view/export 或历史 run。回补球只用 `train_known`，cal-80 只用 `calibration_known`。
- 执行：45个粒球重拟合单元+180个评分单元；命令 `python -u tools/analysis/run_mogb_selected_class_rescue_v1.py`，收口复用 `--resume`。45/45覆盖完成、0失败、16受影响、29零变化控制；manifest SHA256=`0968ada93bcbdb04b19bc5b733c4451267feba357cdaf1820f1369152717a05e`。
- 结果：默认阈值下StackOverflow受影响单元恢复Known Recall `+8.40pp`，同时 false acceptance `+6.23pp`、AUROC `-13.87pp`；cal-80 下 OOS F1 `-15.66pp`、false acceptance `+18.61pp`。当前 S2C Trainable K1 对回补版 MOGB cal-80 的 OOS F1/F1-All 仍为45/45胜出。
- 测试：单格烟测、旧MOGB默认指标逐项等价、零缺类零变化、180行唯一键/有限值检查、`py_compile`、Ruff和四张图人工检查完成；随后运行 research-state、development-log、data-tracking、compileall和 `git diff --check`。
- 风险与下一步：该回补是组件归因而不是官方MOGB修复；剩余差距仍指向子中心训练信号、粒球几何和平均半径联合失配。PyTorch runtime恢复前不启动corrected-loss BERT训练，可继续做无需训练的同协议样本/边界归因。

## 2026-08-09：MOGB 逐粒球开放空间风险归因 V1

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；工作树已有未提交研究文件，未执行 `git add`、`git commit`、`git push` 或 destructive reset。
- 目标：在共享完全相同 MOGB 自适应粒球分区的前提下，定位 MOGB-Fair 与 S2C 边界混合版的 OOS 误接收和 Known 误拒绝究竟由哪些粒球承担，并解释与 Trainable K1 的剩余差距。
- 修改：新增 `tools/analysis/build_mogb_ball_risk_attribution_v1.py`、`tests/unit/test_mogb_ball_risk_attribution_v1.py`、中文报告 `docs/analysis/MOGB_BALL_RISK_ATTRIBUTION_V1.md`、轻量结果目录 `results/analysis/mogb_ball_risk_attribution_v1/` 和 5 张机制图；同步状态、实验索引、可视化索引、ledger 与决策日志。
- 数据影响：只读取冻结 `mogb_baseline_v1` 的 `balls.jsonl`、`predictions.tsv`、`metrics.json` 和已完成 Trainable 汇总；没有训练、重算 embedding、修改 split/registry、调阈值或覆盖任何历史 artifact。test 标签只用于事后错误归因。
- 执行：`MPLBACKEND=Agg python tools/analysis/build_mogb_ball_risk_attribution_v1.py`；3 数据集×3 KIR×5 seed×2 方法共 90/90 单元、8,930 条 selected-ball 记录、0失败。
- 完整性：两种方法共享粒球的 ID、标签、训练支持、深度和纯度逐单元一致；逐样本重算的 OOS assignment、false acceptance、Known false rejection 与源粒球计数完全一致；manifest SHA256=`aca7b154c60aedbe964c4582c9bfa0d5acd02ab0cd6640ae24a59ff64e047e28`。
- 结果：每单元最高风险10%粒球承担约89%--98%的 OOS误接收；tiny ball 占36%--51%但不是主要错误载体，Q3/Q4大粒球承担更多错误。MOGB-Fair 在三个数据集的平均 Known Recall 仅约27%--34%；共享分区换用S2C边界可恢复到约50%--54%，但false acceptance增加且仍低于Trainable K1。
- 测试：3个新增单元测试通过，`py_compile`、Ruff、图像人工检查和阶段内唯一键/计数等价性检查通过；全局维护检查在本条之后执行并记录结果。
- 收口检查：90方法单元/45配对单元/8,930粒球行唯一性、核心有限值和源计数等价性通过；`compileall`、`git diff --check`、research-state、development-log、data-tracking、registry check-only 均通过。`export_public_results.py --verify` 未通过，唯一原因是全局 `results/` 存在白名单外研究目录（130条已登记记录、`results contains files outside the public whitelist`）；未为本阶段扩大全局白名单，避免意外公开无关结果。
- 风险与下一步：该阶段解释的是Frozen MiniLM下的MOGB-Fair组件，不是官方BERT MOGB。不得用test OOS粒球风险删球或选参；稳定PyTorch runtime恢复后，下一项训练证据仍应是预注册的corrected-loss官方BERT单格，而不是扩大粒球筛选网格。

## 2026-08-09：S2C 与 MOGB-Fair 开放意图五状态转移 V1

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；工作树已有未提交研究文件，未执行git add/commit/push。
- 目标：补足既有逐样本分析只看Known/OOS二分类的缺口，直接解释S2C与MOGB-Fair的F1-All差距来自Known正确、Known错类、Known拒绝、OOS正确拒绝还是OOS误接收。
- 修改：新增`tools/analysis/build_trainable_mogb_open_intent_transitions_v1.py`、3项单测、中文报告、6个轻量CSV、4张图和独立CLOSEOUT；同步状态、实验、可视化、ledger和决策日志。
- 数据影响：只读取`cross_protocol_tradeoff_v1`登记的Trainable K1与MOGB-Fair冻结预测；不训练、不调阈值、不改split/registry、不覆盖历史artifact。逐样本ID和原始文本不进入Git轻量结果。
- 执行：`MPLBACKEND=Agg python -u tools/analysis/build_trainable_mogb_open_intent_transitions_v1.py`；45/45配对单元、443,400条预测、0失败。Banking77/CLINC150/StackOverflow每方法单元分别为3,080/5,700/6,000条。
- 结果：S2C相对MOGB-Fair平均F1-All增量为`+32.48/+35.14/+43.67pp`；平均净Known正确增量`+734.87/+958.73/+1734.93`，平均净OOS正确拒绝增量`-192.20/-96.13/-169.80`。优势主要是恢复MOGB边界外Known，而不是更保守拒绝OOS。
- 完整性：逐配对sample_id/gold_intent/Known-OOS一致；F1-All重放最大绝对误差0；意图图只展示至少3/5 seeds进入Known集合的意图。script SHA256=`be6adb5e0e76fc4ffa66e563d5b5f10338bc15ab81491eefc4dac5b55a43375c`；manifest SHA256=`ed7cef1d3a208e25d5dd4201de2670059e254fe0ed1a9087823d02f7b71189f7`。
- 测试：3项新增单测、py_compile、Ruff和4张图人工检查通过；全局维护检查在本条之后运行。
- 收口：45配对单元、225个dataset×KIR×五状态矩阵格、90个指标重放和11个manifest输出哈希全部通过；6项相关单测、Ruff、compileall、research-state、development-log、data-tracking、registry check-only、`git diff --check`通过。公开结果verify仍报告全局`results contains files outside the public whitelist`（130条已登记记录），未扩大白名单。
- 风险与下一步：MOGB-Fair不是官方BERT MOGB；结论不能扩展为无条件SOTA。下一训练实验仍需等待稳定PyTorch runtime，优先执行一个corrected-loss BERT MOGB配对单格。

## 2026-08-09：MOGB corrected-loss BERT 单格与 S2C/基线统一入口

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；分支 `main` 比远端超前1个
  commit，工作树已有用户和前序研究文件；未执行 `git add`、`git commit` 或 `git push`。
- 目标：验证 MOGB 公开代码中 L1-normalized 最近子中心损失是否是本地官方逻辑单格显著低于论文
  公开参考的主要原因，并把 fulltex 历史 Ours、当前 S2C Gate 和 MOGB 合同整理成一个中文入口。
- 修改：新增 `src/protocol_v2/experiments/mogb_loss_contract.py`、单元测试、
  `configs/baselines/mogb_corrected_subcentroid_loss_v1.yaml`；在现有 exact runner 中通过实例注入替换损失，
  不修改 pinned `third_party/mogb_official`；新增统一分析脚本、中文报告、4张图和机器可读结果。
- 数据影响：只读取固定 StackOverflow 20,000条快照，KIR=.50/seed=0；没有修改 canonical、registry、
  E2/E3/R1 或原 MOGB artifact。温度1.0在测试前固定；checkpoint 只由 Known dev accuracy 选择。
- Artifact：`../artifacts/s2c/external/mogb_corrected_subcentroid_loss_v1/`；provenance SHA256
  `93166e6dd8e89f72a0482946334b3274fced18bf25891ae6e4b535e952a2a903`；checkpoint SHA256
  `127bd347fe9912f81b771630ae8f5d70bd91348567fb8bfd3b2c418da68dfc22`。
- 执行：dry-run 1/1、正式训练 1/1；61 epoch，best epoch=51，Known dev accuracy=92.10%，
  33个最终 selected balls；数值兼容 loss/gradient 最大差均为0。
- 结果：相对原本地官方逻辑单格，Accuracy `+2.08pp`、F1-All `+4.29pp`、F1-U `+0.99pp`、
  F1-K `+4.62pp`、Known Recall `+7.67pp`；OOS Recall `-3.50pp`。修正后 F1-All=72.64，
  仍比论文公开参考87.49低14.85pp，故只修损失不能完成复现。
- 输出：`docs/analysis/S2C_BASELINE_MOGB_COMPARISON_OVERVIEW_V1.md`、
  `results/analysis/s2c_baseline_mogb_overview_v1/`、`figures/s2c_baseline_mogb_overview_v1/`。
- 测试：新增8项损失契约/runner注入单测通过，目标文件 Ruff/compileall 通过；四张图已人工检查。全局维护检查在
  本条后运行并记录；不把 corrected-loss diagnostic 写成 MOGB-official 或 S2C 新方法。
- 风险与下一步：作者原始样本ID/Known列表和旧运行时仍缺失；剩余差距首先检查 corrected checkpoint
  上的 Known-only 半径覆盖，不扩 seed/KIR/数据集，不启动新的 adaptive-K 或 Pipeline。

## 2026-08-09：S2C 与 MOGB-Fair 机制对比仪表盘 V1

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；分支 `main`，工作树已有前序研究文件，
  未执行 `git add`、`git commit` 或 `git push`。
- 目标：用相同 dataset/KIR/seed 的逐单元配对回答当前 S2C-Trainable-K1 为什么优于
  MOGB-MiniLM-Fair，并将历史 Cascade、当前 Gate、MOGB 公平组件和论文 BERT MOGB 四种合同隔离。
- 修改：新增 `tools/analysis/build_s2c_vs_mogb_mechanism_dashboard_v1.py`、三项单元测试、中文报告、
  四个轻量 CSV/JSON 结果和六张机制图；同步状态、实验总账、决策日志、实验索引和可视化索引。
- 数据影响：只读取冻结的 `minilm_trainable_5seed_fair_v1` 与 `mogb_fair_per_seed.csv`；没有训练、
  重编码、调整阈值、修改 canonical/registry/split 或覆盖任何历史 artifact。
- 覆盖：45 个 dataset×KIR×seed 配对单元、198 行配对效应，0 失败；bootstrap RNG=20260725，
  10,000 次重采样。manifest SHA256=`1f3c73102a094a0b087fdbe95d02e68d06e0275c75bb77b4dc3a313d8f2dbf4e`。
- 结果：S2C 对 MOGB-Fair 的 OOS F1 44胜1负、F1-All 45胜0负；平均 Known Recall `+49.01pp`、
  OOS Precision `+20.58pp`、OOS Recall `-7.83pp`。共享粒球换 S2C 边界只恢复部分差距。
- 测试：三项新增单元测试、Ruff、目标文件 compileall、六张图人工检查、research-state、
  development-log、data-tracking、registry check-only 和 `git diff --check` 均通过。
- 风险与下一步：该阶段不是新训练或完整 MOGB 排名。外部 baseline 仍需同合同运行；不得据此宣称当前
  S2C 已超过论文 BERT MOGB 或成为无条件 SOTA。

## 2026-08-09：S2C 与 MOGB-Fair 工作点和排序能力归因 V1

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；分支 `main`，工作树已有前序研究文件；
  未执行 `git add`、`git commit` 或 `git push`。
- 目标：回答当前 S2C 的优势是分数排序更好，还是只因 MOGB mean-radius 默认阈值过窄，并在相同 Known
  coverage 与各自事后最优阈值下验证差距是否保留。
- 修改：新增 `tools/analysis/build_s2c_mogb_operating_curve_attribution_v1.py`、四项单元测试、中文
  报告、五个轻量CSV、CLOSEOUT、manifest和六张机制图；同步状态、实验总账、决策日志、实验/可视化索引。
- 数据影响：只读取 45 个 Trainable-K1 与 MOGB-Fair 已冻结逐样本预测；没有训练、重编码、修改 split、
  registry、checkpoint、正式阈值或历史artifact。test标签只用于明确标记的post-hoc曲线和oracle诊断。
- 覆盖：90方法单元、72配对效应、540同覆盖工作点、9,090条稠密阈值曲线，0失败；默认阈值90/90重放一致。
- 结果：AUROC 45/45胜出、平均 `+7.93pp`；AUPR `+12.00pp`；默认 OOS F1 `+12.36pp`；
  事后最优阈值下仍 `+6.43pp`。80% Known coverage下 OOS F1为 `86.05% vs 74.94%`。
- Artifact：`results/analysis/s2c_mogb_operating_curve_attribution_v1/`；manifest SHA256=
  `500bf78e3c2347d8d96a6df1f2804b72a851e112acdc9ed4171b971f60f19d58`。
- 测试：四项新增单元测试、Ruff、compileall、六张图人工检查、research-state、development-log、
  data-tracking、registry check-only 和 `git diff --check` 均通过。
- 风险与下一步：oracle/matched-coverage不得进入正式选参；该阶段解释MOGB-Fair组件，不是完整BERT MOGB。

## 2026-08-09：S2C 与 MOGB-Fair 逐意图结构--收益桥接 V1

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；工作树已有前序研究文件，未执行
  `git add`、`git commit` 或 `git push`。
- 目标：判断当前 S2C--MOGB-Fair 差距是否只由少数缺球/高碎片 intent 主导，以及 MOGB 每 intent
  selected-ball 数量能否稳定预测 Known 覆盖恢复或新增 OOS 代价。
- 修改：新增 `tools/analysis/build_s2c_mogb_intent_structure_bridge_v1.py`、五项单测、中文报告、七个
  轻量 CSV/JSON、CLOSEOUT、manifest 和六张图；同步当前状态、实验总账、决策日志和统一中文入口。
- 数据影响：只读取已冻结的逐 intent 错误归因和 MOGB selected-ball 统计；不训练、重编码、修改 split、
  registry、粒球、阈值或历史 artifact。test 标签只用于明确标记的事后机制关联。
- 覆盖：1,850 条 intent×seed 行、45 个单元、663 个 intent 汇总和 90 个结构相关结果，0失败。
- 结果：九个 dataset×KIR 组的正向 Known 恢复 intent 比例均为100%，平均逐 intent Known 拒绝率降低
  `46.70pp`；selected-ball 缺类占 `1.41%`；ball-count/recovery rho 为 `-0.382--+0.445`，
  没有统一方向。恢复 Gini 为 `0.10--0.21`，不是少数 intent 主导。
- Artifact：`results/analysis/s2c_mogb_intent_structure_bridge_v1/`；manifest SHA256=
  `2b139c5b23fea877e5c01cf1ba73ed8e65b6c9f3088c72fdc77d968b45ccc0f9`。
- 测试：新增单测、Ruff、compileall、六张图人工检查已通过；全局 research-state、development-log、
  data-tracking、registry check-only 与 `git diff --check` 在本条后执行并记录。
- 风险与下一步：该阶段只解释 MOGB-Fair 组件，不是完整 BERT MOGB；不得用 test-defined intent 风险
  删球或选择 K。外部同协议 baseline 仍未完成。

## 2026-08-09：corrected-loss BERT-MOGB Known-only 半径覆盖归因 V1

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；工作树已有前序研究文件，未执行
  `git add`、`git commit` 或 `git push`。
- 目标：固定 corrected-loss BERT checkpoint，检验本地MOGB与论文差距中有多少来自官方mean-radius
  工作点过窄；半径倍率仅用Known dev覆盖率确定，test OOS只评价。
- 修改：新增`tools/analysis/run_mogb_corrected_radius_coverage_v1.py`、3项单测、中文报告、4个轻量CSV、
  replay audit、manifest和2张中文图；同步CURRENT_STATUS、EXPERIMENTS、ledger、决策日志和分析索引。
- 数据与artifact影响：只读固定StackOverflow/KIR=.50/seed0数据快照和checkpoint
  `127bd347...`；不重训、不改canonical/registry/split、旧MOGB结果或第三方源码。
- 执行与覆盖：1个fixed checkpoint、33个确定性重建球、5个工作点，0运行失败。源artifact未保存最终
  粒球RNG/中心，故`strict_selected_ball_replay=false`；默认指标最大重放差1.87pp。
- 结果：default/cal-80/cal-95的F1-U为79.76/73.45/35.68，Known Recall为58.63/77.00/87.17；
  cal-95使OOS误接受从197增至2311。mean radius过窄真实存在，但放大半径不能恢复论文权衡。
- 测试：`pytest tests/unit/test_mogb_corrected_radius_coverage_v1.py -q`为3 passed；目标文件Ruff、
  py_compile通过；两张图完成中文字体与视觉检查；全局维护检查在本条后执行。
- 产物：`docs/analysis/MOGB_CORRECTED_RADIUS_COVERAGE_V1.md`、
  `results/analysis/mogb_corrected_radius_coverage_v1/`、`figures/mogb_corrected_radius_coverage_v1/`；
  manifest SHA256=`1e5ee8361243d052e53a4db34efe121c1adf0ebdfd81f571e1546cafcaf9b927`。
- 风险与下一步：当前是fixed-checkpoint deterministic reconstruction，不是原球结构严格重放。下一步
  只收口作者数据/Known列表/旧运行时/最终球状态；不再扫描半径、seed或扩大MOGB矩阵。

## 2026-08-09：外部基线 runtime 补充探针

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：在不启动训练的前提下检查其余本地 Python 环境是否存在可运行的 PyTorch/BERT runtime。
- 执行：对 `minimind`、`dreamer`、`dreamer_tf` 做受限 `torch import`/张量 smoke；结果分别为超时、未安装 PyTorch、未安装 PyTorch。
- 数据与 artifact 影响：无；没有读取或修改实验数据、registry、checkpoint 或结果文件。
- 结果：ADB/DA-ADB 仍为 `runtime_blocked_no_metrics`；GPU 可见不等于 Python/PyTorch forward 可用。
- 测试：状态文档、开发日志、数据跟踪和研究状态检查随后执行；没有产生新的性能指标。
- 风险与下一步：不得继续重复当前本地解释器探针；需要一个新的、先通过 torch import、BERT forward 和 CPU/GPU smoke 的隔离环境后，才允许启动 StackOverflow/KIR=.50/seed=42 单格。

## 2026-08-09：MOGB 官方代码合同逐行审计

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；上游 pinned commit：`5b689e2a03de0d86ec41212825e5db8d7f0e5c02`。
- 目标：把 MOGB 已观测的 loss、边界、随机性和运行时问题落实到可复核的源码行，避免把兼容运行低分直接解释为算法失败。
- 修改：新增 `docs/analysis/MOGB_CODE_CONTRACT_LINE_AUDIT_V1.md` 和
  `results/analysis/mogb_code_contract_line_audit_v1/evidence.csv`；未修改 third-party 源文件。
- 结果：确认 L1-normalized subcentroid loss、硬编码 `cuda:0`、随机递归球拆分、selected-ball 过滤、mean-radius
  和 nearest-ball acceptance 六类合同行为，与此前 loss/覆盖/replay 归因一致。
- 数据与 artifact 影响：无训练、无数据变更、无测试选参、无历史 artifact 覆盖。
- 测试：随后运行 research-state、development-log、data-tracking 和 `git diff --check`。
- 风险与下一步：该审计加强 MOGB discrepancy evidence，但不改变 `official_code_not_reproduced_under_available_materials`；
  下一步仍需可用旧 runtime 和作者数据合同，才可重新运行严格单格。

## 2026-08-09：统一对比实验图谱与可视化证据链 V1

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：把当前已有的大量对比结果组织为“方法合同—性能—错误预算—组件归因”图谱，避免历史 Cascade、当前 Gate 和外部兼容单格被混排。
- 修改：新增 `tools/analysis/build_comparison_atlas_v1.py`、中文报告、10 个轻量 CSV/JSON、7 张图和 manifest；同步当前状态与实验索引。
- 数据影响：只读取 `cross_protocol_tradeoff_v1`、历史 `fulltex` 解析、baseline contract matrix 和 MOGB exact summary；未重训、未调参、未修改 canonical/registry/split/checkpoint 或历史 artifact。
- 覆盖：当前 fair 汇总 63 行、per-seed 315 行、历史 72 行、外部合同参考 10 行、MOGB exact 本地参考 2 行；0 失败。
- 结果：Trainable K=1 相对 Frozen K=1 的提升、固定 K=2 的 false-accept 风险、MOGB 低误接受/高 Known 拒绝工作点均生成了可视化；直接排名仅限 current protocol_v2 fair Gate。
- Artifact：`docs/analysis/COMPARISON_ATLAS_V1.md`、`results/analysis/comparison_atlas_v1/`、`figures/comparison_atlas_v1/`；manifest 记录输入层级和不可混排行为。
- 测试：图谱脚本执行成功且无字体警告；`check_research_state.py`、`check_development_log.py`、`check_data_tracking.py` 和 `git diff --check` 随后执行。
- 风险与下一步：图谱仍不能替代同一监督合同下的 ADB/DA-ADB/DCLOOS 多 seed；下一步优先解决外部 baseline runtime，再把有效单格接入相同图谱。

## active experiment dashboard 数据合同修复（2026-08-09）

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；本次仅修改分析脚本和生成的轻量报告/图表，未执行 `git add`、`git commit` 或 `git push`。
- 目标：修复活动 dashboard 将旧 3-seed RACAL 数字误显示为当前正式 Trainable 结果、并把 Frozen K=2 误标为 Trainable K=2 的问题。
- 修改：`tools/analysis/build_active_experiment_dashboard_v1.py` 现在优先读取 `results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv` 的 5-seed fair rows；旧 RACAL 文件仅保留为诊断来源。当前主图明确区分 Frozen K=1、Trainable K=1、Frozen K=2。
- 结果：StackOverflow/KIR=.50 当前主图使用 Frozen K=1 `76.55%`、Trainable K=1 `87.67%`、Frozen K=2 `63.53%`；旧 `86.71%` 不再进入 current fair dashboard。重新生成 14 张图、156 行 overview 和新的 `DASHBOARD_MANIFEST.json`。
- 数据影响：无训练、无 canonical/registry/split/checkpoint 修改；历史 artifact 保留不覆盖。
- 测试：dashboard 脚本执行成功；14 张 PNG 尺寸检查通过；`compileall`、`check_research_state.py`、`check_development_log.py`、`check_data_tracking.py`、`git diff --check` 均通过。
- 风险与下一步：外部 ADB/DA-ADB/DCLOOS 仍不是同协议正式主排名；后续若扩展活动 dashboard，必须继续以 5-seed fair matrix 为 current source，并显式标记 legacy/compatibility rows。

## 2026-08-09：ADB 同协议数据根与运行前置复核

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：在不污染历史结果的前提下，验证 StackOverflow/KIR=.50/seed=42 的 protocol_v2 数据能否进入 ADB 外部 runner。
- 修改/产物：使用现有 `build_protocol_data_root.py` 物化独立 artifact 数据根；生成 ADB dry-run manifest；未修改 canonical、registry、split 或历史 run。
- 数据影响：train/dev/test/known_labels 复制前后 SHA256 一致；外部 runner 未读取 `textoir/data`。
- 执行命令：`build_protocol_data_root.py`；`run_external_textoir.py --method ADB --dataset stackoverflow --known-cls-ratio 0.50 --seed 42 --dry-run`；随后一次非 dry-run 前置探针。
- 测试/结果：dry-run 成功；实际运行未进入训练，默认解释器缺少 `easydict`，带该依赖的 `textoir-py39` 在 CUDA 探针超时；成功=0，失败/阻断=1，跳过=DA-ADB（共享 runtime blocker）。
- Artifact：`../artifacts/s2c/external/adb_protocol_v2_probe_data_v1/`、`../artifacts/s2c/external/adb_protocol_v2_probe/`。
- 风险与下一步：没有同协议 ADB/DA-ADB 指标，不能加入主排名；需要新隔离 runtime 先通过 torch import、BERT forward 和 CPU/GPU smoke，再重试单格。

## 2026-08-09：外部基线有效性审计与同协议单格比较

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；本次未执行 `git add`、`git commit` 或 `git push`。
- 目标：把已获得的 ADB 兼容运行与 protocol_v2 fair Gate 分开记录，并以逐样本预测审计 DA-ADB，禁止只信任旧 runner 的汇总 CSV。
- 修改：新增 `tools/analysis/build_external_single_cell_comparison_v1.py`；新增 `results/analysis/comparison_atlas_v1/stackoverflow_kir50_external_and_fair_cells.csv`、`stackoverflow_kir50_external_summary.csv`、无效运行表、manifest、中文报告和单格对比图；更新 `docs/analysis/BASELINE_EXECUTION_STATUS_V1.md`。
- 数据影响：只读取 protocol_v2 导出的 StackOverflow/KIR=.50 数据根、外部 runner 预测和既有 fair per-seed 结果；未修改 canonical、registry、split、checkpoint 或历史 MOGB artifact。
- 执行：`build_external_single_cell_comparison_v1.py`；ADB seed42/87 结果从 `y_true.npy/y_pred.npy` 重算；DA-ADB 原始及 clamp30 运行逐样本审计。
- 结果：ADB seed42/87/100 为有效同协议外部兼容单元，OOS F1=86.30%/86.99%/89.13%，F1-All=84.30%/85.32%/87.40%；DA-ADB 两次运行分别出现 NaN 或全类别预测，均标记 invalid，不进入排名。
- 测试：脚本执行成功；外部结果层显式记录 `layer`、`contract`、`valid_semantic_metrics` 和无效原因；后续继续运行维护检查。
- 风险与下一步：ADB 的 BERT/TextOIR 合同仍不同于 MiniLM fair Gate，不能直接宣称全面超越；三 seed 只构成 StackOverflow/KIR=.50 单元参考，DA-ADB 需另行修复数值稳定性才可获得有效结果。

## 2026-08-10：统一活动报告的外部基线状态

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 Git 提交、推送或清理。
- 目标：消除活动中文报告中“ADB 仍阻断/单元测试未通过”等过期表述，使实验状态与当前 artifact 证据一致。
- 修改：更新 `docs/CURRENT_STATUS.md`、`docs/CURRENT_METHOD_AND_EXPERIMENT_STATUS.md`、`docs/analysis/COMPARISON_ATLAS_V1.md`、`docs/analysis/EXPERIMENT_ANALYSIS_MASTER_V1.md` 和 `docs/analysis/EXPERIMENT_VISUAL_EVIDENCE_BUNDLE_V1.md`；保留旧 runtime blocker 条目作为历史审计。
- 结果：活动入口现在明确记录 ADB 三个有效同数据合同单元、DA-ADB 无效原因、DCLOOS 外部负样本阻断和 348 个 unit tests 通过；不改变任何训练结果。
- 验证：`pytest tests/unit -q` 348 passed；对比图谱和外部单格脚本重建成功；research-state、development-log、data-tracking、ruff、compileall 和 diff 检查通过。
- 风险与下一步：ADB 仍使用 BERT/TextOIR 外部合同，不能直接作为 MiniLM fair 主表；继续补 DA-ADB/DCLOOS 合同审计和错误预算可视化，不重复 E2/E3/K 扫描。

## 2026-08-10：当前方法—历史 Cascade—MOGB—外部基线对比总览

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：把“当前究竟比较哪个自有方法、历史 fulltex 哪个方法曾取得表内领先、MOGB 本地兼容结果为何不能冒充论文复现、ADB/DA-ADB/DCLOOS 当前证据边界”收束为一个中文入口，避免继续把 Gate-only、完整 Cascade、BERT 外部合同和 Known-only MiniLM 混成一张 SOTA 表。
- 新增：`docs/analysis/EXPERIMENT_COMPARISON_OVERVIEW_V2.md`、`tools/analysis/build_experiment_comparison_overview_v2.py`、`results/analysis/experiment_comparison_overview_v2/` 和 `figures/experiment_comparison_overview_v2/`；来源包含当前 fair 63 行、StackOverflow 外部单格、fulltex 历史 72 行和 MOGB 论文/本地差距表，全部保留来源哈希。
- 结果：当前自有候选明确为 S2C-Trainable-K1；它在当前 3 数据集×3 KIR×5 seed fair 矩阵的 OOS F1 9 格中 8 格第一、F1-All 9 格第一。StackOverflow/KIR=.50 为 OOS F1 `87.67±1.66%`、F1-All `86.55±1.30%`；ADB 同数据兼容参照为 `87.47±1.48%`/`85.67±1.58%`，但仍是 BERT/TextOIR 合同。历史 `fulltex.tex` 的 Ours 是完整 Gate–Router–Expert Cascade，旧合同下 9/9 格高于表内基线，不能替换成当前 Trainable-K1。
- DA-ADB：在原始和 clamp30 无效 artifact 基础上，隔离兼容 overlay 额外加入 reachability `exp` 截断和 `CosNorm_Classifier` 特征/权重范数 `clamp_min(1e-12)`；源 TEXTOIR 未修改，新增回归测试覆盖。由于本轮安全尝试没有生成语义有效 manifest，DA-ADB 继续标记 `invalid`，没有被提升为结果。
- 数据影响：无 canonical、registry、split、checkpoint 或历史 run 修改；只新增轻量分析产物和隔离兼容代码。
- 测试：对比总览脚本成功；四张 PNG 和五个 CSV/manifest 已生成；DA-ADB overlay 定向测试 `19 passed`，当前 unit 全集 `317 passed`（3 个旧依赖 deprecation warnings）；研究状态、数据跟踪、日志、compileall、ruff 和 diff 检查通过。
- 风险与下一步：当前仍不能宣称跨合同 SOTA；MOGB 官方论文工作点和 DCLOOS 监督条件尚未形成同协议正式排名。下一步优先维护合同分层的可视化/错误预算证据，或在独立、可验证 runtime 下恢复 DA-ADB/DCLOOS 单格，不重复 E2/E3/K 网格。

## 2026-08-10：DCLOOS 外部负样本来源收口到 reduced 兼容证据

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 Git 提交、推送或清理。
- 目标：纠正活动对比入口中“DCLOOS 外部负样本仍未找到”的过期状态，并把已恢复的 DCLOOS 结果与 Known-only fair 矩阵分层。
- 证据：作者 DCLOOS README 指向的 Drive `squad.tsv` 已有本地-only SHA256 `f6bf61866c86d3b11565826c3ca1faa00e31f196e0ad9bfd000ec45575fd426e`；官方默认预算单元 `dcloos_official_oos_kir75_seed888_v1` 超时，但 reduced 单元 `dcloos_official_oos_kir75_seed888_reduced_v2` 恢复出 OOS F1 `87.0527`、F1-All `90.2629`、Known Recall `92.1429`、Accuracy `88.6842`。
- 修改：更新 `docs/analysis/EXPERIMENT_COMPARISON_OVERVIEW_V2.md` 和 `docs/EXPERIMENT_LEDGER.csv`，明确 DCLOOS 来源已闭合、默认预算仍 timeout、reduced 结果使用 pseudo-OOS/外部 OOS 且不进入公平主排名。
- 数据影响：未复制或提交原始外部语料；未修改 protocol canonical、registry、split、checkpoint 或既有预测；仅更新中文入口和台账。
- 结果：DCLOOS 不再标记为“缺外部文件阻断”，而标记为“默认预算未完成、reduced 兼容结果可用”；这仍不能支持与 S2C 的同协议 SOTA 排名。
- 测试/检查：待本条记录后运行 research-state、development-log、data-tracking、compileall、ruff 和 diff 检查。
- 风险与下一步：若需要端到端主表，必须单独登记默认预算或统一 KIR/seed 的 DCLOOS 运行；不得将 KIR=.75/seed=888 reduced 结果与 S2C KIR=.50/5-seed fair 行直接比较。

## 2026-08-10：外部监督条件可视化补充

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 Git 提交、推送或清理。
- 目标：在已有四张总览图之外，单独展示 S2C Trainable、ADB 和 DCLOOS reduced 的 OOS F1/F1-All，同时把 KIR、seed、骨干、监督条件和预算写进图内，避免视觉上形成伪公平排名。
- 修改：`tools/analysis/build_experiment_comparison_overview_v2.py` 新增 `external_supervision_reference.csv` 和 `external_supervision_reference.png`；manifest schema 升为3并记录 DCLOOS recovery_metrics SHA256；更新 `EXPERIMENT_COMPARISON_OVERVIEW_V2.md` 与 `VISUAL_ANALYSIS_INDEX_V1.md`。
- 结果：DCLOOS reduced 为 OOS F1 `87.1`、F1-All `90.3`，但其 KIR=.75/seed=888、BERT、pseudo-OOS+外部 SQuAD OOS 合同与 S2C/ADB 不同；图明确标注“数值参考，不是统一 SOTA 排名”。
- 数据影响：只读取既有 recovery_metrics.json 和 fair/external summary；无训练、无测试选择、无 canonical/registry/split/checkpoint 修改。
- 测试/检查：总览脚本成功，新增 PNG 已人工检查可读性，manifest/source hash 已更新；后续执行全套维护检查。

## 2026-08-10：机制闭环分析包

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 Git 提交、推送或清理。
- 目标：把“当前到底比较哪个自有方法、Trainable-K1 的优势来自什么、MOGB-Fair 的低误接受为何伴随 Known 拒绝、哪些外部结果不能混排”收束为一份可复核的机制入口。
- 修改：新增 `tools/analysis/build_mechanism_closure_v1.py`、`docs/analysis/MECHANISM_CLOSURE_V1.md`、`results/analysis/mechanism_closure_v1/` 和 `figures/mechanism_closure_v1/`；更新 `docs/analysis/VISUAL_ANALYSIS_INDEX_V1.md` 与 `docs/EXPERIMENT_LEDGER.csv`。
- 数据影响：仅读取已冻结的 current fair matrix、Trainable/MOGB 逐样本转移摘要、MOGB 粒球风险摘要和 KIR 敏感性摘要；没有训练、调阈值、选 K、修改 canonical/registry/split/checkpoint 或覆盖历史 artifact。
- 结果：生成 63 行 fair performance、9 行 Trainable-MOGB transition、18 行 MOGB risk、21 行 KIR robustness、4 张机制图和 manifest。结论保持合同分层：Trainable-K1 是当前自有 fair 候选；不能据此声称超过完整 MOGB/DCLOOS 或历史 Cascade。
- 执行命令：`python tools/analysis/build_mechanism_closure_v1.py`。
- 测试：脚本成功；图表已人工检查；`pytest tests/unit -q` 为 `349 passed, 3 warnings`；`compileall`、`ruff check`、`check_research_state.py`、`check_development_log.py`、`check_data_tracking.py` 和 `git diff --check` 均通过。
- 风险与下一步：外部基线仍缺完整同协议多 seed；继续优先做可验证的外部单格/逐样本错误预算，不重复 E2/E3/K 网格。

## 2026-08-10：实验台账字段校正

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 Git 提交、推送或清理。
- 目标：修正三条 analysis-only 台账记录缺少 `distances` 字段导致的列位偏移，使 `status/planned_units/completed_units` 与实际证据可解析一致。
- 修改：`docs/EXPERIMENT_LEDGER.csv` 中 `statistical_stability_v1`、`cross_dataset_intent_risk_visuals_v1`、`cross_dataset_error_attribution_v1` 三行补入 `distances=mixed`；不改变任何结果文件。
- 验证：台账 105 行全部恢复合法状态；`check_research_state.py`、`check_development_log.py`、`check_data_tracking.py` 和 `git diff --check` 通过。
- 风险与下一步：本次仅修正元数据列位，不重新运行分析；后续新增台账行必须按 26 列 schema 校验。

## 2026-08-10：研究台账 schema 防回归

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 Git 提交、推送或清理。
- 目标：防止后续新增 analysis-only 记录再次因缺失 `distances` 等宽表字段而发生列位漂移。
- 修改：`tools/maintenance/check_research_state.py` 新增 26 列、字段数、数值状态和状态/距离错位检查；`tests/unit/test_research_state.py` 新增缺字段回归测试。
- 验证：台账 105 行、每行 26 列；研究状态检查通过；新增测试与研究状态单元测试 `4 passed`；ruff、compileall、development-log、data-tracking 和 diff 检查通过。
- 风险与下一步：校验器只检查元数据结构，不验证每个 analysis-only 数字的科学含义；实验结果仍需来源 manifest 和报告共同审计。

## 2026-08-10：原生 detector 机制归因

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 Git 提交、推送或清理。
- 目标：在同一 Trainable/Frozen MiniLM 表示上对照 MSP、Energy、kNN、LOF 与 Trainable-K1 Gate，区分表示训练收益和 Gate 决策收益。
- 修改：新增 `tools/analysis/build_detector_mechanism_v1.py`、`docs/analysis/DETECTOR_MECHANISM_ANALYSIS_V1.md`、`results/analysis/detector_mechanism_v1/`、`figures/detector_mechanism_v1/`；更新可视化索引和实验台账。
- 数据影响：只读取已完成的 KIR=.50、3 seed 原生 detector 控制；未训练、未调阈值、未修改 canonical/registry/split/checkpoint 或历史 artifact。
- 结果：生成 27 行 detector summary、12 行 Trainable-Frozen 表示差值、36 行 Gate-native 配对差值和 3 张图。StackOverflow 上 Trainable Gate OOS F1=`86.71%`，同一表示的 kNN/LOF/MSP/Energy 为 `68.83/63.53/45.61/42.95%`；Gate 的 false acceptance=`11.14%`，原生 detector 为 `44.59%--71.43%`，但 Known Recall 代价必须同时报告。
- 执行命令：`python tools/analysis/build_detector_mechanism_v1.py`。
- 风险与下一步：这是三 seed KIR=.50 analysis-only 控制，不替代五 seed 主矩阵；继续补逐样本 score/ROC/PR 错误交集和可验证外部基线。

## 2026-08-10：原生 detector 配对置信区间与 forest 图补充

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：在已有 Trainable/Frozen MiniLM detector 控制上，利用已登记的 paired bootstrap，明确区分表示收益和 Gate 收益，避免只报告三 seed 均值。
- 修改：扩展 `tools/analysis/build_detector_mechanism_v1.py`；新增 `detector_paired_ci.csv`、`representation_paired_ci.csv` 和两张 OOS F1 forest 图；更新 `DETECTOR_MECHANISM_ANALYSIS_V1.md`、`VISUAL_ANALYSIS_INDEX_V1.md`、`CURRENT_STATUS.md` 和台账。
- 数据影响：只读取已完成的 KIR=.50、3 seed 源表；不重新训练、不重新抽 bootstrap、不使用 test OOS 选参、不修改 canonical/registry/split/checkpoint 或历史 artifact。
- 结果：Gate 相对同一 Trainable 表示的 MSP/Energy/kNN/LOF 在三个数据集均为 3/3 seed 胜出，OOS F1 的 95% paired bootstrap 区间均为正；StackOverflow 相对 MSP/Energy 分别为 `+41.10/+43.76pp`。Trainable-Frozen 表示增益在 StackOverflow 的 MSP/Energy 为负、kNN/LOF 为正，说明收益来自表示与几何 Gate 的组合而非任意 detector 都改善。
- 执行命令：`python tools/analysis/build_detector_mechanism_v1.py`。
- 测试：`ruff check tools/analysis/build_detector_mechanism_v1.py` 通过；脚本输出 5 张 detector 图、2 个 48 行配对表和 schema=2 manifest；图像尺寸与 PNG 可读性检查通过。
- 风险与下一步：仍属于三 seed analysis-only 证据，不提升为五 seed 主矩阵或跨合同 SOTA；后续继续维护外部基线合同分层和逐样本错误分析。

## 2026-08-10：DCLOOS 端到端外部合同隔离 smoke

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：验证审稿人点名的 DCLOOS 官方代码、外部 SQuAD 负样本和当前 StackOverflow protocol 快照能否进入
  BERT 端到端训练/验证/测试链路；不把不同 Known-list 和外部 OOS 监督伪装成 fair 主表。
- 修改：扩展 `scripts/experiments/run_dcloos_official.py` 的运行时 overlay 兼容记录；新增
  `docs/analysis/DCLOOS_CONTRACT_STATUS_V1.md`；更新 `docs/analysis/BASELINE_EXECUTION_STATUS_V1.md`、
  `COMPARISON_ATLAS_V1.md`、`EXPERIMENT_ANALYSIS_MASTER_V1.md`、`EXPERIMENT_COMPARISON_ZH.md`、
  `docs/CURRENT_STATUS.md` 和 `docs/EXPERIMENT_LEDGER.csv`。
- 数据影响：只复制当前 protocol 的 train/dev/test 和已冻结外部 SQuAD 快照；负样本 SHA256 为
  `f6bf61866c86d3b11565826c3ca1faa00e31f196e0ad9bfd000ec45575fd426e`；没有修改 canonical、registry、
  E2/E3/R1/fair matrix 或第三方源码。
- 运行：100 epoch 长预算首轮遇到 Python 标量 `.item()`；后续 smoke 发现 epoch schedule 越界；最终
  `dcloos_official_stackoverflow_kir050_seed42_adapted_smoke_v4` 以 1 epoch、batch=500、n_oos=20、
  convex=20/10 完整导出 metrics/predictions。
- 结果：OOS F1=`0`、F1-All=`8.31%`、Known Recall=`100%`（低预算初始模型几乎不拒 OOS），只作为链路
  验证；当前 registry 的 10 个 Known intent 与 DCLOOS 自身抽取的 5 个类别不一致，状态为
  `complete_smoke_contract_mismatch`，不进入任何性能排名。
- 测试/检查：smoke artifact manifest、metrics、predictions 存在；下一步维护检查需验证 ledger 26 列、
  文档引用和 runner compile；未运行完整 DCLOOS 训练矩阵。
- 风险与下一步：先实现固定 `known_labels_file` adapter，再运行一个收敛单格；在此之前不启动 DCLOOS
  多 seed、不把 reduced/smoke 数字与 S2C/MOGB/ADB 混排，也不声称 SOTA。

## 2026-08-10：将 detector 配对区间纳入综合中文入口

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 Git 提交、推送或清理。
- 目标：让综合实验入口明确区分“表示训练收益”和“Gate 决策收益”，避免用户只看到 OOS F1 均值而误解当前方法。
- 修改：更新 `docs/analysis/EXPERIMENT_ANALYSIS_MASTER_V1.md` 与 `docs/CURRENT_METHOD_AND_EXPERIMENT_STATUS.md`，链接 detector 配对区间报告和新增 forest 图。
- 数据影响：只增加已有 analysis-only 证据的导航和解释；没有训练、重采样、阈值选择或历史 artifact 修改。
- 结果：综合入口现在明确记录：同一 Trainable MiniLM 表示下 Gate 相对四种原生 detector 在三个数据集均为 3/3 seed 胜出，但 Trainable-Frozen 原生 detector 增益在 StackOverflow 并非全方向为正。
- 测试：文档路径、manifest、研究状态和开发日志检查在本轮收口时统一执行。
- 风险与下一步：外部 BERT/OOS 监督合同仍不能混入 fair 主表；继续做同协议逐样本错误和工作点分析。

## 2026-08-10：外部基线状态与活动分析入口一致性修正

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 Git 提交、推送或清理。
- 目标：清理活动中文报告中已过期的“DCLOOS 外部文件仍缺失/同协议所有外部指标均不可用”表述，避免把历史 runtime 记录误读为当前状态。
- 修改：更新 `docs/analysis/COMPARISON_ATLAS_V1.md`、`docs/analysis/MOGB_OPERATING_POINT_VISUALS_V1.md` 和 `docs/analysis/EXPERIMENT_ANALYSIS_MASTER_V1.md`。
- 当前状态：ADB 三个 StackOverflow/KIR=.50 BERT/TextOIR 兼容单元有效；DA-ADB 仍因 NaN/全类预测无效；DCLOOS 外部 SQuAD 来源已闭合但默认预算超时，只有不同监督合同的 reduced 单元。
- 数据影响：仅修正文档合同分层；没有训练、重评分、参数选择或历史 artifact 修改。
- 验证：文档引用的 manifest 和结果文件均已存在；后续统一运行维护检查。
- 风险与下一步：仍不能将这些外部单元与 Known-only MiniLM fair 行混排；继续使用逐样本错误和工作点分析支撑机制结论。

## 2026-08-10：跨数据集逐意图风险证据接入综合入口

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 Git 提交、推送或清理。
- 目标：把已有三数据集逐样本错误归因纳入综合报告，解释 StackOverflow、CLINC150、Banking77 的多中心风险为何不同。
- 修改：更新 `docs/analysis/EXPERIMENT_ANALYSIS_MASTER_V1.md`，链接 `CROSS_DATASET_ERROR_ATTRIBUTION_V1.md` 和 `CROSS_DATASET_INTENT_RISK_VISUALS_V1.md`。
- 证据：已有跨数据集审计覆盖 `1,890,000` 条对齐记录；StackOverflow/KIR=.50 的 Frozen K=2 前五个 OOS 吸收 intent 合计约 `64.9%`，支持少数边界吸收器主导的 union-risk 解释。
- 数据影响：只重排已有逐样本结果，不训练、不调参、不使用 test 结果选择新配置、不修改历史 artifact。
- 验证：源 CSV、PNG 和 manifest 均存在；维护检查在本阶段统一执行。
- 风险与下一步：逐意图风险是后验机制证据，不是自适应 K 选择器；继续保持外部 baseline 合同分层。
## 2026-08-10：Trainable Gate 与旧 Cascade 合同断点审计

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：确认当前 protocol_v2 Trainable K=1 是否能直接替换旧 v19 Cascade 的 Gate，并阻止不合法的
  Gate→Router→Expert 混合结果进入比较表。
- 修改：新增 `tools/analysis/build_cascade_trainable_contract_gap_v1.py`、
  `docs/analysis/CASCADE_TRAINABLE_CONTRACT_GAP_V1.md`、
  `results/analysis/cascade_trainable_contract_gap_v1/` 和
  `figures/cascade_trainable_contract_gap_v1/split_count_contract_gap.png`；同步状态入口、证据索引和台账。
- 数据影响：只读取 Trainable manifest、当前 protocol 的 gate split 和旧 v19 component plan；未训练、未重建
  canonical/registry/views、未修改 E2/E3/R1/MOGB/DCLOOS 结果。
- 结果：Trainable 当前协议为 train/calibration/test=`6000/1000/6000`，旧 v19 为 `5995/1998/5990`；
  当前 test 有 `sample_id`，旧 v19 gate JSON 没有当前 `sample_id`。直接替换 Gate 会同时改变输入样本、Known
  列表、split 和下游分布，因此当前 Trainable 只能报告 Gate-only。
- 执行命令：`python tools/analysis/build_cascade_trainable_contract_gap_v1.py`。
- 验证：脚本 compileall、研究状态、开发日志、数据跟踪和 `git diff --check` 均需通过；没有提交或删除历史 artifact。
- 风险与下一步：必须在 protocol_v2 registry/views 上重新训练 Router/Expert，先运行 StackOverflow/KIR=.50、
  seeds=13/42/87 的 Frozen K=1 Cascade vs Trainable K=1 Cascade；不得复用 v19 下游组件冒充同协议结果。

## 2026-08-10：当前协议 Cascade bridge 完成

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：在同一 `protocol_v2_textoir_v1` StackOverflow/KIR=.50 数据合同内，重新训练一个 Known-only SmolLM Expert，配对比较 Frozen K=1 与 Trainable K=1 Gate，消除旧 v19 下游组件混入。
- 修改：新增 `scripts/experiments/run_protocol_v2_cascade_bridge_v1.py`、`tools/analysis/build_cascade_bridge_v1.py`、
  `docs/analysis/CASCADE_BRIDGE_V1.md`、`results/analysis/cascade_bridge_v1/` 和
  `figures/cascade_bridge_v1/`；实验台账追加 `cascade_bridge_v1`。
- 数据影响：只读取当前 protocol 的 train_known/calibration_known/test_combined 和既有 Gate 预测；每个 seed 的 Expert 仅用 Known train 训练、Known calibration 选 checkpoint；不读取旧 v19 Router/Expert，不修改 E2/E3/R1/MOGB/DCLOOS artifacts。
- 执行命令：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /home/bo/anaconda3/envs/bo/bin/python scripts/experiments/run_protocol_v2_cascade_bridge_v1.py --seeds 13 42 87 --epochs 5 --batch-size 32 --device cuda --resume`；`/home/bo/anaconda3/envs/bo/bin/python tools/analysis/build_cascade_bridge_v1.py`。
- 实验数量：6/6 评价行，3 个 seed Expert checkpoint，0 failed/missing/duplicate；`test_used_for_selection=false`、`oos_used_for_training=false`。
- 结果：Trainable K=1 相对 Frozen K=1 的 OOS F1 `+9.42pp`、F1-All `+6.70pp`、Accuracy `+8.57pp`、Known Recall `+0.21pp`、false acceptance `-15.40pp`；该结果只证明同协议下 Gate 改善能传递到下游，不是历史 fulltex 或外部 SOTA 复现。
- 风险与下一步：仅覆盖 StackOverflow/KIR=.50，StackOverflow Router 为常量路由；仍不能与历史 Cascade、官方 BERT MOGB 或带外部 OOS 监督的 DCLOOS 混排。下一步应先完成合同分层的主表与可视化，再决定是否扩展其他数据集。

## 2026-08-10：合同分层比较图谱 V2

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：把当前 fair Gate、当前协议 Cascade bridge、MOGB MiniLM 组件、ADB 兼容单格和 MOGB 论文/本地差距放入同一可读入口，同时保持监督、骨干和系统层级隔离。
- 修改：新增 `tools/analysis/build_comparison_atlas_v2.py`、`docs/analysis/COMPARISON_ATLAS_V2.md`、
  `results/analysis/comparison_atlas_v2/`、`figures/comparison_atlas_v2/`；实验台账追加 `comparison_atlas_v2`。
- 数据影响：只读取既有汇总、MOGB gap 和 cascade_bridge 结果；没有重训、调参、改变 split、覆盖历史 artifact 或把 DCLOOS reduced 数字加入 Known-only fair rows。
- 结果：生成 10 条合同分层行和 3 张图；明确当前自有方法的 Gate 与 Cascade 层级，显示 Trainable K=1 的当前协议下游增益，并把 MOGB 论文差距拆成损失、半径、selected-ball 和数据合同因素。
- 执行命令：`/home/bo/anaconda3/envs/bo/bin/python tools/analysis/build_comparison_atlas_v2.py`。
- 验证：manifest、源哈希、研究状态、开发日志、数据跟踪和 `git diff --check` 均检查；无新增模型运行。
- 风险与下一步：该图谱仍不是跨合同 SOTA 排名；下一步继续补充同协议可验证的 baseline 证据和逐样本机制图，不重复 E2/E3/R1/MOGB 旧矩阵。

## 2026-08-10：Cascade Gate/Expert 误差预算分析

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：解释同一 protocol_v2 Expert 下 Frozen/Trainable K=1 Cascade 的差异来自 Gate 误收、Gate 误拒还是 Expert 分类错误。
- 修改：新增 `tools/analysis/build_cascade_error_budget_v1.py`、`docs/analysis/CASCADE_ERROR_BUDGET_V1.md`、
  `results/analysis/cascade_error_budget_v1/` 和 `figures/cascade_error_budget_v1/`；实验台账追加 `cascade_error_budget_v1`。
- 数据影响：对齐已完成的 6 条 Gate/Cascade 逐样本预测，检查 sample_id 完全一致；无训练、无调参、无测试 OOS 选参、无历史 artifact 修改。
- 结果：生成 6 个 per-seed 误差预算行和 24 个转移行；Trainable 的主要收益仍来自降低 Gate OOS false acceptance，Known false rejection 变化很小，Expert 不是主要误差来源。
- 执行命令：`/home/bo/anaconda3/envs/bo/bin/python tools/analysis/build_cascade_error_budget_v1.py`。
- 验证：输出、图、报告和 ledger 已生成；后续只将它作为当前协议机制证据，不混入历史 fulltex/MOGB/DCLOOS 排名。

## 2026-08-10：三数据集当前协议 Cascade 桥接

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：把 StackOverflow 之外的 CLINC150、Banking77 接入同一当前协议 Cascade，验证 Trainable K=1 的收益是否依赖单一数据集或 Expert 偶然性。
- 修改：新增 `scripts/experiments/run_protocol_v2_multidataset_cascade_bridge_v1.py`、`tools/analysis/build_cascade_bridge_cross_dataset_v1.py`、`docs/analysis/CASCADE_BRIDGE_CROSS_DATASET_V1.md`、`results/analysis/cascade_bridge_cross_dataset_v1/` 和 `figures/cascade_bridge_cross_dataset_v1/`。
- 数据影响：复用现有 protocol_v2 views、E2 Frozen K=1 预测和 Trainable control 预测；逐数据集/seed 检查 test sample_id 完全相等；Expert 只读 train_known，calibration_known 选 epoch，未使用 OOS 训练或测试选择。
- 结果：新增 12 个评价单元（2 数据集×3 seed×2 Gate），与 StackOverflow 既有 6 个单元合并生成 18 行跨数据集报告。Trainable 相对 Frozen 的 Cascade OOS F1 配对差值为 Banking77 `+5.18pp`、CLINC150 `+1.12pp`、StackOverflow `+9.42pp`；对应 FA 降幅为 `10.00pp`、`2.86pp`、`15.40pp`。
- 执行命令：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /home/bo/anaconda3/envs/bo/bin/python -u scripts/experiments/run_protocol_v2_multidataset_cascade_bridge_v1.py --datasets clinc150 banking77 --seeds 13 42 87 --epochs 5 --batch-size 32 --device cuda --resume`；`MPLBACKEND=Agg /home/bo/anaconda3/envs/bo/bin/python -c 'import tools.analysis.build_cascade_bridge_cross_dataset_v1 as x; x.main()'`。
- 验证：训练输出、逐样本 ID 对齐、汇总、误差预算和图均生成；后续仍须运行完整研究状态/数据跟踪/编译检查。该结果是当前协议下游证据，不是历史 fulltex、MOGB 论文或 DCLOOS 的公平 SOTA 排名。

## 2026-08-10：MSP 外部基线运行时探测阻塞

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：在当前 StackOverflow/KIR=.50/seed=42 数据合同上启动 TEXTOIR MSP 前，先验证外部 Python/PyTorch runtime，避免把环境故障误写成算法结果。
- 修改：更新 `docs/analysis/BASELINE_EXECUTION_STATUS_V1.md` 和 `docs/EXPERIMENT_LEDGER.csv`；保留独立 artifact `../artifacts/s2c/external/msp_protocol_v2_stackoverflow_kir50_seed42_v1/`。
- 数据影响：dry-run 已核对 10 个 Known labels、train/dev/test SHA256 和独立数据根；没有训练、预测、checkpoint、参数选择或历史 artifact 修改。
- 执行命令：`/home/bo/anaconda3/bin/python tools/compat/textoir/run_external_textoir.py --textoir-root ../textoir --data-root ../artifacts/s2c/external/adb_protocol_v2_probe_data_v1 --known-labels-file ../artifacts/s2c/external/adb_protocol_v2_probe_data_v1/stackoverflow/known_labels.json --dataset stackoverflow --method MSP --known-cls-ratio 0.50 --seed 42 --bert-model ../artifacts/s2c/external/bert-base-uncased-pytorch --python-executable tools/compat/textoir/runtime_shims/python_with_easydict --run-dir ../artifacts/s2c/external/msp_protocol_v2_stackoverflow_kir50_seed42_v1 --gpu-id 0`。
- 结果：环境探针 90 秒超时，训练未开始；台账标记 `blocked_runtime_probe`，而非 MSP 性能失败。
- 验证：保留 `run_manifest.json`，其状态为 `dry_run`；后续不得将该单元加入比较排名，只有隔离 runtime 通过 torch import、BERT forward 和单批 CPU/GPU smoke 后才允许重试。
- 风险与下一步：MSP 仍无可审计指标；当前比较继续使用已有 315 行 fair matrix、18 行 protocol_v2 Cascade bridge、MOGB gap 和有效 ADB 单元，避免重复旧矩阵。

- 补充：同步更新中文对比入口 `docs/对比实验/MOGB_DCLOOS_对比结果报告.md`，加入三数据集 Cascade bridge 和 MSP runtime 状态；仅更新证据索引，不改变任何结果或排名。

- 补充：同步更新 `docs/analysis/VISUAL_ANALYSIS_INDEX_V1.md`，把三数据集 Cascade bridge 的三张图纳入性能—决策可视化索引；没有新增训练或重新选择配置。

## 2026-08-10：五 seed 可视化证据包 V2

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：消除旧可视化索引中早期三 seed 数字与当前五 seed 主矩阵并存造成的歧义，建立一个只引用当前权威结果的中文入口。
- 修改：新增 `docs/analysis/EXPERIMENT_VISUAL_EVIDENCE_BUNDLE_V2.md`，同步 `docs/CURRENT_STATUS.md`；报告链接当前 fair matrix、MOGB 机制图、Cascade bridge 图和外部合同状态。
- 数据影响：仅读取已完成的 315 行 fair matrix、45 个 MOGB 配对单元、18 个 Cascade bridge 单元和已有外部基线结果；没有训练、重评分、调参或修改历史 artifact。
- 结果：V2 明确给出 Trainable K=1 的五 seed 均值、Known/OOS 错误预算、MOGB 论文差距和跨数据集 Cascade 配对差值；旧 V1 图索引保留作为历史索引，但不再作为当前数字来源。
- 验证：报告引用的 CSV、PNG 和 manifest 均存在；研究状态、开发日志、数据跟踪和 `git diff --check` 继续作为收口检查。
- 风险与下一步：仍不能把不同 backbone/监督合同混成 SOTA 排名；后续实验继续优先补可运行外部 baseline 和错误机制证据。

- 补充：将已完成的同一 Trainable MiniLM 表示下 MSP、Energy、kNN、LOF 检测器对照加入 V2 可视化证据包；只读取 36 个已完成 detector 单元和配对 bootstrap 结果，没有新增训练或测试选参。

## 2026-08-10：外部 baseline runtime 再探针

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：在不启动训练的前提下确认 MSP/ADB/DA-ADB 是否已有可用 Python/PyTorch runtime。
- 执行：分别对当前 base、`bo` 和 `textoir-py39` 候选解释器运行最小 `import torch`/CUDA 探针；实际首个探针在 import 阶段进入 D-state，20 秒 timeout 无法回收。
- 结果：没有 BERT forward、训练、预测或指标；外部 baseline 继续标记 `runtime_blocked_no_metrics`，不能作为算法失败或性能数字。
- 风险与下一步：避免重复启动会进入 D-state 的外部训练；只有隔离 runtime 通过 torch import、BERT forward 和单批 CPU/GPU smoke 后才恢复外部单格。

## 2026-08-10：历史合同分层图字体修复

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：修复 `stackoverflow_contract_layers.png` 中中文字体缺失造成的方框和颜色分层错误，保证历史 fulltex Cascade 与当前 protocol_v2 Gate 的合同边界可读。
- 修改：`tools/analysis/build_historical_sota_comparison_v1.py` 将图内标签改为 ASCII/英文并修正历史/当前合同颜色判断；重新生成同一数据源图和 manifest。
- 数据影响：仅重新渲染 72 条历史表和 63 条当前汇总，不改变任何数值、split、模型或结果文件语义。
- 验证：重新运行分析脚本；PNG 通过文件格式检查并人工视觉检查，标题、图例、历史/当前颜色分层均正确。

## 2026-08-10：跨合同性能差距审计与单一阅读入口

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：解决“当前到底在比较哪个方法”的歧义，逐格对齐历史 `fulltex.tex` 完整 Cascade 与当前 `protocol_v2_textoir_v1` 的 Trainable-K1，同时保留 MOGB、ADB、DCLOOS 的监督/骨干合同边界。
- 新增：`tools/analysis/build_cross_contract_gap_v1.py`、`docs/analysis/CROSS_CONTRACT_GAP_V1.md`、`results/analysis/cross_contract_gap_v1/`、`figures/cross_contract_gap_v1/current_vs_historical.svg`；同步 `docs/CURRENT_STATUS.md` 与 `docs/analysis/EXPERIMENT_COMPARISON_OVERVIEW_V2.md`。
- 数据影响：只读取已完成的历史 72 行、当前 63 行和外部合同 CSV；没有训练、重评分、阈值选择、测试 OOS 调参或修改既有 artifact。
- 结果：当前 Trainable-K1 在同一九格 fair Gate 矩阵中 OOS F1 85.75%（8/9 第一）；与历史完整 Cascade 的描述性差值为 3/9 格胜出、平均 -3.42pp。该差距明确标记为合同/系统层级差距，不作为跨合同 SOTA 排名。
- 验证：`/usr/bin/python3 tools/analysis/build_cross_contract_gap_v1.py` 成功生成 9 行 CSV、manifest 和 SVG；源文件 SHA256 写入 manifest；随后需运行研究状态、开发日志、数据跟踪和 `git diff --check`。
- 风险与下一步：外部 ADB/DA-ADB/DCLOOS 仍不能直接并入 Known-only fair 主表；后续只在独立 runtime 通过探针后补有效单格，避免重复冻结矩阵。
- 补充：跨合同 CSV 进一步加入 F1-K/历史 Known F1 和 Accuracy 的逐格差值；StackOverflow/KIR=.50 当前 F1-K 高于历史 Known F1 约 10.96pp、Accuracy 高约 1.27pp，说明 OOS F1 差距不能简单归因于当前表示分类能力不足。

## 2026-08-10：外部 baseline 解释器穷举探针

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：确认 ADB/DA-ADB/MSP 的 runtime blocker 是否只来自单一 Python 环境。
- 执行：对 `bo`、`textoir-py39`、`implicit_intent` 运行 8 秒 `import torch` 探针；对 `dreamer` 检查 PyTorch 安装；系统 Python 仅作模块存在性检查。
- 结果：前三个解释器均超时，`dreamer` 无 PyTorch，系统 Python 无 PyTorch；没有启动 BERT forward、训练、预测或新指标。
- 修改：补充 `docs/analysis/BASELINE_EXECUTION_STATUS_V1.md` 的 runtime 证据；不改变任何实验 artifact、registry 或结果。
- 验证：探针退出状态已记录；外部方法继续按 `valid_same_protocol_external_cell`、`invalid`、`reduced/adapted` 分层，不进入混合排名。
- 风险与下一步：继续重复当前解释器不会产生新信息；必须先建立可回收的独立 PyTorch 环境，再恢复外部单格实验。

## 2026-08-10：Trainable-MOGB 错误预算重算与可视化

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：用 45 个同 split Trainable↔MOGB-Fair 配对单元回答性能差异主要来自 Known 覆盖恢复还是 OOS 拒识变化。
- 新增：`tools/analysis/build_trainable_mogb_error_budget_v1.py`、`docs/analysis/TRAINABLE_MOGB_ERROR_BUDGET_V1.md`、`results/analysis/trainable_mogb_error_budget_v1/`、`figures/trainable_mogb_error_budget_v1/error_budget_decomposition.svg`；同步可视化索引。
- 数据影响：只读取冻结的 `cell_decomposition_per_seed.csv`，没有训练、阈值选择、测试 OOS 调参或修改逐样本 artifact。
- 结果：45/45 行的 `Known gain + OOS gain = total gain` 算术审计通过；9 个 dataset×KIR 组均有正向 Known 正确恢复，OOS 正确拒绝变化通常为负，但 F1-All 增量均为正。StackOverflow/KIR=.50 平均恢复约 1,682 个 Known 正确、减少约 256 个 OOS 正确拒绝。
- 验证：stdlib 脚本生成 9 行汇总、integrity JSON、SVG；后续运行研究状态、开发日志、数据跟踪、lint 和 `git diff --check`。
- 风险与下一步：该结果只支持同合同 MOGB-Fair 机制解释，不能外推到完整 BERT MOGB 论文；外部基线仍等待可用 PyTorch runtime。

## 2026-08-10：Trainable-MOGB KIR 趋势分析

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：检查 Trainable-K1 相对 MOGB-Fair 的优势是否只属于某个 KIR，还是随 Known intent 比例系统变化。
- 新增：`tools/analysis/build_trainable_mogb_kir_trend_v1.py`、`docs/analysis/TRAINABLE_MOGB_KIR_TREND_V1.md`、`results/analysis/trainable_mogb_kir_trend_v1/`、`figures/trainable_mogb_kir_trend_v1/kir_trend.svg`；同步可视化索引。
- 数据影响：只读取 3 数据集×3 KIR×5 seed 的现有 `summary_mean_std.csv`，没有训练或测试集选参。
- 结果：9/9 单元的 OOS F1、F1-All 和 Known Recall 差值均为正；StackOverflow 的 OOS F1 差值由 KIR=.25 的 +6.04pp 增至 .75 的 +30.72pp，F1-All 差值由 +33.57pp 增至 +54.19pp。
- 验证：脚本生成 9 行趋势 CSV 和 SVG；`ruff`、研究状态、开发日志、数据跟踪和 `git diff --check` 通过。
- 风险与下一步：KIR 趋势只支持同合同 Trainable/MOGB-Fair 机制解释，不能外推为完整论文 MOGB 或 DCLOOS 的 SOTA 结论。

## 2026-08-10：实验结论口径审计

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：解决大量历史/当前/外部报告并存时“当前到底比较哪个方法、哪些结论可以写”的歧义。
- 新增：`docs/analysis/RESULT_CLAIM_AUDIT_V1.md`；同步 `docs/CURRENT_STATUS.md` 与可视化索引。
- 数据影响：只审阅现有报告和 manifest，没有训练、重评分、选参或修改结果。
- 结果：明确 `S2C-Trainable-K1`、历史 fulltex Cascade、MOGB-Fair、官方 MOGB、ADB、DA-ADB、DCLOOS reduced 的对象和禁止混排规则。
- 验证：研究状态、开发日志、数据跟踪和 `git diff --check` 继续作为收口检查。
- 风险与下一步：外部基线仍缺同 runtime/同监督主表；在 runtime 修复前不宣称跨合同 SOTA。

## 2026-08-10：实验决策面板与独立 CUDA runtime

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：把当前 fair Gate、MOGB 组件、MOGB 官方逻辑单格和外部参照压缩成一个不混合同的实验入口，并恢复可审计的外部 BERT 单格运行条件。
- 新增：`tools/analysis/build_experiment_decision_dashboard_v1.py`、`docs/analysis/EXPERIMENT_DECISION_DASHBOARD_V1.md`、`results/analysis/experiment_decision_dashboard_v1/`、`figures/experiment_decision_dashboard_v1/`；外部比较工具新增 `--external-root`，以便隔离运行根复用同一评价审计。
- 数据影响：决策面板只读取已有 63 个 fair summary 行、MOGB exact summary 和 DCLOOS/ADB reference；没有训练、阈值选择、测试标签选择或历史 artifact 覆盖。
- 新实验：使用隔离 `/tmp/s2c_gpu_runtime_20260810`（PyTorch `2.9.1+cu128`、RTX 5070）完成 StackOverflow/KIR=.50/seed=42 的 ADB BERT/TextOIR 单格；run manifest 为 `../artifacts/s2c/external/adb_gpu_runtime_v1/stackoverflow/ADB/kir50/seed42/run_manifest.json`，状态 `complete`，逐样本预测和结果文件均存在。
- 结果：以 `y_true.npy/y_pred.npy` 重算 ADB OOS F1=`85.95%`、F1-All=`84.25%`、Known Recall=`79.30%`、false acceptance=`9.03%`；与当前 Trainable-K1 的差异为 OOS F1 `+1.60pp`、F1-All `+1.52pp`、Known Recall `+4.20pp`，但仍是 BERT 外部合同，不能并入 MiniLM fair 排名。
- 验证：CUDA 单张量 smoke、TextOIR environment/method preflight、artifact audit、外部逐样本指标重算和 `build_external_single_cell_comparison_v1.py --external-root ...` 均已运行；seed=87/100 尚在独立运行队列。
- 风险与下一步：CUDA runtime 解决了旧环境导入阻断，但 RTX 5070 需要 CUDA 12.8；先完成 ADB 三 seed，再尝试 DA-ADB 单格。任何 DA-ADB NaN/全类预测仍保持 invalid，不生成伪排名；DCLOOS 继续保持额外 OOS 监督合同。

## 2026-08-10：ADB 隔离 CUDA 三 seed 收口与决策面板更新

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：完成 StackOverflow/KIR=.50/seed={42,87,100} 的 ADB 外部合同参照，并把最新结果接入不混合同的实验决策面板。
- 新增：`docs/analysis/ADB_GPU_RUNTIME_THREE_SEED_V1.md`；更新 `tools/analysis/build_experiment_decision_dashboard_v1.py`、`docs/analysis/EXPERIMENT_DECISION_DASHBOARD_V1.md`、`results/analysis/external_gpu_runtime_comparison_v1/` 和 `figures/experiment_decision_dashboard_v1/`；追加 `baseline_external_adb_gpu_runtime_v1` 台账行。
- 运行：隔离 `/tmp/s2c_gpu_runtime_20260810`，PyTorch `2.9.1+cu128`、Transformers `4.46.3`、RTX 5070；三个 run manifest 均为 `complete`，逐样本 y_true/y_pred audit 完整。
- 结果：ADB OOS F1=`87.36±1.61%`、F1-All=`85.66±1.59%`、Known Recall=`80.78±1.44%`、FA=`7.52±1.97%`；同 seed S2C Trainable K=1 高 `+0.85pp` OOS F1、`+1.03pp` F1-All、`+2.90pp` Known Recall，但 FA 高 `+0.66pp`。
- 解释：ADB 使用 BERT/TextOIR 外部合同，不能与 MiniLM fair Gate 合并排名；该结果只验证了可审计外部参照已恢复，不代表跨合同 SOTA。
- 验证：外部比较审计生成 3 个 external rows、35 个 fair rows、0 invalid；dashboard manifest 更新为 63 fair rows、6 external references；ruff、py_compile、`git diff --check` 已通过。
- 风险与下一步：DA-ADB 仍为 invalid，DCLOOS 仍为额外 pseudo/external-OOS 监督；继续补齐外部基线时保持合同隔离，不重复 E2/E3 或修改历史 artifacts。

## 2026-08-10：基线合同图切换到最新 ADB 三 seed 汇总

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：修正早期合同可视化仍引用旧 ADB 单格的口径，统一改用最新隔离 CUDA 三 seed 审计。
- 修改：`tools/analysis/build_baseline_contract_visuals_v1.py`；重生成 `docs/analysis/BASELINE_CONTRACT_VISUALS_V1.md`、`results/analysis/baseline_contract_visuals_v1/` 和 `figures/baseline_contract_visuals_v1/`；追加 `baseline_contract_visuals_refresh_v2` 台账行。
- 结果：ADB 图表点更新为 OOS F1=`87.36%`、F1-All=`85.66%`、3 seeds；DA-ADB、BRAK、MOGB 官方行继续保留历史合同标签，不与 MiniLM fair 行合并。
- 验证：输出 manifest SHA256=`855fc5074a4e0c88d79aea09ae71fafbb587b4fa59000656db92e9a19a888c26`；分析只读现有 CSV，无训练、调参、测试选择或历史 artifact 覆盖。

## 2026-08-10：ADB Banking77 KIR=.50 三 seed 外部对比

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：把 ADB 外部合同从 StackOverflow 扩展到 Banking77 的同 KIR、同 seed 配对，观察差异是否具有数据集依赖。
- 修改/新增：`tools/compat/textoir/build_protocol_data_root.py` 增加 legacy runtime dataset alias；新增 `tools/analysis/build_adb_cross_dataset_v1.py`、`results/analysis/adb_cross_dataset_v1/`、`figures/adb_cross_dataset_v1/`、`results/analysis/adb_cross_dataset_v1/ADB_CROSS_DATASET_REPORT.md`；追加台账行。
- 运行：Banking77/KIR=.50/seed=`42,87,100`，隔离 `/tmp/s2c_gpu_runtime_20260810`，BERT/TextOIR ADB；3/3 manifest complete、逐样本预测可审计。
- 结果：ADB OOS F1=`74.30±1.50%`、F1-All=`78.18±1.67%`、Known Recall=`89.25±0.61%`、FA=`34.68±2.37%`；同 seed S2C Trainable K=1 高 `+9.79pp` OOS F1、`+3.68pp` F1-All，FA 低 `19.89pp`，但 Known Recall 低 `7.13pp`。
- 解释：Trainable 与 ADB 的差异在 Banking77 与 StackOverflow 方向不同，说明不能用单个 StackOverflow 结论代表所有数据集；差异仍同时包含 BERT/MiniLM 表示、训练和边界合同。
- 验证：`ruff`、`py_compile`、ADB 跨数据集汇总脚本均通过；图表包含绝对工作点和同 seed 配对 OOS F1 差值；没有测试集选参或历史 artifact 覆盖。

## 2026-08-10：ADB 三数据集 KIR=.50 外部参照收口

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：完成 CLINC150 ADB seed=`42,87,100`，与已有 Banking77、StackOverflow 外部运行合并，形成三数据集同 KIR 外部合同参照。
- 修改/新增：扩展 `tools/analysis/build_adb_cross_dataset_v1.py` 支持多个 runtime root；更新 `results/analysis/adb_cross_dataset_v1/`、`figures/adb_cross_dataset_v1/`、`results/analysis/adb_cross_dataset_v1/ADB_CROSS_DATASET_MANIFEST.json`；追加 `adb_cross_dataset_v2` 台账行。
- 运行：隔离 `/tmp/s2c_gpu_runtime_20260810`，PyTorch `2.9.1+cu128`、Transformers `4.46.3`、RTX 5070；CLINC150、Banking77、StackOverflow 各 3 seed，共 9/9 manifest complete。
- 结果：ADB OOS F1=`90.01±0.45%`（CLINC150）、`74.30±1.50%`（Banking77）、`87.36±1.61%`（StackOverflow）；Trainable K=1 同 seed OOS F1 差值分别为 `+0.35pp`、`+9.79pp`、`+0.85pp`，F1-All 差值为 `-4.14pp`、`+3.68pp`、`+1.03pp`。这说明外部差距具有数据集依赖，且 Trainable-K1 并非三数据集综合指标都占优。
- 解释：ADB 是端到端 BERT/TextOIR 外部合同；Trainable-K1 是 Known-only MiniLM。结果用于工作点与机制参照，不并入 MiniLM fair 主排名，也不支持跨合同 SOTA 声明。
- 验证：从审计后的 `y_true.npy/y_pred.npy` 重算指标；生成 9 行逐 seed、63 行配对差值和三数据集摘要；`ruff`、`py_compile`、数据跟踪、研究状态和开发日志检查待本批收口时复核。
- 风险与下一步：MOGB 官方论文值和 DCLOOS 额外 OOS 监督仍需保持合同隔离；不重复 E2/E3，不用测试集选择阈值或模型。

## 2026-08-10：ADB 跨 KIR 敏感性实验启动

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：在已有三数据集 KIR=.50 ADB 外部参照上，补跑 KIR=.25/.75，判断 Trainable-K1 与 ADB 的工作点差异是否具有数据集/KIR 依赖。
- 计划：3 数据集 × 3 KIR × 3 seed = 27 个 ADB BERT/TextOIR 单元；KIR=.50 的 9 个 anchor 已完成，本批新增 18 个单元。
- 新增：每个数据集/KIR/seed 使用独立 `../artifacts/s2c/external/adb_gpu_runtime_data_v3/` 数据 root 和 `adb_gpu_runtime_v3/` run root；新增汇总脚本 `tools/analysis/build_adb_kir_sensitivity_v1.py`，结果输出至 `results/analysis/adb_kir_sensitivity_v1/` 和 `figures/adb_kir_sensitivity_v1/`；追加 `adb_kir_sensitivity_v1` 台账行。
- 合同：固定 protocol_v2 split、Known 列表和 BERT/TextOIR ADB 配置；不使用 test OOS 选 epoch、阈值或参数；不修改已有 KIR=.50 artifacts。
- 状态：截至启动时已完成 9/27，新增 18 个按 seed 顺序串行运行；完成后再生成跨 KIR 曲线、配对热力图和中文报告。
- 风险与下一步：该批仍是外部 BERT 合同，不是 MiniLM 同骨干 fair 主表；若任意 run 出现 invalid prediction 或 split hash 不符，标记失败并停止汇总。

## 2026-08-10：ADB 跨 KIR 敏感性实验收口

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 结果：新增 KIR=.25/.75 的 18 个单元全部完成，连同 KIR=.50 anchor 共 `27/27` 有效；逐 seed、配对差值和三面板曲线写入 `results/analysis/adb_kir_sensitivity_v1/`、`figures/adb_kir_sensitivity_v1/`。
- ADB OOS F1 随 KIR 增加而下降：CLINC150 `91.53→90.01→85.09`、Banking77 `83.83→74.30→65.64`、StackOverflow `93.36→87.36→73.08`（KIR=.25/.50/.75）。
- Trainable-K1 的 OOS F1 在 9 个 dataset×KIR 组中 8 个高于 ADB；但 CLINC150 的 F1-All/ Known Recall 在三个 KIR 都低于 ADB，Banking77 KIR=.75 的 F1-All 也低于 ADB。该结果排除了“当前方法在所有数据集和开放程度上全面领先”的表述。
- 解释：KIR 提高时 ADB 的 OOS 拒识工作点明显恶化，但 Trainable 与 ADB 的差值仍依赖数据集；StackOverflow 的接近/领先不能外推为跨数据集 SOTA。ADB 仍是 BERT/TextOIR 外部合同，不并入 MiniLM fair 主表。
- 验证：27 个 run manifest 均 `complete`，逐样本预测存在；指标从 `y_true.npy/y_pred.npy` 重算；`selection_used_test_oos=false`；脚本 `ruff`/编译检查通过。输出 manifest SHA256=`b18e7b56c2a9205f3ac1d755e050aa33a64de6bd74ebd48ece3b8c5568b35272`。
- 下一步：将跨 KIR ADB 曲线与已有 MOGB/Trainable 机制图合并到中文实验总览；不重复 E2/E3，不把外部 BERT 结果写成 MiniLM 同合同 SOTA。

## 2026-08-10：跨 KIR 合同感知对比图收口

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：把当前 protocol_v2 的 S2C Trainable-K1、MOGB-MiniLM-Fair 与已完成 ADB BERT/TextOIR KIR sweep 放入同一张合同感知图，降低“到底在比较哪个方法”的歧义。
- 输入：`results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv`、`results/analysis/adb_kir_sensitivity_v1/adb_summary.csv`；仅读取冻结摘要，不训练、不调参、不读取测试标签选择参数、不修改历史 artifact。
- 新增：`tools/analysis/build_cross_kir_contract_atlas_v1.py`；输出 `results/analysis/cross_kir_contract_atlas_v1/rows.csv`、`CROSS_KIR_CONTRACT_ATLAS_MANIFEST.json`、`CROSS_KIR_CONTRACT_ATLAS_REPORT.md` 和 `figures/cross_kir_contract_atlas_v1/cross_kir_contract_atlas.png`。
- 结果：27 个派生组合全部有效；Trainable-K1 的 OOS F1 在 9 个 dataset×KIR 组中高于 ADB 的 8 个，但 CLINC150 的 F1-All/ Known Recall 在全部 KIR 都更低；MOGB-MiniLM 的低 false acceptance 与严重 Known rejection 同时出现；ADB 的 OOS F1 随 KIR 增大在三数据集均下降。该图仅用于合同和机制对照，不能把不同 backbone/监督条件合并为 SOTA 排名。
- 数据影响：无 canonical、registry、embedding、历史 E2/E3 或 baseline run 修改。
- 验证：atlas manifest SHA256=`5913c4924225fe3f404dbac9085482b25c1f309c35bc437979e23e30971f5a49`；待收口执行研究状态、开发日志、数据跟踪、编译/lint 和 `git diff --check`。
- 风险与下一步：DCLOOS 仍是额外 pseudo/external-OOS 监督合同，DA-ADB 仍无效；下一步应优先补齐同一 StackOverflow/KIR/seed 下可审计的外部基线或继续做错误归因图，不得据此宣称跨合同 SOTA。

## 2026-08-10：ADB 跨 KIR 五 seed 扩展收口

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：把三数据集×三 KIR 的 ADB BERT/TextOIR 外部参照从 3 seeds 扩展到正式 5 seeds，验证 Trainable-K1 与 ADB 的差异不是三 seed 偶然波动。
- 计划/运行：新增 seeds `13`、`123`，共 18 个单元；协议数据根为 `../artifacts/s2c/external/adb_gpu_runtime_data_v4/`，运行根为 `../artifacts/s2c/external/adb_gpu_runtime_v4b/`；旧 v1/v2/v3 运行根只读合并，未覆盖已有结果。
- 结果：18/18 新单元完成，连同既有 27 个单元达到 `45/45`；每个 dataset×KIR 均为 5 seeds，所有 manifest `complete`，指标从逐样本 `y_true.npy/y_pred.npy` 重算，未使用 test OOS 选参。
- 五 seed ADB 摘要：CLINC150 OOS F1=`91.60/89.39/84.76`，Banking77=`82.80/74.97/66.27`，StackOverflow=`93.14/87.21/73.70`（KIR=.25/.50/.75）。Trainable-K1 的 OOS F1 在 9 个工作点中 8 个高于 ADB；CLINC150 的 F1-All/Known Recall 仍低于 ADB，Banking77 KIR=.75 的 F1-All 仍低于 ADB，StackOverflow 在三个 KIR 均保持较高 Known Recall。
- 新增/更新：`results/analysis/adb_kir_sensitivity_v2/`、`figures/adb_kir_sensitivity_v2/`、`tools/analysis/build_adb_kir_sensitivity_v1.py` 的动态 seed 说明；新增合同图 `results/analysis/cross_kir_contract_atlas_v2/` 与 `figures/cross_kir_contract_atlas_v2/`，并扩展 atlas 脚本支持显式 fair/ADB source。
- 研究边界：ADB 仍为 BERT/TextOIR 外部合同，不能进入 MiniLM fair 主排名；五 seed 扩展只加强外部机制参照，不证明跨 backbone SOTA，也不替代 MOGB 官方论文复现或 DCLOOS 额外 OOS 监督审计。
- provenance：v4b 18 个新 manifest aggregate SHA256=`060f6561172f181e090386e07f168992fbae1377dd0e7075a4af4a92b9ba9879`；五 seed ADB summary SHA256=`b948e56cadf0edd7480cf6a5d9d4ad6b12c8dce98fb973ab2c62ab95eb5023ad`。
- 验证：运行状态、开发日志、数据跟踪、脚本编译/lint 和 `git diff --check` 在本批结束后统一复核；当前工作树仍未提交。

## 2026-08-10：ADB 五 seed 配对推断与森林图

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：在不重跑训练的前提下，量化 Trainable-K1 相对 ADB 的五 seed 配对差异、置信区间和错误工作点代价。
- 新增：`tools/analysis/build_adb_paired_inference_v1.py`；输出 `results/analysis/adb_paired_inference_v1/paired_inference.csv`、`ADB_PAIRED_INFERENCE_MANIFEST.json`、`ADB_PAIRED_INFERENCE_REPORT.md` 和 `figures/adb_paired_inference_v1/trainable_minus_adb_paired_forest.png`。
- 统计：36 个 dataset×KIR×metric 单元；每个单元使用相同 registry 的 5 个 seed，固定 RNG seed=`20260810`、10,000 次 paired bootstrap，另报 sign-test 与 Cohen dz；没有使用 test OOS 选参。
- 结果：CLINC150 KIR=.25/.50 的 OOS F1 差值 CI 均为正，但 KIR=.75 转为负；CLINC150 三个 KIR 的 F1-All/ Known Recall 均低于 ADB。Banking77 KIR=.25/.50 的 OOS F1 和 F1-All 差值稳定为正，KIR=.75 F1-All 转负；StackOverflow OOS F1/F1-All 多数为小幅正差，Known Recall 在 KIR=.50/.75 稳定更高。该结果把“Trainable 部分领先、但综合指标和数据集依赖”从均值描述推进为五 seed 配对证据。
- 边界：ADB 仍是 BERT/TextOIR 外部合同；bootstrap 只证明当前 split 下的配对稳定性，不消除 backbone、训练目标和监督合同差异，不能写成同条件 SOTA。
- 验证：脚本运行成功生成 36 行；后续统一执行研究状态、开发日志、数据跟踪、Ruff、编译和 `git diff --check`。

## 2026-08-10：DA-ADB 隔离 CUDA 单格数值稳定性收口

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；本批未执行 `git add`、`git commit` 或 `git push`。
- 目标：区分 DA-ADB 的 NaN/全类预测是运行时阻断，还是在可用 runtime 下仍然产生的算法/合同结果；不重复 E2/E3，不修改历史 external artifacts。
- 运行：StackOverflow/KIR=`0.50`/seed=`42`；使用 protocol_v2 外部适配数据根、Known labels 文件和独立 CUDA runtime `/tmp/s2c_gpu_runtime_20260810`；run root 为 `../artifacts/s2c/external/da_adb_gpu_runtime_v1/`。
- 结果：1/1 单元完成；训练日志全程有限，`run_manifest.status=complete`，返回码为 0；逐样本预测形状一致、无 NaN/Inf，包含全部 11 个标签。
- 指标：OOS F1=`70.82%`、F1-All=`72.03%`、F1-Known=`72.15%`、Accuracy=`70.15%`、Known Recall=`72.93%`、false acceptance=`30.33%`、false rejection=`27.07%`。
- 解释：DA-ADB 从“无有效预测”细化为 `valid_external_cell_pending_replication`。新单格明显低于旧 seed=0 兼容单格 OOS F1=`90.90%`；旧/新差异可能来自 Known list、seed、数据快照、环境或兼容配置，尚未完成归因，因此不挑选高值、不进入 MiniLM fair 主排名。
- 新增：`tools/analysis/build_da_adb_gpu_runtime_report_v1.py`、`docs/analysis/DA_ADB_GPU_RUNTIME_SINGLE_CELL_V1.md`、`results/analysis/da_adb_gpu_runtime_v1/`；ledger 追加 `da_adb_gpu_runtime_single_cell_v1`。
- 风险与下一步：只做旧/新 DA-ADB 合同差异审计或在明确同合同下重复一个控制单格；不直接扩展 DA-ADB 多 seed，不据此宣称超过 ADB/MOGB/DCLOOS 或 SOTA。
- 验证：独立 y_true/y_pred 重算与报告生成通过；随后运行研究状态、开发日志、数据跟踪、Ruff、编译和 `git diff --check`。

## 2026-08-10：DA-ADB 当前协议三 seed 配对收口

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；本批未执行 `git add`、`git commit` 或 `git push`。
- 目标：在当前 protocol_v2 StackOverflow/KIR=.50 合同下补齐 DA-ADB seed=`87,100`，与已有 seed=`42` 组成三 seed 外部参照，并与同 seed S2C Trainable K=1 做配对描述；不把 BERT/TextOIR 与 MiniLM 结果混成公平排名。
- 运行：`../artifacts/s2c/external/da_adb_gpu_runtime_v1/`，独立 `/tmp/s2c_gpu_runtime_20260810`；新增 seed87/100 run，三个 manifest 均 complete，测试预测均有限、形状一致、11 类可达。
- 结果：DA-ADB OOS F1=`72.48±6.24%`、F1-All=`74.02±3.13%`、Known Recall=`75.97±4.10%`、FA=`29.07±11.19%`；同 seed S2C Trainable K=1 为 OOS F1=`88.21±1.72%`、F1-All=`86.69±1.45%`。
- 配对差值：S2C−DA-ADB 的 OOS F1=`+15.73pp`、F1-All=`+12.67pp`、Known Recall=`+7.71pp`、FA=`−20.89pp`；固定 RNG=`20260810`、20,000 次 bootstrap，仅作合同感知描述。
- 新增：`tools/analysis/build_da_adb_current_protocol_summary_v1.py`、`docs/analysis/DA_ADB_CURRENT_PROTOCOL_SUMMARY_V1.md`、`results/analysis/da_adb_current_protocol_summary_v1/`、`figures/da_adb_current_protocol_summary_v1/`；manifest SHA256=`8ae6c13e37439296a240adeee6c47091a315c3977d7411a95d24bf01856c1dc3`。
- 风险：DA-ADB 仍为 BERT/TextOIR 外部合同；旧 seed0 的 `90.90%` 与当前三 seed 不能混合解释，不能据此声称跨合同 SOTA。
- 验证：脚本编译、图像人工检查、Ruff、`pytest tests/unit`（350 passed）、研究状态、开发日志、数据跟踪和 `git diff --check` 均通过；未修改 canonical、registry、E2/E3、MOGB 或历史 external artifact。

## 2026-08-10：DCLOOS 固定 Registry 单格中断收口

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；本批未执行 `git add`、`git commit` 或 `git push`。
- 目标：在 StackOverflow/KIR=`0.50`/seed=`42` 的当前 protocol_v2 train/dev/test 上运行 DCLOOS，并固定读取同一 registry 的 Known labels；保留 DCLOOS 的 BERT、pseudo-OOS 和外部 SQuAD 监督，不将其伪装成 Known-only fair baseline。
- 新增：`run_dcloos_official.py` 的 `--known-labels-file` overlay 适配；该适配只在运行时 overlay 中读取 JSON Known 列表，第三方 checkout 不变。新增 `tools/analysis/build_dcloos_current_protocol_summary_v1.py`，待最终 metrics 落盘后生成合同感知表格和两张图。
- 运行根：`../artifacts/s2c/external/dcloos_stackoverflow_kir050_seed42_fixed_registry_v1/`；外部 SQuAD 快照 SHA256=`f6bf61866c86d3b11565826c3ca1faa00e31f196e0ad9bfd000ec45575fd426e`。
- 状态：运行约 3530 秒后进程结束，没有生成最终 `metrics.json`、`stdout.log` 或 `stderr.log`；只留下中间 validation-best `predictions.npz`，已明确排除出结果汇总。历史 reduced、timeout 和 smoke artifact 保持只读。
- 新增阻塞报告：`docs/analysis/DCLOOS_CURRENT_PROTOCOL_BLOCKER_V1.md`；artifact manifest 标记 `interrupted_no_final_metrics`，不把这次运行写成性能结果。
- 验证：overlay 编译通过；新脚本 `ruff check` 与 `py_compile` 通过；本次中断后需重新运行研究状态、开发日志、数据跟踪和 `git diff --check`。

## 2026-08-10：DCLOOS 固定 Registry reduced 资源诊断收口

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；本批未执行 `git add`、`git commit` 或 `git push`。
- 目标：在不扩大 DCLOOS 矩阵的前提下，用 `max_epochs=10`、`patient=3` 判断当前固定 Known-list 单格是否可以通过缩短预算完成；仍保留 BERT、pseudo-OOS 和外部 SQuAD 合同。
- 运行根：`../artifacts/s2c/external/dcloos_stackoverflow_kir050_seed42_fixed_registry_reduced10_v1/`；固定 registry Known list、外部 SQuAD 和运行时 overlay 均保留。
- 结果：运行约 1,246 秒后仍未生成最终 `metrics.json`，实际走 CPU 路径；人工停止并写入 `run_manifest.json`，状态为 `timeout_incomplete`。中间 `predictions.npz` 不进入指标、图表或任何性能结论。
- 实验数量：计划 1，完成 0，失败/中断 1；没有修改 canonical、registry、E2/E3、MOGB 或历史 external artifact。
- 风险：DCLOOS 当前 registry 的最终可比单格仍未闭合；不得用 reduced KIR=.75/seed=888 或中间 prediction 替代。
- 下一步：继续使用已完成的 S2C/MOGB/ADB/DA-ADB 证据和合同感知可视化；若重新运行 DCLOOS，必须先解决独立 CUDA runtime/预算问题并重新登记新 experiment_id。

## 2026-08-10：跨 KIR 匹配 Known 覆盖率前沿分析

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：检验 S2C-Trainable-K1 相对 MOGB-Fair 的 OOS F1 优势是否在相同 Known Recall 工作点下仍存在，避免把默认阈值差异误解为排序能力差异。
- 输入：冻结 `results/analysis/s2c_mogb_operating_curve_attribution_v1/matched_known_recall.csv`，源 SHA256=`bbccc3389ede2bbe55c7d4059283981127b4ab5d345ee14f6157cca0d8804fe3`；不训练、不重新编码、不调参。
- 规模：3 数据集×3 KIR×5 seed×3 个目标 Known Recall，共 270 个方法行、135 个配对单元；全部有效，0 failed/missing/duplicate/invalid。
- 结果：目标覆盖率 80%、90%、95% 下，S2C 在全部 27 个 dataset×KIR×target 组合中 OOS F1 胜过 MOGB-Fair，且每组 5/5 seed 配对胜出；报告和图表位于 `docs/analysis/CROSS_KIR_MATCHED_FRONTIER_V1.md`、`results/analysis/cross_kir_matched_frontier_v1/`、`figures/cross_kir_matched_frontier_v1/`。
- 解释边界：这是 test-score 的事后工作点诊断，不是正式阈值选择，也不构造跨 backbone SOTA；源表没有 F1-All，未伪造该指标。
- 验证：脚本 `py_compile`、Ruff、源完整性检查和图像人工检查通过；没有修改 E2/E3/MOGB/历史 artifact。

## 2026-08-10：跨 KIR 错误预算归因

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：在已有 Trainable-K1/MOGB-Fair 五 seed 逐样本转移审计上，量化当前方法的 F1-All 优势来自 Known 覆盖恢复还是 OOS 正确拒识；不重复训练、不调阈值、不修改协议。
- 输入：`results/analysis/trainable_mogb_open_intent_transitions_v1/cell_decomposition_summary.csv`，SHA256=`19eb0d0fa50c07f3106ee3f6afd1b72fb9d86c1a1d23b8a820bd4ed04a1526bf`；9 个 dataset×KIR 单元，每单元 5 seed。
- 新增：`tools/analysis/build_cross_kir_transition_attribution_v1.py`、`docs/analysis/CROSS_KIR_TRANSITION_ATTRIBUTION_V1.md`、`results/analysis/cross_kir_transition_attribution_v1/`、`figures/cross_kir_transition_attribution_v1/`。
- 结果：跨三个 KIR，Trainable 相对 MOGB-Fair 的 F1-All 平均提升为 CLINC150 `35.14pp`、Banking77 `32.48pp`、StackOverflow `43.67pp`；主要来自恢复 MOGB 拒绝的 Known（平均恢复比例 `41.41%/46.52%/56.14%`），OOS 正确净增分别为 `-2.85/-14.11/-6.53pp`。
- 解释：当前 Trainable 不是靠额外接受未知样本取得优势，而是减少 MOGB 平均半径造成的 Known false rejection；该结论只适用于 MiniLM Known-only MOGB-Fair 组件合同，不是完整 MOGB 或 SOTA 结论。
- 验证：脚本运行成功、CSV/manifest 完整、中文图像人工检查通过；随后运行研究状态、开发日志、数据跟踪、编译、Ruff 和 `git diff --check`。

## 2026-08-10：跨 KIR 多指标 Pareto 分析

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：将当前 7 个 fair Gate 方法在 3 数据集×3 KIR×5 seed 的 OOS F1、F1-All、Known Recall 和 false acceptance 放入同一多指标前沿，检查 Trainable-K1 是否只赢单项 OOS F1。
- 输入：`results/analysis/cross_protocol_tradeoff_v1/per_seed.csv`，SHA256=`31ccdd433b38b86ec92b0cd81b460114a7fe1e7f12b94d4a5d031aab7d703145`；315 行、无重复且全部有限。
- 新增：`tools/analysis/build_cross_kir_pareto_frontier_v1.py`、`docs/analysis/CROSS_KIR_PARETO_FRONTIER_V1.md`、`results/analysis/cross_kir_pareto_frontier_v1/`、`figures/cross_kir_pareto_frontier_v1/`。
- 结果：Trainable-K1 在全部 9 个 dataset×KIR 单元均处于四指标均值 Pareto 前沿；没有单一 fair 方法同时支配所有目标，MOGB 的低误接收伴随低 Known 覆盖，Trainable 位于更高 F1-All/OOS F1 的平衡区域。
- 解释边界：Pareto 标记是描述性均值关系，不是显著性检验，也不代表跨骨干/监督合同 SOTA；外部 MOGB、ADB、DA-ADB、DCLOOS 仍保持合同隔离。
- 验证：315 行完整性、manifest、脚本运行、中文图像人工检查通过；后续运行研究状态、开发日志、数据跟踪、编译、Ruff 和 `git diff --check`。

## 2026-08-10：DCLOOS 固定 Registry GPU 预算诊断收口

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 `git add`、`git commit` 或 `git push`。
- 目标：在不扩大 DCLOOS 矩阵的前提下，验证固定 protocol_v2 Known 列表的 DCLOOS 单格是否能在真实 CUDA 和有限预算内产生最终指标。
- 运行：`dcloos_stackoverflow_kir050_seed42_fixed_registry_gpu_v1`，StackOverflow/KIR=.50/seed=42，BERT、pseudo-OOS、外部 SQuAD、`max_epochs=10`、`patient=3`、GPU 0，预先声明 30 分钟上限。
- 结果：RTX 5070 实际运行，峰值显存约 11.8GB；达到运行上限后人工停止，manifest=`timeout_incomplete`，没有最终 `metrics.json`。中间 `predictions.npz` 被明确排除，不能计算或报告性能。
- provenance：manifest SHA256=`7350346fa602eb8bbb1e7d36caff4b1bfe0ed262422821913b8619f81e725d9f`；中间预测 SHA256=`fa4121110028cf1604b07f3a02ac47a429d9c15ff5eb7d4d923e27a5f40c88a9`；外部 SQuAD SHA256=`f6bf61866c86d3b11565826c3ca1faa00e31f196e0ad9bfd000ec45575fd426e`。
- 风险与下一步：DCLOOS 当前 registry 的最终可比单格仍未闭合；不重复同一超时命令、不启动 DCLOOS 多 seed。若继续，必须先设计可恢复/流式预算方案并重新登记实验；当前研究结论继续依赖已完成的 S2C fair、MOGB 组件、ADB 和 DA-ADB 合同分层证据。

## 2026-08-10：实验进展与合同分层短快照

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 Git 提交、推送或清理。
- 目标：把已完成的大量对比实验、当前自有方法、MOGB/ADB/DA-ADB/DCLOOS 合同边界和可视化入口压缩为一个中文短入口，避免后续继续混淆 Gate-only、Cascade、BERT 和外部 OOS 监督。
- 新增：`docs/analysis/EXPERIMENT_PROGRESS_SNAPSHOT_V2.md`；同步 `EXPERIMENT_ANALYSIS_MASTER_V1.md`、`CURRENT_STATUS.md` 和 `EXPERIMENT_LEDGER.csv`。
- 结果：快照确认 `S2C-Trainable-K1` 是当前同协议 Known-only MiniLM Gate 的最强平衡候选；E0--E3、MiniLM、MOGB-Fair、ADB、DA-ADB 和 DCLOOS 的状态与路径已集中列出；没有新训练、重评分、参数选择或历史 artifact 修改。
- 风险与下一步：当前仍不能声称跨合同 SOTA；下一阶段继续以同协议 fair matrix 的错误预算、KIR/seed 稳定性和可视化为主，并只在独立可验证 runtime 下补外部基线单格。

## 2026-08-10：Trainable-K1 与 ADB 跨数据集/KIR 对比分析

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 Git 提交、推送或清理。
- 目标：利用已经完成的 ADB 45 单元和 Trainable K=1 45 单元，在相同 dataset/KIR/seed 下拆解 OOS F1、F1-All、Known Recall、false acceptance 和 false rejection 的差异；保留 BERT/TextOIR 与 MiniLM 合同边界。
- 新增：`tools/analysis/build_trainable_vs_adb_kir_v1.py`、`docs/analysis/ADB_TRAINABLE_KIR_ANALYSIS_V1.md`、`results/analysis/trainable_vs_adb_kir_v1/`、`figures/trainable_vs_adb_kir_v1/`；同步可视化索引、综合分析入口、CURRENT_STATUS 和 ledger。
- 结果：45 个配对行、54 个指标效应；跨 9 个 dataset×KIR 组的 Trainable−ADB 均值为 OOS F1 `+3.11pp`、F1-All `+1.27pp`、Known Recall `−7.12pp`、false acceptance `−9.68pp`、false rejection `+7.12pp`。该结果说明数据集/KIR 工作点差异明显，不能只凭单一 OOS F1 宣称全面超过 ADB。
- 统计：固定 bootstrap seed=`20260810`，每个 cell 10000 次重采样；不使用 test OOS 选择 checkpoint、阈值或参数。
- 风险与下一步：ADB 仍是 BERT/TextOIR 外部合同，不能并入 MiniLM fair SOTA 表；继续用错误预算和可视化解释差异，外部 DCLOOS 仍需可恢复的收敛单格。

## 2026-08-10：Trainable-K1 与 ADB StackOverflow 逐样本错误预算

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；未执行 Git 提交、推送或清理。
- 目标：在 StackOverflow/KIR=.50/seed={42,87,100} 上核验 ADB 与 Trainable-K1 是否使用同一测试样本顺序，并把 OOS 误接收、Known 误拒绝和 Known 错意图拆成可审计的错误预算。
- 新增：`tools/analysis/build_trainable_vs_adb_error_budget_v1.py`、`docs/analysis/TRAINABLE_VS_ADB_ERROR_BUDGET_V1.md`、`results/analysis/trainable_vs_adb_error_budget_v1/`、`figures/trainable_vs_adb_error_budget_v1/`；同步 `CURRENT_STATUS.md`、`EXPERIMENT_ANALYSIS_MASTER_V1.md`、`VISUAL_ANALYSIS_INDEX_V1.md` 和实验台账。
- 数据影响：只读取 protocol view、Trainable predictions 和每 seed ADB 独立测试快照；不训练、不调阈值、不修改 registry/canonical；原始文本和逐样本预测不写入轻量结果。
- 规模：3 seed×6000 条，共 18,000 条逐样本对齐；ADB 测试快照 SHA256 与 run manifest 一致，文本顺序与 protocol view 一致，Trainable sample_id 顺序一致；0 failed/missing/misaligned。
- 结果：Trainable-K1 的 Known 条件误拒为 `16.32%`，ADB 为 `19.22%`；Trainable 的 OOS 条件误接收为 `8.18%`，ADB 为 `7.52%`。因此 StackOverflow 当前是少 2.90pp Known 误拒、但多 0.66pp OOS 误接收的工作点交换，不是无条件胜出。
- 风险：ADB 是 BERT/TextOIR 外部兼容合同，Trainable 是 Known-only MiniLM 合同；逐样本对齐提升了错误预算可信度，但不能消除 backbone/训练差异，也不能构造跨合同 SOTA 排名。
- 验证：脚本运行成功，18,000 行审计一致；需继续运行 `py_compile`、Ruff、研究状态、数据跟踪、开发日志和 `git diff --check`。

## 2026-08-10：Trainable-K1 与 ADB 跨数据集/KIR 逐样本错误预算

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；本批未执行 `git add`、`git commit` 或 `git push`。
- 目标：把 StackOverflow 三 seed 的错误预算扩展到 CLINC150、Banking77、StackOverflow 的 3 个 KIR 和 5 个 seed，验证 Trainable-K1 相对 ADB 的优势是否由统一机制解释。
- 新增：`tools/analysis/build_trainable_vs_adb_cross_dataset_error_budget_v1.py`、`docs/analysis/TRAINABLE_VS_ADB_CROSS_DATASET_ERROR_BUDGET_V1.md`、`results/analysis/trainable_vs_adb_cross_dataset_error_budget_v1/`、`figures/trainable_vs_adb_cross_dataset_error_budget_v1/`；同步 `CURRENT_STATUS.md`、`EXPERIMENT_ANALYSIS_MASTER_V1.md`、`EXPERIMENT_PROGRESS_SNAPSHOT_V2.md`、`VISUAL_ANALYSIS_INDEX_V1.md` 和 `EXPERIMENT_LEDGER.csv`。
- 数据影响：只读取已完成 Trainable/ADB 预测、protocol view、ADB 测试快照和 manifest；每个单元验证文本顺序、标签映射、Trainable sample-id 顺序以及 ADB/protocol/test SHA256；不训练、不调阈值、不修改 canonical、registry 或 checkpoint；不导出原始文本和逐样本公共结果。
- 规模：45/45 配对单元，实际 protocol test 行数合计 221,700；状态计数仅保留聚合 CSV，所有对齐标记为 true。
- 结果：CLINC150 的 Trainable Known 条件误拒增加约 16--17pp、OOS 条件误接收减少约 10--13pp；Banking77 分别增加约 6--8pp、减少约 16--18pp；StackOverflow 则减少约 1--3pp Known 误拒，但在 KIR=.50/.75 增加约 0.7--1.1pp OOS 误接收。Trainable 的优势是数据集相关的覆盖--拒识重分配，不是跨骨干的无条件胜出。
- provenance：结果 manifest SHA256=`1f24c722689857a23b2148869ef2eefbcf4bd9f8deeae110b3fc917c2d8d8796`；分析脚本 SHA256=`e6af5b1882bea514474fb36bb00add6f91c820f5aa44d4589670d5c9b02b6e90`。
- 风险与下一步：ADB 仍为 BERT/TextOIR 外部合同，不能与 MiniLM fair 行合并成 SOTA 排名；继续使用错误预算、表示和边界可视化解释差异，DCLOOS 未闭合单元仍不进入性能表。
- 验证：分析脚本已运行成功并生成四张图；本次文档与脚本变更后继续执行 `py_compile`、Ruff、研究状态、数据跟踪、开发日志和 `git diff --check`。

## 2026-08-10：Trainable-K1 与 ADB 指标—错误预算机制合并

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；本批未执行 `git add`、`git commit` 或 `git push`。
- 目标：把已经完成的 Trainable−ADB OOS F1/F1-All/Known Recall 差值与逐样本 Known/OOS 状态差值按同一 `dataset×KIR` 合并，明确“当前方法为什么看起来更好”对应的错误预算机制。
- 新增：`tools/analysis/build_trainable_vs_adb_mechanism_summary_v1.py`、`docs/analysis/TRAINABLE_VS_ADB_MECHANISM_SUMMARY_V1.md`、`results/analysis/trainable_vs_adb_mechanism_summary_v1/`、`figures/trainable_vs_adb_mechanism_summary_v1/`；同步 `CURRENT_STATUS.md`、实验综合分析、短快照、可视化索引和实验台账。
- 数据影响：仅读取已完成分析 CSV；不训练、不重评分、不调阈值、不读取或导出原始文本、不修改任何 registry/canonical/checkpoint。
- 结果：9 个 `dataset×KIR` 行全部成功。CLINC150 和 Banking77 的机制均为“保守拒识”（Known 误拒增加、OOS 误接收下降）；StackOverflow 为“覆盖恢复，但中高 KIR 伴随轻微 OOS 误接收权衡”。
- provenance：源 metrics SHA256=`3b0ae26ec6f83140f437ff5f81f763e464429a987e8df8e1ac0182d8455d6d74`；源 error-budget SHA256=`ba83df8365327597a93569270ed4eb83b0beb3cb791681944270a58ec84c0eac`；分析脚本 SHA256=`202a9a842b64e6430fb16cc4560ae17c27982fefc0e51854e251ab42c0576240`；输出 manifest SHA256=`3fb2b01579b5f16be9f3603e687c42e0d0b5b3a25908a3623cc448ad3d131e0d`。
- 风险与下一步：ADB 仍是 BERT/TextOIR 外部合同；这组图用于解释数据集/KIR 异质性，不构成跨骨干 SOTA 结论。后续继续围绕已有 fair matrix 和外部合同层做可视化，不重复 E2/E3。
- 验证：脚本成功生成 9 行机制表和两张图；本批收口后运行 `py_compile`、Ruff、研究状态、开发日志、数据跟踪和 `git diff --check`。

## 2026-08-10：Trainable-K1 与 ADB 的 OOS Precision/Recall 分解

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；本批未执行 `git add`、`git commit` 或 `git push`。
- 目标：从已经通过对齐审计的五状态计数重算 OOS precision、OOS recall、OOS F1 和 Known acceptance，解释 Trainable 相对 ADB 的 OOS F1 差异到底来自哪一部分。
- 新增：`tools/analysis/build_trainable_vs_adb_oos_decomposition_v1.py`、`docs/analysis/TRAINABLE_VS_ADB_OOS_DECOMPOSITION_V1.md`、`results/analysis/trainable_vs_adb_oos_decomposition_v1/`、`figures/trainable_vs_adb_oos_decomposition_v1/`；同步状态入口、实验综合分析、短快照、可视化索引和台账。
- 数据影响：仅读取 45 个 Trainable/ADB 配对单元的状态汇总；OOS precision 分母按“预测为 OOS 的全部样本”（正确拒绝 OOS + 误拒 Known）计算；不训练、不调阈值、不导出原始文本或逐样本公共结果。
- 结果：Banking77 平均 OOS precision `-4.28pp`、recall `+16.74pp`、F1 `+6.61pp`；CLINC150 分别 `-8.79pp`、`+11.49pp`、`+0.90pp`；StackOverflow 分别 `+2.12pp`、`+0.81pp`、`+1.80pp`。CLINC/Banking 的 F1 增益主要来自 recall，StackOverflow 中高 KIR 的增益主要来自 precision。
- provenance：源状态 SHA256=`4ce790f6df804c67c7636c621520ac9345395b1fbd83e0fc115df79ba3d182fc`；分析脚本 SHA256=`d01d5e920638f40a2a41e5286c686b7b4346b772b73f02fcd40485875a3f4f66`；输出 manifest SHA256=`f02c2e968f60bedeb74023094945e68c04c8870cc81dc4e6186e97d8c9a416ac`。
- 风险与下一步：ADB 仍是 BERT/TextOIR 外部合同；指标分解支持机制解释，不支持跨合同 SOTA 排名。继续制作现有 fair matrix、MOGB 和外部基线的可视化，不重复旧 K/KIR 训练矩阵。
- 验证：脚本成功生成 90 个 per-seed 方法行、45 个配对行和两张图；本批收口后运行 `py_compile`、Ruff、研究状态、开发日志、数据跟踪和 `git diff --check`。

## 2026-08-10：Trainable-K1 与 ADB 的意图级错误归因

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；本批未执行 Git 提交、推送或清理。
- 目标：把跨数据集/KIR 的平均错误预算进一步展开到 intent 层，区分“真实 Known intent 被误拒”和“OOS 被哪个预测 Known intent 吸收”，解释当前 Trainable 相对 ADB 的优势/代价是否集中在少数类别。
- 新增：`tools/analysis/build_trainable_vs_adb_intent_error_v1.py`、`docs/analysis/TRAINABLE_VS_ADB_INTENT_ERROR_V1.md`、`results/analysis/trainable_vs_adb_intent_error_v1/`、`figures/trainable_vs_adb_intent_error_v1/`；同步状态入口、实验综合分析、短快照、可视化索引和实验台账。
- 数据影响：只读取已经完成并通过 sample-id、文本、标签和 manifest 对齐审计的 Trainable/ADB 预测；Known 误拒按真实意图聚合，OOS 误接收按预测吸收意图聚合；不训练、不调阈值、不选择 checkpoint、不导出文本或逐样本公共结果。
- 规模：45 个 dataset×KIR×seed 对齐单元；1,850 个 per-seed intent 记录；663 个 intent×dataset×KIR 聚合记录；全部对齐标记为 true。
- 结果：三个数据集的差异均分散在多个意图，而非单一类别造成。StackOverflow 的 `spring`、`osx`、`sharepoint` 是中高 KIR 下 OOS 误接收增加较明显的吸收意图；`bash`、`cocoa`、`apache` 等意图的 Known 误拒下降较明显。CLINC150/Banking77 的整体方向仍以减少 OOS 误接收、增加 Known 误拒为主。
- provenance：源 fair SHA256=`31ccdd433b38b86ec92b0cd81b460114a7fe1e7f12b94d4a5d031aab7d703145`；源 ADB SHA256=`0a287bf154bdb0d2db06ec916cfbb4aea40c8b536fabf419942f684e587e1223`；分析脚本 SHA256=`49670ab893863ff052642a77ae1fb479091282d37df8a03de09983f672a19b20`；输出 manifest 内 SHA256=`beffb22bc4ecbcbbb50f4c5f0720d7baa29f081cc7a9a2353141592544524f9b`。
- 验证：脚本运行成功并通过图像人工检查；后续执行 `py_compile`、Ruff、单元/集成/Smoke、研究状态、数据跟踪、开发日志和 `git diff --check`。

## 2026-08-11：统一 prediction contract 与机制证据入口

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；本批未执行 `git add`、`git commit` 或 `git push`。
- 目标：把当前 protocol_v2 的 Trainable/Frozen MiniLM Gate、native detector、MOGB fair component 和完整 ADB 外部标签结果转换为统一的匿名逐样本合同；将 DCLOOS、DA-ADB 和未完成传统 TextOIR baseline 保留为阻塞/外部监督登记。
- 新增：`tools/analysis/unified_prediction_contract_v1.py`、`tools/analysis/build_unified_comparison_and_mechanism_v1.py`、`tests/unit/test_unified_prediction_contract_v1.py`、`results/analysis/unified_prediction_contract_v1/`、`results/analysis/unified_comparison_v1/`、`figures/unified_comparison_v1/`、`docs/analysis/UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md`。
- 结果：统一合同包含 2,394,360 条匿名逐样本行、486 个运行组；Trainable/Frozen/MOGB/native MiniLM 的 441 个方法-cell 与参考 sample-id/标签序列一致；45 个 ADB 外部 BERT cell 保留为 external contract，未并入 fair 对齐或排名。MOGB ball id/count/size/radius/purity 字段来自原始 ball artifact，不做填补。
- 统计与图：统一汇总包含 86 条方法/数据集/KIR 行、486 个合同运行组、已有 10,000 次配对 bootstrap 统计入口和 17 条图索引；新图仅从现有 CSV 生成，不进行训练、阈值选择或 test OOS 调参。
- DCLOOS 边界：固定注册单元的 `interrupted_no_final_metrics`、官方 timeout 和 reduced pseudo/external-OOS intermediate prediction 均不写入正式统一逐样本结果；具体状态见 `blocked_methods.csv`。
- 风险与下一步：ADB 的 sample-id 顺序可桥接，但标签/score/selection 语义仍属于外部 BERT 合同；不能由本入口推出跨骨干 SOTA。未生成缺少公开 embedding artifact 的新 UMAP，避免用不可审计中间表示伪造几何证据；现有 geometry/error-quadrant/MOGB/统计图通过 figure manifest 收录。
- 验证：统一构建脚本成功运行，合同单元测试 `4 passed`；仍需完成全量 unit/integration/smoke、compileall、Ruff、数据跟踪、开发日志、research-state 和 `git diff --check` 收口。

## 2026-08-11：统一对比 plan 与附件要求差距审计

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；本批未执行 Git 提交、推送或清理。
- 目标：核对用户提供的十阶段 S2C 统一对比 plan 是否覆盖附件中的 Q1–Q4、P0–P5 和主文八图要求，并在不增加新模型的前提下补充可执行的 entry/exit gate。
- 新增：`docs/analysis/PLAN_ALIGNMENT_AUDIT_V1.md`、`docs/analysis/UNIFIED_COMPARISON_SUPPLEMENT_PLAN_V1.md`、`results/analysis/plan_alignment_audit_v1/requirement_matrix.csv`、`manifest.json`；同步 `CURRENT_STATUS.md` 和统一对比报告的 UMAP/阻塞项表述。
- 结果：23 条要求中 1 条已完成、15 条部分完成、7 条未完成；当前统一合同的 `2,394,360` 行、486 个运行组和 441 个 fair/existing aligned method-cell 保持不变。审计确认方向高度一致，但缺少 KNNCL/OpenMax/DOC/DeepUnk final metrics、完整 DA-ADB、DCLOOS supervision frontier、backbone 控制、error-aware UMAP、low-resource 和统一 performance-cost Pareto。
- 决策：补充计划规定先单格 smoke、再矩阵扩展；没有 final metrics、合同对齐或校准选择审计的 cell 不进入统一行级结果，timeout/smoke/intermediate prediction 不得用于排名。
- 验证：CSV/JSON schema 解析、矩阵状态计数、reader QA 和 `git diff --check` 已通过；reader QA 建议的合同层命名映射已补入补充计划。research-state、development-log 和 data-tracking 检查也通过。

## 2026-08-11：baseline 范围与提交候选审计

- Base commit：`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`；本批只做工作树/来源/提交范围审计，不训练模型、不修改 `../artifacts`，尚未提交或推送。
- 校正：StackOverflow、Banking77 和当前 S2C `clinc150`（TextOIR `data/oos`）与独立 TextOIR snapshot 完成字节级来源核对；ADB、DA-ADB、KNNCL 等差异属于 backbone、Known-list、seed、训练和评估合同差异，不是数据内容差异。
- 校正：TextOIR detection inventory 不止 ADB/DA-ADB；MSP、SEG、OpenMax、LOF、DOC、DeepUnk、`(K+1)-way`、MDF、ARPL、KNNCL-last/all、DA-ADB-llama 和 EliDecide 均保留在方法地图，published/legacy reference 与当前 final-metrics rerun 分层记录。
- 提交策略：只提交外部适配/统一 prediction contract/MOGB loss contract 的可复用代码、测试、当前状态记录和聚合证据；不提交模型、checkpoint、embedding、逐样本/逐意图/逐粒球大表、运行日志、重复分析脚本、过渡图和第三方 checkout。
- 风险：未提交的 analysis-only 文件继续保留在本地工作树，当前报告中的历史/本地 artifact 路径不代表公开轻量结果；后续若需发布某一分析包，必须单独通过 manifest、来源 CSV 和 claim audit。
- 验证计划：提交前运行选定单元测试、compileall、Ruff、research-state、development-log、data-tracking 和 `git diff --check`；提交后复核 `git show --stat`、分支和剩余未提交文件。

## 2026-08-11：分析资产目录治理与可恢复归档

- Base commit：`4a8742f9af53d6face95afece48fb55e97445575`；本批未执行 `git add`、`git commit` 或 `git push`。
- 目标：停止继续增加平行分析 Markdown、版本文件夹和一次性入口；把结果、图、报告、manifest、builder 及合同层关系收束到现有 `configs/experiment_registry.yaml` 的 `analysis_bundles`。
- 修改：扩展 `configs/experiment_registry.yaml`，登记 9 个 canonical bundle 和 2 个可恢复 archive 记录；新增 `tools/maintenance/audit_asset_catalog.py` 及 `tests/unit/test_asset_catalog.py`；同步 `README.md`、`AGENTS.md`、`docs/CURRENT_STATUS.md`、`docs/EXPERIMENTS.md`、`docs/REPRODUCIBILITY.md` 的资产治理入口。
- 归档：仅将经引用审计确认未被当前代码、测试或文档直接引用的两个一次性 builder 移至 `tools/analysis/archive/`；没有移动结果、图、数据、原始 artifact 或第三方 checkout。
- 数据与 artifact：不训练、不重评分、不改阈值、不改 `data/`、`../assets/`、`../artifacts/` 或 `third_party/mogb_official`；新增审计器只读本地轻量结果关系。
- 结果：资产审计 `9 bundles + 2 archives` 通过；当前保留 `78` 个未登记结果目录和 `71` 个未登记图目录作为历史 orphan warning，不批量判错或删除。原有实验登记审计也通过。
- 验证：新增相关单元 `3 passed`；全量 unit `358 passed`、integration `13 passed`、smoke `3 passed`；`compileall`、`check_research_state.py`、`check_data_tracking.py` 和 `git diff --check` 通过。新增审计器和测试自身未触发 Ruff 错误，但全仓 Ruff 仍被既有 `scripts/experiments` 与 `src/protocol_v2/experiments` 的 15 条问题阻断；public-results verify 仍因工作树中已有未列入 whitelist 的结果文件失败，未扩大本批范围。
- 风险与下一步：当前仍不能把所有历史 orphan 视作可提交证据；下一步按 bundle/ledger 引用逐项关闭或归档，单独形成闭合 evidence bundle 后再决定 Git 提交边界。
