# s2c 项目知识库

> GPT-6 适配变更说明：去掉每次回复加载 Skill、所有修改追加历史日志和固定全量验证；保留数据准入、协议隔离、产物追溯及梯度约束。按任务读取证据，自主修复普通错误后完成定向验证。

## HERO：反过度防御

=== 范围约束（约束你提议什么修法，不约束你找什么）===
凡是这里真的有问题，都要报——包括听起来罕见但本项目确实会产生的情况。
然后把修法收在范围内：
1. 这不是一篇安全攻防论文。可以校验，禁止过度防御。除非本项目另有说明，默认操作者是
   自己机器上的合作者；如果它真有对手，它会写明，以那个范围为准。
2. 不要加哈希、校验和或指纹，除非它替代了一个实质上更贵的操作，并且结果会改变下一步做什么。
3. 禁止防御性脚手架：不为这里不会发生的情况加 feature flag、迁移框架、兼容层或包装层。
4. 禁止钻牛角尖：冷门编码、符号链接竞态、RTL 文本、毫秒级竞态一律不在范围内，
   除非该情况经由本项目受支持的用法可达——它的文档示例、它公开的接口、它真实的数据。
   可达即可，不需要你复现出来；但“理论上构造得出”不算。
5. 该判断的地方就判断，不要换成评分表、检查清单，或对已经定论的东西再跑一遍校验。
已经见过的形状，供你校准。是例子不是清单——一个真问题不会因为“长得像其中一条”就被驳回：
  H  为了比对两个表格的差异，给每一行都算哈希——直接比单元格就能回答
  H  写下一堆校验和文件，而没有任何代码会去读它们
  E  给一个没有用户、没有部署的应用做账号安全加固
  R  用一整夜对自己的补丁反复审计，而功能一行没写
  R  一个对任何提交都给不通过的审阅者
  O  一层守卫的理由是上一层守卫，而不是需求
另有两种长得像上面、但不是的。这些要报：
  ✓  用摘要比对来跳过重读一个你已经有的大文件
  ✓  本项目自己的文档示例就会产生的那种“听起来罕见”的输入
运行检查前说明它要验证的具体失败，以及失败后会改变的下一步；没有明确作用就跳过该检查。
对的就说对。不要为了交差硬找问题。

## 当前边界

这是 `Gate → Router → Expert` 开放世界意图识别项目。当前活动源码、配置、测试、
文档和轻量公开结果均在 `s2c/`；本地原始产物在 `../artifacts/s2c/`，数据和基础模型
在 `../assets/`，独立 TextOIR 仓库在 `../textoir/`。

执行与 Skill 优先级遵循上级 `../AGENTS.md`。先给结果，保持改动最小；
根据任务读取相关证据，不要求每次回复重新加载通用 Skill。

当前文档入口只有：

- `README.md`
- `docs/METHOD.md`
- `docs/CURRENT_STATUS.md`（唯一研究状态入口）
- `docs/EXPERIMENTS.md`
- `docs/REPRODUCIBILITY.md`
- `docs/EXPERIMENT_LEDGER.csv`（追加式实验总账）

历史开发日志、决策日志和 claim audit 位于 `docs/archive/`，不作为当前事实入口。

`docs/archive/` 仅保存历史资料，不是当前事实来源。不要重新建立平行文档索引或版本号
文档树。

## 用户指定的后续实验默认（2026-08-26）

除非用户另行指定，新的实验默认使用与 `fulltex.tex` 历史数据家族对应的
`historical_v19_paper_main` 协议；当前可执行的具体快照是已核验的 H1 controlled v19，
不能把它写成严格 H0 主 Cascade：

- 数据根为 `../assets/datasets/s2c/prepared/data/multidataset/v19`；主表三项使用
  `clinc150`、`stackoverflow`、`banking77_oos`，后者必须按实际名称报告，不能写成标准
  `banking77`。
- 论文主表对齐使用 `seed=42`、`KIR={0.25,0.50,0.75}`、Gate 的 `K_y=2`、对角
  Mahalanobis、`mean+lambda*std`（CLINC `lambda=0.5`，其余 `lambda=1`）以及历史
  Router/Expert/Cascade 设置。
- Frozen/Trainable 机制对照可以使用同一旧数据根，但必须单独标为 Gate-only 控制，不能
  把它填回论文完整 Cascade 表。
- 论文严格 H0 只有在 `data/v19`、历史 Gate detector、匹配的 Router/Expert 和必要的
  语义输入全部恢复后才允许运行；缺任何一项都必须停止并标记为缺输入，不得自动回退到 H1。
- `protocol_v2_textoir_v1` 的既有结果只作为冻结参考；没有用户明确要求，不再启动新的
  `protocol_v2` 实验，也不把两套结果混排。

## 源码位置

```text
src/protocol_v2/      当前活动协议（data/evaluation/experiments/gate/runtime/tracking）
src/legacy/           v19/历史兼容实现，不作为当前协议默认依赖
src/legacy/gate/      历史 Gate；活动多球 detector 位于 protocol_v2/gate
src/legacy/router/    历史 Router
src/legacy/models/    历史 Expert、SVDD 和 Transformer 封装
src/legacy/pipeline/  历史完整级联推理
src/legacy/runtime/   历史 WorkspacePaths 和 artifact 约定
tools/                训练、评价、分析、兼容和维护入口
tools/legacy/analysis_v19/
                      v19 历史分析与复现脚本，不是活动分析入口
docs/analysis/        当前注册 bundle 报告和少量读者入口
docs/archive/analysis/
                      已完成或过渡性的分析报告归档，不作为当前入口
configs/              运行配置、实验登记和公开导出白名单
tests/                单元、协议和回归测试
results/              GitHub 可提交的轻量 CSV/JSON 快照
results/analysis/archive/
                      已被明确替代的轻量分析结果；保留可审计历史但不作为当前入口
results/analysis/archive/analysis/
                      未登记或已收口的分析结果归档
figures/archive/      与归档结果对应的历史图，不作为当前主图入口
figures/archive/analysis/
                      未登记或已收口的分析图归档
```

完整放置规则见 `docs/METHOD.md`。禁止重新创建 `src/s2c/`，也禁止从 `src`
根目录导入；活动包使用 `protocol_v2.*`，历史包使用 `legacy.*`。

`tools/maintenance/export_public_results.py` 只按
`configs/public_results.yaml` 导出公开文件；不得复制 `../artifacts` 整个目录。

## 维护规则

- 改变实验协议、数据或研究结论时，在已有研究记录中记录 base commit、相关文件、
  数据/artifact 影响及验证证据。普通代码、提示词和文档维护由 Git diff 与最终验证报告留痕，
  不要求向历史开发日志重复追加。
- 不运行训练来完成工作区整理，不修改或重命名 `../artifacts` 原始实验目录。
- 不把 Gate-only 的 Frozen/CE/SupCon 结果写成完整 Pipeline 结果。
- 不提交模型、checkpoint、embedding、Parquet、逐样本 scores 或运行日志。
- 新实验入口使用功能命名；历史 `_v19/_v20/_v21` 文件保留为兼容入口，不再扩展同类版本号。
- 分析资产统一登记在 `configs/experiment_registry.yaml` 的 `analysis_bundles`；结果、图、报告、
  manifest 和 builder 必须从 bundle 关系可追溯。新资产只使用 lower_snake_case，不再用新增的
  `V2/V3` 平行文件夹表达迭代；历史文件通过 `aliases`/`supersedes` 保留。
- `tools/maintenance/audit_asset_catalog.py` 是资产关系审计入口。它只读检查并报告 orphan 目录，
  不自动删除或移动结果；归档必须是显式、可恢复、逐项登记的操作。
- 分析代码按职责放置：`scripts/experiments/` 负责运行，`tools/analysis/` 负责已有结果的后处理，
  `tools/maintenance/` 负责审计；不要为同一张表重复创建新的入口脚本。
- 实验、指标、数据协议或论文论断任务先读 `docs/CURRENT_STATUS.md`，按涉及的实验查询
  `docs/EXPERIMENT_LEDGER.csv`；历史决策有疑义时再查 `docs/archive/protocol_and_data/DECISION_LOG.md`。
  新实验或研究状态变更更新相应台账/状态并运行 `python tools/maintenance/check_research_state.py`；
  只读状态查询不启动实验、不追加台账、不要求阶段 closeout。
- 新计划若与 ledger 中 `do_not_repeat` 且已完成的 protocol/dataset/KIR/seed/representation/K/distance/
  partition/boundary 完全相同，必须拒绝为 `duplicate_completed_experiment`；只有带明确 rerun reason
  的显式覆盖才允许继续。
- 冻结的 `protocol_v2` 运行只从 `protocol_v2.*` 导入；历史 v19 运行使用 `legacy.*`，
  历史 Router 只从 `legacy.router` 导入，禁止在 `legacy.models` 重新导出 Router。
- 训练循环不要添加会破坏 LoRA 梯度的 `torch.no_grad()`。
- 不在初始化阶段调用 `torch.cuda.is_available()`；部分环境会触发原生运行时问题。
- `configs/data/protocol_v2_admission.json` 是唯一数据准入开关。只有同时满足 dataset_version、
  dataset-level admission 和 materialized view/export 的任务才可运行；不得绕过 Gate runner 或 E4
  adapter 向任何 `../artifacts/s2c/runs/<dataset_version>/` 写入。已有
  `protocol_v2_textoir_v1` 作为冻结参考；StackOverflow 为 `admitted_benchmark_local_only`，允许本地实验但
  禁止完整语料进入 Git、论文附件或任何 s2c 再分发包。`protocol_v2_official_v1` 冻结审计，
  legacy `protocol_v2` 仍被拒绝。新的历史 v19 实验按上面的用户指定默认走
  `multidataset/v19` 数据根和历史 runner，不改写已冻结的 protocol_v2 产物。

## 最小验证

从 `s2c/` 按改动选择，不默认全部执行：

- 提示词/文档：检查 Git diff、文件引用和规则一致性。
- 代码行为：运行受影响的 `tests/` 测试；Gate/指标契约改变须有相应回归测试。
- 实验注册关系：`python tools/analysis/audit_experiment_registry.py`。
- 公开导出逻辑或白名单：`python tools/maintenance/export_public_results.py --verify`，
  修改导出脚本时另做语法/相关测试检查。
- 研究状态或台账：`python tools/maintenance/check_research_state.py`。

失败先修复并重跑相关检查；通过后报告实际命令与结果。缺少数据时说明无法验证的项，
不得伪造通过、修改冻结结果或改换实验协议来消除失败。

任何公开结果数字都必须能通过 `results/MANIFEST.csv` 或对应 artifact manifest 追溯。
