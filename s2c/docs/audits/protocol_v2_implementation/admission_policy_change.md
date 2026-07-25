# StackOverflow 本地基准准入变更

## 决策

从 `protocol_v2_textoir_v1` 起，StackOverflow 固定 TEXTOIR 快照采用
`admitted_benchmark_local_only`。该状态允许在本地进行 canonical 构建、embedding、
Gate、Pipeline 及外部方法复现；它不授权 s2c 重新分发完整语料。

## 原因

此前规则把“公开重新发布完整 StackOverflow 语料”的逐行溯源与许可证要求，错误地
作为本地科研 benchmark 的前置条件。对固定的 20,000 title、20 label 快照而言，
这一要求超出了复现实验所需的最小证据，并阻断了与 TEXTOIR 的公平协议比较。

## 约束

- 入口数据是固定 TEXTOIR commit `dffe2b1b848a069a6808f8089b4cb9bd16e2062b` 的
  `textoir/data/stackoverflow`，复制后运行时不读取 TEXTOIR 工作树。
- canonical 不 lower-case、去重、重分 split 或改写标签，且必须保留 20,000 条和 20 个标签。
- `redistribution_by_s2c=false`、`public_git_tracking=false`：完整文本不得进入 Git、论文附件或公开结果目录。
- manifest 仍保留 `jacoxu/StackOverflow` 上游参考、local-only 标识和逐行归因不完整的事实。

## 版本隔离

`protocol_v2_official_v1` 保留为 frozen audit version；旧 `protocol_v2` 仍是
rejected legacy candidate。三个版本的 registry、views、exports、run manifests 和结果
不得混合。

## 验证

本版本的 source copy、canonical、165 个 registry、views/exports 和运行时独立性检查
均由同一 audit 目录的 CSV/JSON 报告追溯。该结论只涉及本地实验准入，不构成对原始
Stack Overflow 内容逐条许可或官方分类数据集身份的声明。
