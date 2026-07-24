# s2c 项目结构

`s2c` 的活动实现采用两个明确边界：`src/s2c/` 是 protocol_v2 的可测试核心，旧的
`src/gate/`、`src/router/`、`src/pipeline/` 保留为 v19 兼容实现。新协议不会静默改写
旧路径或旧 artifacts。

```text
s2c/
├── data/       # 本地语料、canonical、registry、views 和 exporter；语料不入 Git
├── configs/    # 声明式数据、方法和实验矩阵
├── src/s2c/    # protocol_v2 runtime、data、evaluation、experiments、tracking
├── scripts/    # 仅参数解析的稳定命令入口
├── tools/      # 审计与维护，不实现论文方法
├── tests/      # unit、integration、smoke 和 fixtures
├── docs/       # 当前契约；archive/ 只保存历史说明
├── results/    # 可公开的轻量汇总，和 raw artifact 分离
└── paper/      # 论文源文件（若存在）
```

## 数据和结果的边界

- `data/sources/official/` 是通过三方核验后从固定官方 commit 字节复制的 raw snapshot。
- `data/sources/textoir/` 只保留被阻断 candidate 的审计快照，不能作为新的 raw source。
- `data/canonical/`、`views/`、`exports/` 是可再建的本地产物，全部被 Git 忽略。
- `data/manifests/` 与 `registries/` 是轻量可追溯记录，可提交 Git。
- `../artifacts/s2c/` 保存 embedding、逐样本预测、checkpoint、运行日志和完整结果。
- `results/` 只保存白名单轻量结果，不能替代完整 raw evidence。

## 当前活动入口

| 任务 | 入口 |
| --- | --- |
| 官方数据导入与 canonical | `python -m s2c.data.official_import`、`python -m s2c.data.build_canonical` |
| registry、views、exports | `python -m s2c.data.build_registries`、`build_views`、`export_protocol` |
| Gate 网格 | `python -m s2c.experiments.plan`、`runner`、`summarize`、`verify` |
| 数据裁决 | `docs/audits/data_provenance/` |
| 历史实现 | `src/gate/`、`src/router/`、`src/pipeline/`，仅在明确 legacy 情境使用 |

不要通过路径猜测数据协议：任何新实验都必须从当前 dataset_version 的 manifest、registry
和 run manifest 读取 provenance。

## Lint 边界

`ruff` 默认检查 protocol_v2 核心、稳定脚本、审计工具和新测试。它显式排除 v19 历史脚本、
历史测试与历史分析目录；这些文件保留是为了复现既有 artifact，不能在 protocol_v2 重整中
被顺手改写。新代码不得添加到这些 legacy 路径。
