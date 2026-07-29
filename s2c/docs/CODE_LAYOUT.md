# s2c 代码布局规范

这是当前唯一的源码分层规则。目录名称与 Python import namespace 一一对应；不要再
创建 `src/s2c/`、`src/gate/` 或在 `src/` 根目录放业务代码。

```text
s2c/
├── src/
│   ├── protocol_v2/       # 当前活动协议；import protocol_v2.*
│   │   ├── data/           # canonical、registry、views、exports、schema
│   │   ├── evaluation/     # 统一指标和评价契约
│   │   ├── experiments/    # 当前协议的计划、runner、summary、诊断
│   │   ├── gate/           # 当前协议 Gate 和边界契约
│   │   ├── runtime/        # ProtocolV2Paths 等活动路径
│   │   └── tracking/       # manifest、provenance、原子写入
│   └── legacy/             # v19/历史兼容代码；import legacy.*
│       ├── gate/           # 历史 Gate；多球文件为兼容转发层
│       ├── gate_minimal/   # 严格 SVDD 历史基线
│       ├── models/         # 历史 Router/Expert/模型组件
│       ├── pipeline/       # 历史完整级联
│       ├── router/         # 历史 Router
│       ├── runtime/        # 历史 WorkspacePaths/artifact 兼容层
│       └── utils/           # 历史工具
├── tools/                  # 审计、训练、分析和外部适配 CLI
├── scripts/                # 稳定的薄命令入口，不放核心算法
├── configs/                # 数据准入、实验矩阵、registry、公开结果白名单
├── tests/                  # unit、integration、smoke 和历史回归
├── docs/                   # 当前契约；docs/archive/ 仅为历史材料
├── results/                # Git 可提交的轻量公开汇总
└── data/                   # 本地数据/视图；忽略，不提交
```

## import 规则

1. 活动代码只能使用 `protocol_v2.*` 导入当前协议模块。
2. 历史代码只能使用 `legacy.*` 导入历史模块。
3. 禁止 `from src ...`、`import src ...`、`from s2c ...` 和通过路径猜包。
4. `protocol_v2` 可以通过明确的兼容适配调用 `legacy`；反方向不得发生。
5. `src/` 根目录只保留两个包目录，不放 `__init__.py`、脚本或临时文件。
6. artifact 路径保持原实验血缘，不因源码布局调整而移动或重命名。

## 放置判断

新增文件先回答三个问题：

| 问题 | 放置位置 |
| --- | --- |
| 是否被当前 `protocol_v2_textoir_v1` 运行时直接调用？ | `src/protocol_v2/` |
| 是否只为 v19 或历史 artifact 复现保留？ | `src/legacy/` |
| 是否是命令、审计、导出或训练编排？ | `tools/` 或 `scripts/` |

如果一个文件同时满足两类职责，拆出活动实现与兼容 wrapper，不复制两份算法。
