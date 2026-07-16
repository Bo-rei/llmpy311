# s2c

s2c 是一个面向开放世界意图识别的分层系统：先由 **Gate** 判断输入是否属于已知意图，
再由 **Router** 路由到领域，最后由 **Expert** 完成细粒度意图分类。

当前论文修订不重写模型主线，研究重点是：

1. MiniLM 语义空间中的一个已知意图是否由多个局部语义簇组成；
2. 每意图子中心数 `K_gate` 如何影响 Known/OOS 可分性；
3. 多簇收益在不同数据集、KIR 和 near-OOS 难度下是否稳定；
4. 当前 MultiSphere Gate 与统一 MiniLM 表征上的受控 OOS Baseline 如何比较。

## 工作区边界

代码仓库与大体积数据、模型、实验产物分离：

```text
<workspace>/
├── s2c/                 # 本仓库：源码、配置、测试和研究工具
├── assets/
│   ├── datasets/s2c/    # source 与 prepared 数据
│   └── models/          # MiniLM、SmolLM 等本地模型
└── artifacts/s2c/       # checkpoint、实验输出和论文表图
```

运行时代码通过 `src.runtime.WorkspacePaths` 解析这些位置。不要在仓库内重新创建
`data/` 或 `outputs/`，也不要把机器相关的绝对路径写进代码。

## 代码导航

```text
s2c/
├── src/
│   ├── gate/            # OOS Gate 与多球检测器
│   ├── router/          # 领域路由
│   ├── models/          # Expert 与共享模型组件
│   ├── pipeline/        # Gate -> Router -> Expert 推理链
│   └── runtime/         # 路径和 artifact 契约
├── tools/
│   ├── experiments/     # 可复现实验 runner
│   ├── analysis/        # 历史分析与论文导出工具
│   ├── compat/textoir/  # TextOIR 外部协议兼容层
│   └── eval/            # 系统评估
├── scripts/data/active/ # 数据重建入口
├── configs/             # 当前配置与 artifact 注册表
├── tests/               # 单元与历史协议回归测试
└── docs/                # 唯一详细文档体系
```

`src/` 放可复用的系统实现，`tools/` 放有明确输入输出的研究工作流。实验逻辑不应反向
侵入核心 Pipeline；外部项目只能通过兼容层交换 split、prediction 和 manifest。

## 常用命令

在工作区根目录激活环境后进入本仓库：

```bash
conda activate bo
cd <workspace>/s2c
```

检查数据构建命令而不执行：

```bash
python -m src.cli prepare --dataset clinc150 --kir 50 --seed 42 --dry-run
```

运行单元测试：

```bash
pytest tests/unit -q
```

多簇/OOS 实验的推荐入口和参数以
[工具入口索引](docs/10-工具入口索引.md) 为准；完整项目说明从
[文档总入口](docs/README.md) 开始阅读。

统一入口示例：

```bash
python -m tools.experiments.cluster_separability --help
python -m tools.experiments.cluster_separability grid --phase fixed --grid --resume
python -m tools.experiments.cluster_separability baseline --grid --resume
python -m tools.experiments.cluster_separability analyze --study all --skip-nonlinear
```

## 研究产物约束

- 数据、K、边界和阈值选择只允许读取 train/validation，test 只做最终评估；
- OOS 是二分类正类，所有 OOS score 统一为“越大越像 OOS”；
- Gate-only 指标与端到端意图分类指标分表报告；
- TextOIR 与 s2c 的数据协议和种子独立记录，禁止直接混表；
- 每次运行必须保存 manifest、数据 hash、阈值来源和缺失单元审计；
- Gate-only 的逐样本结果统一以 `scores.parquet` 为准，不再同时复制一份
  大体积 `predictions.json`；TextOIR 外部协议保留独立 `predictions.jsonl`。
- 历史 `paper_results`、detector 和 checkpoint 不得被覆盖。
