# s2c 文档总入口

这份文件只负责导航和解释“当前文档的可信边界”。不要把 `docs/` 中所有 Markdown 都当成同一时刻、同一协议下写成的说明：项目经历过多轮实验，部分文件保留的是历史方案或论文草稿。

## 一分钟理解项目

```text
文本 → Gate：Known / OOS
     → Router：Known 样本所属 domain
     → Expert：domain 内的具体 intent
```

当前论文修订的重点不是重写级联，而是研究 MiniLM 语义空间中的 Known 多簇结构，以及它与 OOS、尤其 near-OOS 的可分性。

## 三个必须先分清的对象

### 1. 源码

当前 checkout 是 `main` 分支时，核心可运行代码在：

```text
src/                         可复用系统实现
tools/analysis/              v19 训练、评估和历史分析入口
tools/eval/                  系统评估入口
tools/experiments/           v19/v20/v21 多簇研究入口
scripts/data/active/         数据重建入口
configs/                     配置和 artifact 注册
tests/                       回归与协议测试
```

更晚的 v22/研究收口实现位于分支 `autoresearch/minilm-collision-20260718`。当前 `main` 不包含该分支后续新增的所有文件，但仍能看到它们留下的实验结果。

### 2. 结果

结果在工作区外的忽略目录：

```text
../artifacts/s2c/outputs/experiments/
```

忽略目录不会随 `git switch` 自动回滚，所以 `main` 上也可能存在 `cluster_separability_v21`、`minilm_semantic_collision_v22`、`study_closeout` 等目录。它们可以被读取，但必须以 manifest 中记录的源码 commit 为准。

### 3. 文档

本目录同时包含当前说明和历史材料。当前事实优先级如下：

```text
closeout_manifest / run_manifest
    > 结果汇总 CSV/Parquet
    > 当前源码与测试
    > 本目录 active 文档
    > 旧论文草稿和历史计划
```

这里的“优先级”是指发生冲突时的证据顺序；不是说旧文档没有价值，而是它们不能覆盖当前 manifest。

下文中的代码路径和 `../assets`、`../artifacts` 均以 `s2c/` 项目根为相对起点；从某个
Markdown 文件所在目录点击时，请按工作区实际布局理解。

## 推荐阅读路径

### A. 只想先看懂项目

1. [项目完整说明](00-项目完整说明.md)
2. [数据协议与版本](03-数据协议与版本.md)
3. [训练与评估流程](04-训练与评估流程.md)
4. [当前状态与里程碑](07-当前状态与里程碑.md)

### B. 想跑当前 `main` 的基础流程

1. [项目完整说明](00-项目完整说明.md) 的“主源码路径”部分
2. [训练与评估流程](04-训练与评估流程.md)
3. [工具入口索引](10-工具入口索引.md) 的“main 分支入口”部分
4. [运维与治理规范](06-运维与治理规范.md)

### C. 想读 MiniLM 多簇/OOS 研究结果

1. 先看 [当前状态与里程碑](07-当前状态与里程碑.md) 的“研究收口”表；
2. 再看 [工具入口索引](10-工具入口索引.md) 的“研究分支入口”表；
3. 打开 `../artifacts/s2c/outputs/experiments/study_closeout/closeout_manifest.json`；
4. 按 `result_inventory.csv` 找到具体汇总 CSV；
5. 用 `paper_source_map.csv` 核对论文表/图的来源；
6. 用 `known_gaps.json` 判断哪些结论仍不能外推。

### D. 需要追历史

再读下面这些材料：

- [01-项目总览](01-项目总览.md)
- [01-项目通俗详细解析](01-项目通俗详细解析.md)
- [05-实验结果与性能基线](05-实验结果与性能基线.md)
- [08-消融实验方案](08-消融实验方案.md)
- [09-消融方案的代码可行性分析](09-消融方案的代码可行性分析.md)
- [12-s2c 意图识别方案技术底稿](12-s2c意图识别方案技术底稿.md)
- [13-本次改动记录](13-本次改动记录.md)
- [14-项目结构重整与职责边界](14-项目结构重整与职责边界.md)

这些文件适合查背景和历史决策，不适合作为“今天该运行什么”的唯一依据。

## 当前主线和实验边界

### 已完成或已有证据

- v19 主网格：KIR、每意图子中心数 `K_gate`、Euclidean/diagonal Mahalanobis；
- v20：KMeans 与随机分簇、K 选择可靠性、near/medium/far OOS、效率和自适应边界对照；
- v21：Frozen MiniLM 邻域纯度、类内/类间几何、语义 margin、near-OOS 表示/边界分解，以及 Frozen/CE/SupCon 表示对照；
- 研究收口：代表性 Cascade、外部 hard-negative 和 MOGB 协议审计。

### 不能过度声称的部分

- `K>1` 不是所有数据集都有效；StackOverflow 有稳定退化现象；
- CE-Recon 或任何表示适配不能只凭 Gate-only 结果声称完整 Cascade 普遍提升；
- 12 个 Cascade 单元是 KIR50/seed42 的代表性 smoke，不是完整三 seed 结论；
- MOGB 只有审计状态，没有公平、可放入主表的复现数字；
- 机制统计目前用于相关性和描述，不写成因果中介。

## 常用核对命令

先在 `s2c/` 目录执行：

```bash
git status --short
git branch -a -vv
python -m src.cli --help
python -m src.cli prepare --dataset clinc150 --kir 50 --seed 42 --dry-run
python -m tools.experiments.cluster_separability --help
```

查看收口文件：

```bash
find ../artifacts/s2c/outputs/experiments/study_closeout -maxdepth 1 -type f -print | sort
```

如果要运行研究分支专有命令，先检查工作树干净，再切换分支；不要在未保存改动时直接切换。

## 文档维护规则

- 本文件和 `00`、`07`、`10` 是当前理解项目的主入口；修改实验协议时优先同步它们。
- 旧编号文件不自动删除，先在本入口标注其历史属性，避免破坏已有论文引用。
- 不新增 `v24/v25/...` 式的平行 Markdown 入口；结果按研究主题归档，轮次只保留在 manifest 和 commit 中。
- 任何新结论都必须写出：结果文件、源码 commit、数据 split/hash、选择集（train/validation/test）。
