# MOGB 复现差距与方法合同分析（V2）

> 结论先行：本地 MOGB 单格不是简单的“没训练好”。闭集 CE 与 Known dev accuracy 已收敛；主要异常是公开代码中的子中心损失被 L1 距离归一化压缩，以及最终平均半径边界严重拒绝 Known。由于作者原始数据和旧环境不完整，本报告只能定位本地差距，不能据此否定论文结果。

## 1. 现在到底在比较哪些方法

| 名称 | 实际方法 | 合同 | 可回答的问题 |
|---|---|---|---|
| fulltex `Ours` | 冻结 MiniLM 多中心 Gate + SmolLM Router/Expert 的完整 Cascade | 历史论文合同 | 为什么旧论文 OOS F1 高于其表内基线 |
| `S2C-Trainable-K1` | Known-only 训练 MiniLM 最后两层与 projection + 单中心 Gate | 当前 protocol_v2，五 seed | 当前自有 Gate 候选是否优于冻结/固定多中心组件 |
| `MOGB-MiniLM-Fair` | 同一冻结 MiniLM embedding + MOGB 粒球/平均半径 | 当前 fair 组件合同 | 固定表示下，粒球与边界组件是否有效 |
| MOGB exact local | BERT + CE + 最近子中心损失 + 自适应粒球 + 平均半径 | 官方逻辑的现代兼容单格 | 公开实现能否在现有材料下接近论文数字 |
| MOGB paper | 论文报告的完整方法 | 作者原始合同 | 公开参考，不与当前 Gate 行直接排名 |

因此，“我的方法超过谁”必须带限定：历史 fulltex `Ours` 在旧表中比较的是 MSP、OpenMax、DOC、DeepUnk、KNNCL、ADB、DA-ADB；当前 `S2C-Trainable-K1` 只在当前统一 Gate/组件矩阵中优于 Frozen K1/K2、random K2 和 MOGB-MiniLM 组件。它尚不能宣称超过完整 MOGB、DCLOOS 或历史完整 Cascade。

## 2. 本地 exact 单格有没有真正训练

- StackOverflow KIR=.50：训练 38 epoch，best epoch=28，Known dev accuracy=91.60%，CE loss 2.3070→0.1199。
- Banking77 KIR=.75：训练 54 epoch，best epoch=44，Known dev accuracy=91.84%，CE loss 4.0635→0.1260。

所以不能再把主要差距解释成 epoch 太少或 CE 完全未收敛。另有十个五 epoch 兼容运行，它们只是执行稳定性证据，明确 `strict_official_reproduction=false`，不能放进论文主表。

## 3. 子中心损失为什么可疑

公开 `myloss.py` 对每个样本先计算到各类别最近粒球中心的距离，然后执行 `F.normalize(distances, p=1)`，再计算 `softmax(-distances)`。距离非负且归一化后总和为 1，这把所有类别 logit 压在 [-1,0] 的极窄区间。

对 StackOverflow 的 10 个 Known 类，均匀概率为 0.1000，即使达到该代码约束下的理论最佳，true-class probability 也最多为 0.1105；子中心交叉熵只能从 log(10)=2.3026 最多降到 2.2032。实测末轮为 2.2586。

对 Banking77 的 58 个 Known 类，均匀概率为 0.01724，理论最大也仅 0.01754；loss 的全部理论动态范围只有 0.0172，实测 4.0581→4.0515。类别越多，这个压缩越严重。

这不证明论文概念无效，但证明当前公开实现的最近子中心目标很难形成高置信的类别梯度，尤其在 Banking77 的多 Known 类设置下。

## 4. 最终边界的主要失败不是误收 OOS，而是拒绝 Known

StackOverflow：OOS Recall=98.80%，Known Recall=51.53%，Known→OOS=1449，OOS→Known=36。
Banking77：OOS Recall=97.89%，Known Recall=43.71%，Known→OOS=1298，OOS→Known=16。

这说明 MOGB 本地工作点非常保守：平均距离半径能拒绝几乎全部 OOS，却只覆盖约一半 Known。闭集 checkpoint 是按 Known dev accuracy 选择的，并没有校准最终多粒球边界的 Known coverage，因此 dev accuracy 约 92% 与 test Known Recall 约 44%–52% 可以同时出现。

## 5. 动态粒球是否实际运行

是。最终 StackOverflow 有 28 个粒球、10 个 Known 类；Banking77 有 99 个粒球、58 个 Known 类。球样本数均值分别为 177.5 和 54.4，平均半径为 2.578 和 4.403。所以差距不是“代码退化成单中心”，而是表示损失、粒球结构、平均半径和 checkpoint 选择之间没有形成论文报告中的工作点。

## 6. 与论文的差距

StackOverflow 的本地 exact 单格相对论文公开参考：Accuracy -13.50 pp、F1-All -19.14 pp、F1-U -9.74 pp、F1-K -20.08 pp。Known 侧差距明显大于 Unknown 侧，和上述过度拒绝一致。Banking 本地单格也显著偏低，但其本地 KIR=.75 与公开参考的精确合同仍未闭合，只能描述，不能作严格一一复现判定。

## 7. 现代兼容层带来的不可消除差异

官方 `pretrain.py` 先完成 CE optimizer step，再把 CE 更新后的 features 保存在 memory bank，最后对整库 loss2 反向传播；现代 PyTorch 会遇到 stale graph。兼容层改为：no-grad 重新提取 feature bank 生成粒球，再对固定 centroids 做第二次可微前向和 optimizer2 更新。这是合理修复，但不是作者旧环境的逐字节执行，可能改变梯度、聚类更新和早停轨迹。

另外，作者原始数据、逐样本 ID、Known 类列表、完整旧依赖环境均未恢复；pinned checkout 本身还缺少 `utils` 包。四组合静态诊断在 `results/diagnostics/mogb_diff/`，结论仍是 `public_code_not_reproduced_under_available_materials`。

## 8. 为什么当前 S2C 行看起来比本地 MOGB 好

当前 `S2C-Trainable-K1` 的 Known-only 训练直接改善 Gate score separation，且 S2C 的 `μ+λσ` 边界比 MOGB 平均半径保留更多 Known coverage。相反，本地 MOGB exact 的 OOS Recall 已接近 98%，但 Known Recall 只有约一半；MOGB-MiniLM-Fair 更是同样呈现低 false acceptance、高 false rejection。当前优势主要是工作点更平衡，不是已经证明 S2C 全面超过论文完整 MOGB。

## 9. 新增证据文件

- `results/analysis/archive/analysis/mogb_reproduction_gap_analysis_v2/exact_run_summary.csv`：两个 exact 单格；
- `loss_trajectory.csv`：CE、子中心 loss 和理论界；
- `paper_gap.csv`：论文公开参考差值；
- `ball_distribution.csv`：127 个最终粒球；
- `five_seed_compatibility_runs.csv`：十个五 epoch 非严格运行；
- `root_cause_evidence.csv`：各归因假设的证据状态；
- `figures/archive/analysis/mogb_reproduction_gap_analysis_v2/`：五张图。

## 10. 当前结论和下一步

1. 历史 SOTA 方法是完整 Cascade `Ours`，不是当前 Trainable-K1，也不是 RC-AMBL。
2. 现有可视化很多，但之前没有把方法合同、MOGB loss 数学动态范围和边界过拒串成一条证据链；本报告补齐了这部分。
3. 不应继续盲目扩大 MOGB 复现矩阵。若要追近论文，优先做两个受控诊断：去掉距离 L1 归一化/加入温度的 loss 公式校验；在不看 test OOS 的前提下，用 Known calibration 检查 mean radius 的 coverage。二者必须注册为 adapted ablation，不能冒充 official reproduction。
4. 当前论文级公平结论仍需同一数据、Known 列表、split、seed 和评估器下的完整 S2C Cascade、MOGB、ADB/DA-ADB/DCLOOS 主表。
