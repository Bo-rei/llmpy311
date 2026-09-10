# MOGB 官方代码合同逐行审计 V1

更新时间：2026-08-09  
上游 commit：`5b689e2a03de0d86ec41212825e5db8d7f0e5c02`  
目的：把已经完成的 MOGB 结果归因落实到可复核的代码行为；本文件不是新的性能实验，也不把兼容层结果称为严格官方复现。

## 1. 关键代码行为

| 文件与行 | 实际行为 | 对结果的可能影响 | 当前证据 |
|---|---|---|---|
| `myloss.py:61-78` | 计算各类别最近粒球距离，随后 `F.normalize(distances, p=1)`，再 `softmax(-distances)` | 距离动态范围被压缩；Known 类越多，最近子中心监督越弱 | 实际距离表上的真类概率/梯度已量化；StackOverflow/Banking exact 单格已收敛但 Known Recall 仅 51.53%/43.71% |
| `cluster.py:44-60` | 虽然创建了 `self.device`，索引张量仍硬编码到 `cuda:0` | CPU、非零 GPU 或隔离设备无法按参数稳定运行 | 官方源代码审计；现代适配层必须修复，不能视为原始环境等价 |
| `cluster3.py:83-103` | 当纯度低于阈值且样本数大于最小值时递归拆分 | 粒球数由纯度/支持度规则产生，而不是遍历 K 选择 | 最终 StackOverflow 28 球、Banking 99 球，动态划分确实执行 |
| `cluster3.py:142-204` | 从混合球中随机抽取其他类别样本作为候选中心，再按距离重分配 | Python `random` 状态会改变粒球成员、中心和半径 | 源运行未保存最终 RNG/中心成员状态，当前 33 球重建明确不是严格球身份重放 |
| `cluster3.py:234-248` | 初始中心随机选择后递归调用 `splits` | 相同 seed 只有在完整运行时、版本和 RNG 状态一致时才可复现 | 四组合诊断将旧环境/数据/状态缺失标为合同风险 |
| `cluster3.py:265-279` | 选择粒球还要满足最小样本数和 `purity_select_ball` | 某些 Known 类可能没有 selected ball，造成 Known 覆盖损失 | selected-class rescue 已验证缺类是真实风险，但不是主要差距来源 |
| `cluster3.py:51-55` | `calculate_center_and_radius` 使用球内均值中心和样本到中心距离的平均值 | 平均半径容易形成过窄 Known 接受区域 | Known-only cal-80/95 显示扩大半径会快速增加 OOS 误接受 |
| `gb_test.py:91-110` | 对所有球取最近中心，仅当最近球距离小于其半径时接受 | 多球边界采用最近球+单球半径，而不是全局覆盖校准 | MOGB-Fair 低 false acceptance 伴随高 Known false rejection |
| `init_parameter.py:70-75` | 纯度参数默认值为小数，但 argparse 类型写成 `int` | 直接按脚本命令传入小数时存在参数解析风险 | 兼容层显式固定参数；原始脚本不能作为现代环境直接入口 |
| `run.sh:1-11` | `#!/usr/bin bash`、孤立 `do/done`、`--seed 0\\` 等旧脚本问题 | 原始入口无法直接证明可运行性 | 已保留原始源码并在外部适配层修复 |

## 2. 与当前 S2C 公平组件的对应关系

当前 `MOGB-MiniLM-Fair` 保留了上述 MOGB 风格粒球、欧氏距离和平均半径，但使用统一的 Frozen MiniLM、TEXTOIR registry 和当前 evaluator；它不包含官方 BERT 表示训练。因此它适合回答“同一轻量表示下组件差异”，不适合证明“完整 MOGB 论文方法失败”。

当前 `S2C-Trainable-K1` 使用 Known-only 训练后的 MiniLM 和单中心边界。它在同协议中较高的 F1-All/比较稳定的 OOS F1，主要对应两项可观察差异：

1. 表示的 score separation 更好；
2. 边界工作点恢复了更多 Known，而没有像 MOGB-Fair 那样极端保守。

## 3. 归因结论

目前可以确认的因果链是：

```text
官方 loss 压缩
        ↓
子中心训练信号弱
        ↓
粒球表示/中心质量不足
        ↓
平均半径工作点偏窄
        ↓
Known false rejection 很高
```

但这不是完整解释。作者原始样本 ID、Known 类列表、旧依赖、最终球成员和 RNG 状态仍未恢复，因此当前 MOGB 只能标记为 `official_code_not_reproduced_under_available_materials`。

## 4. 当前不再重复的实验

- 不再重复同一 MOGB-Fair KIR/seed 矩阵；
- 不再继续用测试 OOS 调半径；
- 不再把确定性球重建称作严格原球 replay；
- 不把兼容单格或 reduced baseline 结果并入当前 SOTA 主表。

## 5. 证据入口

- `docs/archive/analysis/MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md`
- `docs/archive/analysis/MOGB_CORRECTED_RADIUS_COVERAGE_V1.md`
- `docs/archive/analysis/MOGB_SELECTED_CLASS_RESCUE_V1.md`
- `docs/archive/analysis/MOGB_BALL_RISK_ATTRIBUTION_V1.md`
- `results/diagnostics/mogb_diff/`

