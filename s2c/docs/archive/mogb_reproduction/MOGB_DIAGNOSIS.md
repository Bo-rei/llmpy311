# MOGB 四组合差异诊断

## 范围

固定目标为 StackOverflow、KIR=0.50、seed=0。本诊断只读取现有
strict artifact 和 pinned third-party checkout，不启动新的 MOGB 训练，不改写
`../artifacts`。

## 组合状态

| 组合 | 代码 | 数据 | 状态 |
|---|---|---|---|
| A | pinned original | original accompanying data | blocked_preflight |
| B | pinned original | current TextOIR snapshot | blocked_preflight |
| C | modern compatibility | original accompanying data | blocked_preflight |
| D | modern compatibility | current TextOIR snapshot | observed_existing_artifact |

## 首次分叉

`first_divergence.json` 和 `stage_comparison.csv` 保存了每一组合的来源哈希、标签/Known
列表哈希以及可用的训练和粒球字段。原始 MOGB 配套数据未在本地材料中发现，因此不能把
TextOIR 数据伪装成 A/C；pinned checkout 的静态最早失败点是
`MOGB.py:3` 导入缺失的 `utils` 包。D 使用已有 strict artifact，未重新执行。

## 结论

最终分类：**public_code_not_reproduced_under_available_materials**。

这不是对数据或代码单独归因的结论：缺失 A/C 意味着无法通过四组合实验隔离数据契约与
兼容实现。现有 D 仍应保持 `not_reproduced_strict`，不能与论文数字直接混合，也不能由此
继续添加非必要兼容补丁。

## 证据位置

* `results/diagnostics/mogb_diff/stage_comparison.csv`
* `results/diagnostics/mogb_diff/first_divergence.json`
* `third_party/mogb_official/ORIGINAL_SOURCE.md`
* `third_party/mogb_official/PATCH_LOG.md`
* `../artifacts/s2c/external/mogb_exact_reproduction_v1/audit/`
