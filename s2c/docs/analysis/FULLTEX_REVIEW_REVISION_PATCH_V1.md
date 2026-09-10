# `fulltex.tex` 审稿修订补丁

基准文件：`fulltex.tex`  
基准 SHA-256：`83322d7dff65ba22e489c67aaa2bfe70a8afb8ef8526e0a53294dfaf4557a61e`  
生成时间：2026-09-09

本文件是可人工应用的修订清单，不自动修改 `fulltex.tex`。所有行号基于上述 hash；如果源稿先发生其他改动，应优先用句子片段定位而不是机械套用行号。

对应的机器可执行 unified diff：[FULLTEX_REVIEW_REVISION_V1.patch](FULLTEX_REVIEW_REVISION_V1.patch)。当前已对源稿执行 `patch --dry-run`，并在临时副本中实际应用后检查关键替换句；没有执行源稿写入。

## 1. Abstract（约第 147 行）

将：

```text
The results show that the method achieves the state-of-the-art OOS intent detection performance compared the other baselines. Ablation studies are also conducted and the results show that the used MiniLM can better adapt to the workflow and utterance embedding requirements.
```

改为：

```text
Under the reported data, backbone, supervision, and evaluation contracts, the method achieves competitive OOS intent detection performance against the evaluated baselines. The ablations show that Known-only adaptation of the MiniLM Gate improves OOS score ordering and full-pipeline OOS F1 relative to a matched Frozen Gate; this evidence is specific to the evaluated boundary and downstream contracts.
```

理由：当前证据不支持无条件 SOTA，也不支持 MiniLM 普遍优于 Frozen 或自动改善多中心几何。

## 2. Introduction（约第 164–172 行）

将：

```text
Moreover, we found that MiniLM can embed the observation data into multiple clusters. Thus, if the boundaries of clusters are learned, the out-of-domain utterances can be identified as OOS intents.
```

改为：

```text
Moreover, we investigate whether MiniLM embeddings contain useful local structure for OOS rejection. The resulting local boundaries are treated as an empirical modeling choice rather than a universally valid assumption, because additional clusters can also enlarge the acceptance region for near-OOS samples.
```

将贡献 1 改为：

```text
An explicit lightweight OOS Gate is developed with local geometric boundaries. The experiments characterize when multi-cluster structure improves rejection and when it increases the acceptance-region risk, rather than claiming that multiple centroids are universally superior.
```

将贡献 3 中：

```text
The results show that the method achieves the state-of-the-art performance for OOS intent detection. Moreover, ablation studies are also conducted, and the results show that the used MiniLM ... can better adapt to the workflow and utterance embedding requirements.
```

改为：

```text
The experiments report OOS F1 together with Known-side coverage and error metrics under explicit data, supervision, and evaluator contracts. The Trainable-versus-Frozen ablation shows a contract-specific benefit from Known-only MiniLM Gate adaptation; it does not establish universal superiority over all backbones or boundary geometries.
```

## 3. Related Work：Hendrycks / MSP（约第 185 行）

将：

```text
set an extra OOS class, and transform the open-set classification into a closed-set one. Then, the method assigns regular scores to each class through softmax probability.
```

改为：

```text
use the maximum softmax probability as a confidence score for detecting misclassified or out-of-distribution inputs. The classifier remains a closed-set classifier; OOS rejection is obtained by thresholding the confidence score rather than by adding a supervised OOS class.
```

## 4. Related Work：DOC（约第 187 行）

将：

```text
If the softmax score is over the threshold, the input utterance would be rejected.
```

改为：

```text
The input is rejected when its class-wise sigmoid confidence pattern fails DOC's one-versus-rest rejection criterion; the criterion should not be described as a softmax threshold.
```

## 5. Related Work：uncertainty-aware routing（约第 191 行）

将 Zaera 相关表述：

```text
only triggers a fine-tuned LLM according to the Monte Carlo dropout criterion. But the routing decisions are still unstable if there exists distribution drift.
```

改为：

```text
uses predictive uncertainty, estimated with stochastic forward passes such as Monte Carlo dropout, to decide whether an input should be processed by a larger model. Its selective-routing objective and supervision should be distinguished from the explicit geometric OOS Gate studied here; claims about distribution-drift instability require the corresponding experimental evidence.
```

同时将 `multi-agent LLMs require large computation cost` 改为更窄的：

```text
LLM-based routing can introduce additional inference and engineering costs, motivating an explicit lightweight Gate for the front end.
```

## 6. Related Work：MOGB 定位（约第 187 行之后）

在 MOGB 引用之后补充：

```text
Existing work has therefore already explored adaptive class boundaries, multiple local regions, and multi-granularity structures. Our distinction is the combination of an explicit lightweight MiniLM Gate, cascade decoupling before Router and Expert, and an empirical analysis of when local geometry improves OOS rejection or enlarges the acceptance region. The existence of multiple centroids alone is not claimed as a new contribution. A same-backbone MOGB-style component comparison and the status of the official-BERT reproduction are reported separately because their contracts are different.
```

## 7. Baseline / implementation section（约第 313–329 行）

在 DA-ADB 段落后补充：

```text
MOGB is reported in two separated forms: a same-backbone MOGB-style component comparison under the controlled MiniLM contract, and an official-code compatibility audit. The latter is not treated as a byte-identical reproduction of the full BERT-based method because the original environment and complete artifact contract are unavailable.
```

将 `The models are all trained on NVIDIA RTX 5070 GPU` 改为：

```text
The available historical training and inference manifests record CUDA for the completed H1 runs; Gate-level deployment measurements are reported separately and are limited to the tested CUDA device, batch sizes, and Gate scope.
```

## 8. Results interpretation（约第 369 行）

将：

```text
our method maps user utterance into multiple clusters, that helps draw a clearer boundary compared to single cluster
```

改为：

```text
the evaluated local geometry can improve the boundary on some dataset/KIR settings, but the K-sensitivity experiments show that additional clusters are not uniformly beneficial and may increase OOS acceptance-region coverage.
```

将：

```text
if the other baseline methods can perform better than ours, it is easy to replace the router-experts pipeline with that method.
```

改为：

```text
the cascade permits the downstream Router/Expert to be replaced independently; this modularity does not remove the need to report the resulting Known/OOS trade-off under the same evaluator.
```

## 9. Gate geometry interpretation（约第 499–501 行）

将：

```text
According to our method, the threshold score strictly separates the two distributions of the known intent samples and OOS intent samples.
```

改为：

```text
The threshold defines an operating point in the score distributions. Overlap remains around the boundary, so the figure should be interpreted as a descriptive score-separation analysis rather than a strict separation guarantee.
```

将：

```text
That means, the distribution of the unknown intent samples do not influence the learning performances.
```

改为：

```text
This is an OOS-aware validation setting: unknown validation samples are used to select the historical radius coefficient. We therefore report it separately from the Known-only tuning contract and do not claim that the unknown distribution has no influence on the selected operating point.
```

## 10. Conclusion（约第 555 行）

将：

```text
The results show that the proposed method achieve the stete-of-the-art OOS intent detection performance. ... the lightweight models have outstanding advantages for real-world applications compared to the large models.
```

改为：

```text
The results show competitive OOS detection performance under the explicitly reported data, backbone, supervision, and evaluator contracts. The cascade separates the Gate decision from downstream known-intent classification, while the experiments also expose the Known-coverage cost and the limits of fixed multi-cluster geometry. The MiniLM Gate has a smaller parameter footprint than a large encoder, and the CUDA benchmark quantifies resource cost on the tested H1 K=1 Gate-only and full-Cascade paths; broader deployment advantages require additional device and system-level measurements.
```

## 应用前检查

- 将 MOGB、K 值、Known-only/OOS-aware tuning、Known F1 公式和 coverage–rejection trade-off 写入实验部分；
- 在稿件或 supplementary 中加入 [`official_mogb_status.csv`](../../results/analysis/cross_method_oos_mechanism/official_mogb_status.csv) 对应的合同说明；
- 保留 `fulltex.tex` 当前版本的 hash，应用后重新生成 PDF 并检查表格、图注和交叉引用；
- 不把 H1 Trainable 结果回填为严格 H0，也不把 MOGB compatibility 数字标成官方严格复现。
