# Related Work 与 Claim 修正文案

本文档是对当前 fulltex.tex 的逐句修正文案。fulltex.tex 保持不修改，下一次论文改稿时按本文件替换。

## Hendrycks / MSP

当前风险表述：

> Hendrycks sets an extra OOS class and transforms open-set classification into a closed-set one.

建议改为：

> Hendrycks et al. introduce a confidence-based baseline that uses the maximum softmax probability to detect misclassified or out-of-distribution inputs. The classifier itself remains a closed-set classifier; OOS rejection is obtained by thresholding the confidence score rather than by adding a supervised OOS class.

这样可以区分“闭集分类器上的置信度拒识”和“显式加入 OOS 类别”。

## DOC

当前风险表述：

> DOC replaces softmax with independent sigmoid classifiers and rejects an utterance if the softmax score is over the threshold.

建议改为：

> DOC replaces the single softmax classifier with independent one-versus-rest sigmoid classifiers and rejects an input when the class-wise confidence pattern fails the method's rejection criterion.

不要把 sigmoid 分数写成 softmax score，也不要在 Related Work 中固定写 threshold 的方向；具体方向应留给方法原文和实现合同。

## MC Dropout / uncertainty-aware routing

建议使用保守表述：

> Uncertainty-aware routing methods use predictive uncertainty, often estimated with stochastic forward passes such as Monte Carlo dropout, to decide whether an input should be processed by a larger model. Their primary focus is selective routing under uncertainty; the routing criterion and training supervision should be distinguished from the explicit geometric OOS Gate studied here.

不要把某一篇方法概括为“只触发 fine-tuned LLM”或直接推断其在 distribution drift 下不稳定，除非正文给出该论文的具体实验依据。

## MOGB novelty positioning

当前方法不应写成“已有方法只有单中心，而本文首次提出多中心边界”。建议改为：

> Existing work has explored adaptive class boundaries, multiple local regions and multi-granularity structures for open intent detection. Our focus is a lightweight cascade in which an explicit MiniLM Gate performs OOS rejection before the Router and Expert, together with an analysis of when local multi-cluster geometry helps or enlarges the acceptance region. The contribution is therefore the combination of lightweight front-end rejection, cascade decoupling and an empirical characterization of the coverage–rejection trade-off, rather than the existence of multiple centroids alone.

MOGB 应在主表或明确标注的同协议 component supplement 中出现。若完整官方 MOGB 因 checkpoint/环境缺失无法严格复现，应同时报告“component comparison”和“official reproduction status”，不要把兼容运行数字写成完整 MOGB 的正式 SOTA 行。

## Trainable MiniLM claim

建议将：

> MiniLM can better adapt to the workflow and utterance embedding requirements.

改为：

> Under the historical H1 contract, Known-only adaptation of the MiniLM Gate improves the OOS score ordering and full-pipeline OOS F1 relative to the matched Frozen Gate. This evidence is specific to the evaluated data, boundary and downstream contract; it does not imply that adaptation universally improves fixed multi-cluster geometry.

## Deployment claim

如果没有统一的 batch=1 latency、throughput、显存和参数量实验，建议将：

> lightweight models have outstanding advantages for real-world applications

收窄为：

> The cascade uses a lightweight MiniLM Gate and parameter-efficient downstream adaptation, which reduces the model scale relative to a fully fine-tuned large encoder. A deployment-level advantage requires separate latency and memory measurements.

当前 Frozen/Trainable K=1 的 Gate-only 与 full Cascade 已完成统一的参数量、batch=1/32 latency、吞吐和 CUDA 显存 benchmark；完整四变体、CPU 和更广 batch-size 范围仍未测量，因此修正文案只支持当前两种 H1 K=1 资源证据。
