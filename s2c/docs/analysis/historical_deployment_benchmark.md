# Historical H1 Gate and Cascade deployment benchmark

更新时间：2026-09-10。

这是对审稿意见中“参数量、延迟、吞吐和显存”要求的 Gate 与 Cascade 范围补充。实验不改变模型、数据、阈值或历史结果，只测量已有 `historical_v19_paper_main` H1 配置：KIR=.50、seed=42、CUDA、Frozen K=1 与 Trainable K=1，覆盖 CLINC150、StackOverflow 和 `banking77_oos`。

## 结果

下表的延迟是每个样本的毫秒数，吞吐是其倒数换算得到的样本/秒；每个配置先 warm-up 5 次，再测量 20 次。显存为 CUDA peak allocated。

### Gate-only

| 数据集 | Gate | 总参数 | 可训练参数 | batch=1 延迟 / 吞吐 | batch=32 延迟 / 吞吐 | batch=1 显存 | batch=32 显存 |
|---|---|---:|---:|---:|---:|---:|---:|
| CLINC150 | Frozen | 22.71M | 0 | 3.61 ms / 277.0/s | 0.49 ms / 2050.7/s | 360.8 MB | 369.4 MB |
| CLINC150 | Trainable | 22.91M | 3.75M | 3.02 ms / 330.8/s | 0.43 ms / 2305.5/s | 449.1 MB | 457.7 MB |
| StackOverflow | Frozen | 22.71M | 0 | 3.41 ms / 293.3/s | 0.32 ms / 3103.4/s | 96.1 MB | 111.4 MB |
| StackOverflow | Trainable | 22.91M | 3.75M | 3.07 ms / 325.8/s | 0.29 ms / 3407.6/s | 183.5 MB | 199.5 MB |
| BANKING77-OOS | Frozen | 22.71M | 0 | 3.26 ms / 307.2/s | 0.33 ms / 2993.2/s | 183.2 MB | 197.8 MB |
| BANKING77-OOS | Trainable | 22.91M | 3.75M | 2.68 ms / 372.9/s | 0.29 ms / 3434.5/s | 270.6 MB | 285.2 MB |

### Full Cascade

| 数据集 | Gate | batch=1 延迟 / 吞吐 | batch=32 延迟 / 吞吐 | batch=1 显存 | batch=32 显存 |
|---|---|---:|---:|---:|---:|
| CLINC150 | Frozen | 3.44 ms / 290.5/s | 2.74 ms / 365.5/s | 360.8 MB | 629.6 MB |
| CLINC150 | Trainable | 2.89 ms / 346.6/s | 27.04 ms / 37.0/s | 449.1 MB | 1413.0 MB |
| StackOverflow | Frozen | 3.48 ms / 287.3/s | 1.35 ms / 741.6/s | 96.1 MB | 389.7 MB |
| StackOverflow | Trainable | 3.83 ms / 261.2/s | 0.31 ms / 3266.9/s | 183.5 MB | 199.5 MB |
| BANKING77-OOS | Frozen | 3.15 ms / 317.4/s | 1.44 ms / 695.7/s | 183.2 MB | 482.3 MB |
| BANKING77-OOS | Trainable | 2.73 ms / 366.5/s | 1.27 ms / 786.9/s | 270.6 MB | 568.9 MB |

机器可读结果：[summary.csv](../../results/analysis/historical_deployment_benchmark/summary.csv)，运行信息：[MANIFEST.json](../../results/analysis/historical_deployment_benchmark/MANIFEST.json)。

## 可写入论文的范围

这组结果支持：Trainable Gate 只解冻部分 MiniLM 与 residual projection，参数量和 Gate/Cascade 的 CUDA 资源开销可以分别量化。Gate-only 与 full Cascade 使用不同调用入口；前者只执行 `_gate_predict`，后者执行完整 Gate→Router→Expert。结果不支持“完整级联系统在所有设备和 batch size 下都更快/更省”的结论：benchmark 只覆盖单一 CUDA 环境和 batch=1/32，吞吐数值只代表当前设备和当前组件缓存状态。
