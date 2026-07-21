# s2c 运行手册

本手册只保留环境检查、测试、实验登记审计和公开结果导出命令。本轮整理不运行训练，
也不修改 `../artifacts`。

## 环境检查

```bash
cd /home/bo/bo01/llmpy311/s2c
# 项目环境是 bo；未激活时也可把下面的 python 替换为该环境的绝对路径
conda activate bo
python --version
python -c "import numpy, pandas, yaml; print('runtime imports: ok')"
```

## 结果登记审计

```bash
python tools/analysis/audit_experiment_registry.py
```

该命令只检查当前登记表指向的入口、manifest、汇总文件和 unit count；它不会选择 K、
阈值或重跑模型。完整原始产物仍在 `../artifacts/s2c/`。

## 公开结果导出

先查看白名单和容量：

```bash
python tools/maintenance/export_public_results.py --dry-run
```

确认后执行并校验 SHA256：

```bash
python tools/maintenance/export_public_results.py --execute
python tools/maintenance/export_public_results.py --verify
```

导出脚本只复制 `configs/public_results.yaml` 明确列出的文件，并拒绝 checkpoint、模型、
embedding、逐样本输出和超过 10MB 的单文件。

## 测试与静态检查

```bash
pytest tests/unit -q
python -m py_compile \
  tools/maintenance/export_public_results.py \
  tools/analysis/audit_experiment_registry.py
git diff --check
```

## 读取结果

- GitHub 结果：先看 `results/README.md`，再用 `results/MANIFEST.csv` 校验文件来源。
- 完整证据：先读对应 artifact 根目录的 manifest，再读 summary CSV。
- 不要将 Gate-only 的 Frozen/CE/SupCon 数字解释为完整 Cascade。
