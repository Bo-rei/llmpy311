# LLMPY311 工作区

这是一个多项目研究工作区，不是单一 Python 包。

- `s2c/`：当前活动项目，入口是 [s2c/README.md](s2c/README.md)。
- `assets/`：本地数据集和基础模型，只供运行时使用。
- `artifacts/`：原始实验产物、checkpoint、embedding 和逐样本输出，保持 Git 忽略。
- `archives/`：本地历史材料和旧运行日志，保持 Git 忽略。
- `textoir/`：独立 Git 仓库，父仓库不纳入其内容。

公开、可提交 GitHub 的轻量结果位于 `s2c/results/`；完整结果仍保留在
`artifacts/`，两者不能混用。
