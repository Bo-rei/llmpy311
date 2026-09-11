# AI 配置审查记录

> GPT-6 适配变更说明：记录本次配置收口的范围与证据，不把模型效果、速度或 Token 变化当作已验证结论。

审查范围：工作区 `AGENTS.md`、`s2c/AGENTS.md`、`/home/bo/.codex/config.toml`、实际可见的 Skills 入口，以及 OMX 生成的 Hooks、角色和状态目录。

已处理：

- 移除 OMX 的全局注入、MCP/Hook、工作流 Skill、角色提示、项目 `.omx` 状态和 Node 全局包；保留可恢复备份在 `/tmp/omx-removal-backup-20260910.tar.gz` 与 `/tmp/omx-home-state-20260910`。
- 规则优先级固定为用户任务、项目规则、通用 Skill；常规细节自主调查，普通错误自行修复，验证按改动风险选择。
- 默认配置为 `gpt-6-astra` + `low`；`[profiles.astra]` 和 `[profiles.luna]` 分别提供 Astra low 与 GPT-5.6 Luna max。两者共用同一套项目规则。
- 收窄文档协作、全文阅读、Nature/CNS 引用和图表后端的自动触发；修正论文检索/创新性审查的旧路径与不存在工具名；保留来源、数据、版权、权限和证据要求。

未处理：插件缓存和与当前任务无关的 Skills 没有批量重写；它们按任务触发，并受本项目优先级规则约束。完整模型行为、成本和效果需要真实使用或 benchmark，当前没有这类证据。
