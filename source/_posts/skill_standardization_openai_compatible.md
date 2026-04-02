---
title: "Skill 标准化：从私有格式到 OpenAI 兼容的工程决策"
date: 2026-04-02T21:30:00+08:00
draft: false
top_img: /img/covers/skill-standardization-openai-compatible.svg
cover: /img/covers/skill-standardization-openai-compatible.svg
tags: ["OpsMind", "Skill", "Agent", "工程化", "OpenAI兼容", "JSON Schema"]
categories: ["技术文章"]
author: "wwxdsg"
description: "这次改动不是修坏功能，而是把 Skill 从“前端模板快捷键”升级为 Agent 可感知、可调用、可复用的标准化能力。"
permalink: /opsmind/skill-standardization-openai-compatible/
---

![Skill Standardization Cover](/img/covers/skill-standardization-openai-compatible.svg)

> 这篇文章讲的是一次“把已经能用的东西推倒重做”的经历。  
> 不是因为原来的坏了，而是因为原来的方式把一个产品想法做窄了。

---

## 一、原来是怎么做的

OpsMind 上线了一套叫 Skill 的快捷分析功能。

用户打开输入框左下角的「+」菜单，切换到“快捷分析”标签，就能看到四个预置技能：数据体检、数据速览、维度对比、异常排查。点一下，填几个参数，系统自动把描述渲染成一段结构化的分析 prompt 发给 Agent。

从产品角度来说，这个功能是有价值的：让不会写 prompt 的用户也能得到高质量的分析指令。

但实现上，它有一个根本性的定位问题：

**Skill 只是用户端的 prompt 模板快捷键，Agent 完全不感知它的存在。**

流程是这样的：

```text
用户点击 Skill
  -> 前端调 /api/skills/render
  -> 返回渲染好的 prompt 字符串
  -> 以普通用户消息发给 Agent
  -> Agent 不知道这是 Skill，走普通 ReAct 流程
```

Agent 拿到的只是一段文字，和用户自己打字没有区别。

这意味着：Skill 永远不可能被 Agent 主动调用。你没办法说“分析完之后自动触发异常检测”，没办法让 Agent 根据数据特征主动选择合适的 Skill，没办法把 Skill 编排进复杂的多步分析链路。

Skill 的天花板，是“帮用户写好一句话”。

---

## 二、问题出在哪里

除了功能局限，原来的格式设计也有工程上的问题。

每个 Skill 的参数用自研格式定义：

```python
"arguments": [
    {
        "name": "dimension",
        "label": "对比维度",
        "placeholder": "例如：班级、部门",
        "required": True,
    }
]
```

这个格式只服务于一件事：让前端 SkillPicker 渲染一个表单。

它既不能被 Agent 的 function calling 机制识别，也不能被任何外部框架（LangChain、AutoGen、MCP）复用，还需要在后端和前端分别维护对这套自研结构的解析逻辑。

可选参数的从句处理更是硬编码的：

```python
period = args.get("period", "").strip()
args["period_clause"] = f"（{period}期间）" if period else ""

reference = args.get("reference", "").strip()
args["reference_clause"] = f"，参考基准为{reference}" if reference else ""
```

每加一个 Skill，就得在 `render_prompt()` 里加一段这样的逻辑。这不是架构，这是补丁。

---

## 三、标准格式是什么

OpenAI 在推出 function calling 时定义了一套参数格式，本质上是 JSON Schema：

```json
{
  "type": "object",
  "properties": {
    "dimension": {
      "type": "string",
      "description": "对比维度，如：班级、部门、地区"
    }
  },
  "required": ["dimension"]
}
```

这套格式后来被整个行业采纳。Anthropic、Google、LangChain、AutoGen、MCP 都认识这个结构。一个用这种格式定义的工具，理论上可以接入任何兼容 function calling 的框架，不需要任何适配。

它还有一个官方扩展机制：`x-*` 前缀字段。这是 JSON Schema 标准允许的自定义扩展，专门用来携带非标准元数据。比如：

```json
"dimension": {
  "type": "string",
  "description": "...",
  "x-label": "对比维度",
  "x-placeholder": "例如：班级、部门"
}
```

`x-label` 和 `x-placeholder` 是给前端 UI 读的，不影响 LLM 对参数的理解。标准的 LLM 会忽略这些扩展字段，前端可以读取它们来渲染表单。两套需求，一套定义，不冲突。

可选从句的配置也可以放进扩展字段：

```json
"period": {
  "type": "string",
  "description": "时间周期过滤（可选）",
  "x-clause-key": "period_clause",
  "x-clause-template": "（{value}期间）"
}
```

`render_prompt()` 泛化读取这个配置，对每个带 `x-clause-key` 的参数自动处理从句。加新 Skill 不需要动 `render_prompt()` 的代码。

---

## 四、改了什么

### 1. Skill 定义格式升级

`registry.py` 里每个 Skill 的参数字段从 `arguments` 改成了标准 JSON Schema `parameters`。

```python
# 之前
"arguments": [
    {"name": "dimension", "label": "对比维度", "placeholder": "...", "required": True},
    {"name": "period", "label": "时间周期（可选）", "placeholder": "...", "required": False},
]

# 之后
"parameters": {
    "type": "object",
    "properties": {
        "dimension": {
            "type": "string",
            "description": "对比维度（分组字段）",
            "x-label": "对比维度",
            "x-placeholder": "例如：班级、部门",
        },
        "period": {
            "type": "string",
            "description": "时间周期过滤（可选）",
            "x-label": "时间周期（可选）",
            "x-placeholder": "例如：2026年Q1",
            "x-clause-key": "period_clause",
            "x-clause-template": "（{value}期间）",
        },
    },
    "required": ["dimension", "metric"],
},
```

### 2. 新增 `to_tool_schema()`

```python
def to_tool_schema(skill):
    return {
        "type": "function",
        "function": {
            "name": skill["name"],
            "description": skill["description"],
            "parameters": skill["parameters"],
        },
    }
```

这个函数把 Skill 定义直接转成 Agent 可以使用的 tool schema。一行代码，不需要手工维护两套格式。

### 3. Agent TOOLS 动态扩展

在 `agent.py` 的模块加载阶段，从 registry 读取所有 Skill 并追加到 TOOLS 列表：

```python
from src.skills.registry import SKILLS as _SKILL_DEFS, to_tool_schema as _to_tool_schema

TOOLS.extend([_to_tool_schema(s) for s in _SKILL_DEFS])
```

Agent 启动后，TOOLS 从 6 个扩展到 10 个。LLM 在每轮推理时都能看到这 10 个工具，可以主动选择调用任何一个 Skill。

### 4. `_handle_invoke_skill()` 端到端执行

Agent 调用 Skill 工具时，会触发这个方法：

```python
def _handle_invoke_skill(self, skill_name, args, file_path, ...):
    query = render_prompt(skill_name, args)
    plan = self._handle_plan_analysis(file_path, query, ...)
    return self._handle_execute_analysis(file_path, query, plan, ...)
```

一次工具调用，完成从 prompt 渲染到图表生成的全链路。

### 5. 前端 `getSkillArgs()` 派生 UI 参数

前端不再维护独立的 `SkillArgDef[]` 解析逻辑，改为从 JSON Schema 动态派生：

```typescript
export function getSkillArgs(skill: SkillDef): SkillArgDef[] {
  const { properties, required } = skill.parameters
  return Object.entries(properties).map(([name, prop]) => ({
    name,
    label: prop['x-label'] ?? prop.description,
    placeholder: prop['x-placeholder'] ?? '',
    required: required.includes(name),
  }))
}
```

SkillPicker 的渲染逻辑、表单验证、自动触发全部不变，只是数据来源从 `skill.arguments` 换成了 `getSkillArgs(skill)`。

---

## 五、现在是什么效果

**用户端**：体验完全不变。SkillPicker 表单长得一样，填的参数一样，触发行为一样。

**Agent 端**：完全不同。现在 Agent 可以主动调用 Skill：

```text
用户：“帮我看看哪个部门表现异常”
Agent 推理后调用: anomaly-detective(target="哪个部门表现异常")
-> 自动完成 plan_analysis + execute_analysis
-> 输出排序图 + 异常分析文字
```

用户不需要打开 SkillPicker，不需要点击，不需要填表单。Agent 自己判断该用哪个 Skill，自己填参数，自己跑完整个分析流程。

**扩展性**：加新 Skill 只需要在 `registry.py` 里加一个 dict。Agent 侧自动获得新工具，前端自动渲染新表单，`render_prompt()` 不需要改。

**框架兼容性**：现在的 Skill 格式可以直接被任何支持 function calling 的框架识别，不需要任何适配层。如果未来接入 LangChain、AutoGen 或 MCP，Skill 定义可以原样复用。

---

## 六、一个工程判断

这次改动有一个决策值得记录：**什么时候该坚持自研格式，什么时候该跟标准走。**

原来的 `arguments` 格式不是错的，它在当时的需求范围内完全够用。它更简单，更易读，定义起来更快。

但它是私有的。私有格式的代价是：每个消费这份数据的地方都要了解这套格式，任何框架对接都需要先写适配层，任何扩展都需要修改解析代码。

JSON Schema + `x-*` 扩展字段不是我发明的。它是一个有完整规范、有大量工具支持、有明确演进路径的标准。选择标准的代价是短期多写几行代码，收益是长期的可接入性和可组合性。

对于 Skill 这种面向 LLM 和外部框架的结构，跟标准走是对的。

---

*本文记录的改动发生在 2026 年 4 月。核心文件：`src/skills/registry.py`、`src/agent.py`、`frontend/lib/skills.ts`、`frontend/components/chat/SkillPicker.tsx`。*

