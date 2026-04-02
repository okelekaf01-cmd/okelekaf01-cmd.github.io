---
title: "从 Chat 模型陷阱到真正的 Agent 架构：上下文污染与混合模型的工程实践"
date: 2026-04-02T22:00:00+08:00
draft: false
top_img: /img/covers/chat-model-trap-agent-architecture.svg
cover: /img/covers/chat-model-trap-agent-architecture.svg
tags: ["OpsMind", "Agent", "架构设计", "上下文管理", "混合模型", "LLM工程"]
categories: ["技术文章"]
author: "wwxdsg"
description: "这次改动的重点不是再加功能，而是把 Agent 主流程做对：角色分离、逐轮压缩、上下文治理，解决 ReAct 循环中的上下文污染。"
permalink: /opsmind/chat-model-trap-agent-architecture/
---

![Chat Trap to Agent Architecture Cover](/img/covers/chat-model-trap-agent-architecture.svg)

> 这篇文章不是在聊某一次优化，而是在聊一个认知转变：从“疯狂堆功能”回到“把主流程做对”，以及我为什么认为这才是一个产品真正成熟的标志。

---

## 一、被 Chat 模型骗了很久

我用的是 DeepSeek，OpenAI 兼容接口，调用方式极其简单。

简单本身是一个陷阱。

当你只需要 `client.chat.completions.create(model="deepseek-chat", messages=messages)` 就能让系统“跑起来”，你很容易把这个模型当成一个万能胶：路由分类用它，Agent 规划用它，工具失败反思用它，多步结果综合也用它。

它确实都能做。每一步单独测试，效果不差。

但当你把这些步骤真正串成一个 ReAct 循环，问题就开始出现了：

- 第 3 轮迭代，模型开始重复调用相同的工具
- 第 4 轮，它明明有了结论，却还在向前推
- 第 5 轮，它在回答里开始出现上一轮失败工具的幻觉

这不是模型本身的问题。这是**架构设计的问题**：同一个 Chat 模型被同时用于规划、执行调度和结论生成，没有任何角色分离，没有上下文管理，只有越来越脏的 `messages` 数组。

---

## 二、真正的问题：上下文污染

在 OpsMind 的 Agent 实现里，每轮工具调用结束后，结果会原文追加进 `messages`：

```python
messages.append({
    "role": "tool",
    "tool_call_id": tc.id,
    "content": tool_result
})
```

`plan_analysis` 会返回完整的 `chart_plan` 和 `table_plan`，包含所有图表的列映射配置。  
`execute_analysis` 会返回洞察文本、图表列表、清洗报告。  
`get_data_info` 会返回所有列的统计信息。

到第 3 轮迭代，`messages` 数组里已经有 3 条完整 JSON 工具结果，可能超过 8000 tokens。而模型在生成下一步决策时，需要从这堆原始结构化数据里自己筛选有用信息。

这不是 Chat 模型擅长的事情。

更糟糕的是，最初实现的“压缩”只在**超过最大迭代次数之后**才触发。这等于说：模型在整个有效执行期间，都在一个越来越脏的上下文里工作；只有在它已经失控之后，我们才开始清理。

这是一个典型的**被动防御**设计：等问题暴露再处理，而不是在问题出现之前主动管理。

---

## 三、去看别人怎么做的

有个机会让我去看了另一个成熟 Agent 系统（Claude Code）的源码。

它处理上下文污染的方式，不是一个补丁，而是一条有序的**5 层压缩流水线**，在每次 LLM 调用前依序执行：

```text
1. applyToolResultBudget()    -> 单条 tool result 超 50KB -> 写磁盘，消息里只留 2KB 预览
2. snipCompactIfNeeded()      -> 历史片段裁剪
3. microcompactMessages()     -> 距上次消息 >1小时 -> 清除过期 tool result
4. applyCollapsesIfNeeded()   -> 上下文折叠
5. autocompact()              -> 整体超限时 -> 用轻量模型对旧消息做摘要，替换原始内容
```

关键不是每一层的具体实现，而是整体设计思路：**压缩是主动的、逐轮的，不是被动的、超限才触发的。**

它在每次把 `messages` 送给 LLM 之前，都会清理一遍。不是因为快爆了才清，而是因为干净的上下文是好决策的前提。

这是一种完全不同的认知：**上下文管理不是兜底机制，而是 Agent 执行的基础设施。**

---

## 四、我做了什么改动

理解了差距之后，我对 `agent.py` 做了三件事。

### 1. 角色分离：推理模型 + 快速模型

原来两次 LLM 调用全部用 `deepseek-chat`：

```python
response = self._get_client().chat.completions.create(
    model="deepseek-chat", ...
)

final_response = self._get_client().chat.completions.create(
    model="deepseek-chat", ...
)
```

现在：

```python
self._plan_model = os.getenv("AGENT_PLAN_MODEL", "deepseek-reasoner")
self._fast_model = os.getenv("AGENT_FAST_MODEL", "deepseek-chat")

response = self._get_client().chat.completions.create(
    model=self._plan_model, ...
)

final_response = self._get_client().chat.completions.create(
    model=self._fast_model, ...
)
```

规划和文案生成是两种不同的认知任务。  
前者需要推理能力，后者需要表达效率。混用会让两端都不经济。

### 2. 每轮主动压缩：`_prepare_messages_for_llm()`

新增了一个方法，在每次 LLM 调用前执行：

```python
@staticmethod
def _prepare_messages_for_llm(messages: List[Dict]) -> List[Dict]:
    COMPRESS_THRESHOLD = 1500
    KEEP_RECENT = 2

    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    compress_set = set(tool_indices[:-KEEP_RECENT]) if len(tool_indices) > KEEP_RECENT else set()

    result = []
    for i, msg in enumerate(messages):
        if i in compress_set and len(msg.get("content", "")) > COMPRESS_THRESHOLD:
            compressed = OpsMindAgent._compress_old_tool_result(msg["content"])
            result.append({**msg, "content": compressed})
        else:
            result.append(msg)
    return result
```

策略很直接：最近 2 条工具结果保留原文，更早且过长的结果压缩。

### 3. 精准压缩：`_compress_old_tool_result()`

压缩不是粗暴截断，而是字段提取：

```python
@staticmethod
def _compress_old_tool_result(content: str) -> str:
    data = json.loads(content)

    keep_keys = {
        "success", "status", "error",
        "rows", "columns", "data_shape",
        "ok_charts", "skip_charts", "ok_tables",
        "charts_generated", "failed_charts", "table_types",
        "mode", "filename", "truncated",
    }
    summary = {k: v for k, v in data.items() if k in keep_keys}

    for text_key in ("insight_summary", "answer"):
        if text_key in data:
            val = str(data[text_key])
            summary[text_key] = val[:200] + "…" if len(val) > 200 else val

    summary["_compressed"] = True
    return json.dumps(summary, ensure_ascii=False)
```

保留决策关键字段，删除高冗余体积内容，兼顾可读性与 token 成本。

---

## 五、从“疯狂加功能”到“把主流程做对”

回头看这半年，OpsMind 一直在加能力：知识库、分析链路、图表推荐、宽表处理、DB 直连、报告生成、Skill 工具……

每加一层功能，演示效果都更好。  
但我也在这个过程中欠下一笔债：主流程稳定性在下降，因为地基没有同步加固。

这次停下来还债，让我更确定一件事：

**功能堆叠解决的是“能不能做”，架构优化解决的是“能不能一直做好”。**

前者适合早期验证，后者决定长期可维护性。  
一个工程系统真正成熟，往往不是功能最多，而是关键路径最稳。

---

## 六、任务分配与架构设计

这次改动背后还有一个更深的认知：**模型分工就是任务分配，任务分配就是架构设计。**

把所有工作都交给同一个 Chat 模型，和把所有工作都交给同一个角色，本质是同一个问题：职责失焦。

推理模型适合处理：
- 状态判断
- 路径规划
- 失败反思
- 多步协调

快速模型适合处理：
- 文本生成
- 格式化输出
- 简单分类
- 短回复

分开之后，每个模型都在自己擅长的位置工作；再配合逐轮压缩，ReAct 循环才真正从“凭运气”切到“可控执行”。

这次代码改动不算大，但背后的认知转弯很重要。

---

> 好的产品不是功能最多的产品。  
> 好的 Agent 不是工具最多的 Agent。  
> 好的架构不是最复杂的架构。  
>  
> 它们都是在对的时间做了对的事情。

