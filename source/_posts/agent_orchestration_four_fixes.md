---
title: "Agent 编排四项修复：从结构性问题到工程落地"
date: 2026-04-02T21:00:00+08:00
draft: false
top_img: /img/covers/agent-orchestration-four-fixes.svg
cover: /img/covers/agent-orchestration-four-fixes.svg
tags: ["OpsMind", "Agent", "工程化", "架构设计", "可靠性", "SSE"]
categories: ["技术文章"]
author: "wwxdsg"
description: "这次修的不是零散 bug，而是 Agent 编排层里四个结构性问题：迭代上限、plan 透传、局部图表修订路径、SSE 长连接保活。"
permalink: /opsmind/agent-orchestration-four-fixes/
---

![Agent Orchestration Fixes Cover](/img/covers/agent-orchestration-four-fixes.svg)

这篇记录的不是“又修了几个点”，而是一次编排层审查之后，把系统里几处“靠运气撑住”的位置，逐一改成工程性保障的过程。

系统在大部分演示场景里一直能跑通，但只要把视角从“能跑”切到“复杂场景能稳定复现”，问题就会非常清楚：  
很多失败不是功能缺失，而是结构设计在边界条件下不够稳。

这次我落地了四项修复。

---

## 一、修复 1：`max_iterations=6` 在数据库路径上是零容错

之前 ReAct Agent 的上限是 `max_iterations = 6`。  
这个值在简单路径可用，但数据库分析的标准链路是：

```text
get_db_schema -> execute_sql -> get_data_info -> plan_analysis -> execute_analysis
```

5 次工具调用刚好贴着上限走，任意一步重试都会触发迭代截止。  
一旦触发上限，Agent 会进入压缩输出，导致最终回答只有结论，没有可靠图表与分析细节。

### 调整

1. 上限从 `6` 提到 `10`，给复杂路径和一次重试留空间。  
2. 保留总超时预算（`AGENT_TOTAL_TIMEOUT_SECONDS`）作为真正的安全网。  
3. 在 system prompt 中明确：`execute_sql` 成功后可直接进入 `plan_analysis`，避免不必要的 `get_data_info` 兜圈。

这一步做完后，数据库分析路径从“勉强刚好”变成“可容错可恢复”。

---

## 二、修复 2：取消 `chart_plan` 的 LLM 透传，改为 `plan_id`

原先流程是 `plan_analysis` 产出完整 `chart_plan/table_plan`，再由 LLM 原样传给 `execute_analysis`。  
问题在于：这等于把复杂 JSON 的完整性寄托在 LLM 的序列化行为上。

常见失真包括：

- 字段被精简（例如丢掉 `status`）
- 键名被改写（`x_col` 变成 `x_axis`）
- 嵌套 JSON 二次转义

这些都不是语义层 bug，而是“传输层不可靠”。

### 调整

`plan_analysis` 只返回一个短 `plan_id`，完整 plan 存在服务端内存：

```python
plan_id = uuid.uuid4().hex[:8]
self._session_plans[plan_id] = {
    "chart_plan": result.get("chart_plan", {}),
    "table_plan": result.get("table_plan", {}),
    "file_path": file_path,
}
```

`execute_analysis` 只收 `plan_id`，在服务端取回原始 plan：

```python
stored = self._session_plans.get(plan_id)
chart_plan = stored["chart_plan"]
table_plan = stored["table_plan"]
```

这一步的本质是：把“跨步骤数据完整性”从模型层搬回服务端强约束层。

---

## 三、修复 3：新增 `refine_chart`，让图表调整从“全量重跑”改为“局部修订”

过去用户说“把第一张图改成折线图”，系统会重新 `plan_analysis + execute_analysis` 全链路重跑。  
这对单图修改来说成本太高，也容易引入无关波动。

### 调整

新增工具：

```text
refine_chart(plan_id, chart_type, new_chart_type?, new_column_mapping?)
```

能力边界：

- 局部改图类型
- 局部改列映射
- 原地更新同一个 `plan_id`
- 再调用 `execute_analysis` 增量生效

收益很直接：  
“改一张图”不再触发一次完整重规划，延迟和 token 成本都显著下降。

---

## 四、修复 4：SSE 增加心跳帧，解决长分析链路下的空闲断连

分析链路超过 30s 时，如果 SSE 长时间无帧，代理层或浏览器会按空闲连接处理并主动断开。  
本地开发不明显，上线后在反向代理链路中会稳定复现。

### 调整

在事件生成器里使用“超时等待 + 注释心跳帧”：

```python
_HEARTBEAT_INTERVAL = 15.0
while True:
    try:
        item = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_INTERVAL)
    except asyncio.TimeoutError:
        yield ": heartbeat\n\n"
        continue
```

`:` 开头是 SSE 注释帧，不会触发前端业务消息处理，但能持续保活连接。

---

## 五、改动文件一览

| 文件 | 关键改动 |
|---|---|
| `src/agent.py` | `max_iterations` 调整；`_session_plans` 引入；`plan_id` 传递；`refine_chart` 工具与 handler；prompt 更新 |
| `src/api/routers/chat.py` | SSE generator 改为带超时等待；超时发送 heartbeat 注释帧 |

---

## 六、这四项修复背后的共同方向

它们表面上是四个点，底层其实是同一件事：

把“靠模型自觉、靠路径刚好、靠网络运气”的环节，改成“可验证、可恢复、可预期”的工程路径。

对 Agent 产品来说，最难的往往不是“让它做出结果”，而是“在真实复杂场景下持续做对结果”。  
这次四项修复，本质上就是朝这个方向补齐地基。

