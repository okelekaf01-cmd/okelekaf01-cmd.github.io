---
title: "从需求出发的技术发现：我与 AI 架构的三次重逢"
date: 2026-03-01T10:00:00+08:00
draft: false
top_img: /img/covers/blog9-technical-discoveries.svg
cover: /img/covers/blog9-technical-discoveries.svg
tags: ["OpsMind", "AI架构", "技术发现", "Agent", "Tool Calling"]
categories: ["技术文章"]
author: "wwxdsg"
description: "分享在OpsMind项目开发中三次技术重逢的经历：预设代码与Skill、JSON输出与Tool Calling、自主判断与Agent"
permalink: /opsmind/blog9_technical_discoveries/
---

## 引言

在开发 OpsMind 这个智能运营助手项目的过程中，我经历了几次奇妙的"技术重逢"——当我为解决具体问题而独立构思出某种方案后，却发现行业中早已存在相同理念的标准命名和成熟架构。这种体验让我对技术发展的规律有了更深的感悟。

## 第一次重逢：预设代码与 Skill

项目初期，我遇到了一个棘手的问题：用户上传数据文件后，系统需要根据不同的业务场景执行不同的分析流程。如果每次都让大模型从头理解需求、规划步骤，不仅效率低下，而且输出质量难以稳定。

于是我想到了一个办法：**预先定义好一套标准化的分析流程模板**。

```python
# 我的原始想法（简化示意）
ANALYSIS_TEMPLATES = {
    "sales_trend": [
        "load_data",
        "check_columns", 
        "generate_line_chart",
        "calculate_growth_rate",
        "write_insight"
    ],
    "hr_distribution": [
        "load_data",
        "detect_outliers",
        "generate_boxplot",
        "summarize_distribution"
    ]
}
```

当用户提问时，系统先识别意图，然后调用对应的模板，按步骤执行预设的代码逻辑。我把这个机制称为"预设代码"——听起来很朴素，但确实解决了问题。

直到后来，我在阅读 LangChain 和各大 AI 平台的文档时，看到了一个熟悉的词汇：**Skill**。

原来，这种"将特定能力封装成可复用模块"的思路，已经是 Agent 开发的标准范式。各大框架都在推行类似的机制：预定义技能、动态调用、组合执行。我的"预设代码"，不过是 Skill 概念的一次朴素重发明。

## 第二次重逢：JSON 输出与 Tool Calling

项目的意图识别模块需要将用户问题分类到多个维度：主类别（RAG/DATA）、业务领域（财务/人力/增长）、分析逻辑（趋势/对比/分布）。

最初，我尝试让大模型直接输出文本描述，然后用正则表达式解析。但这种方式太脆弱——模型输出稍有变化，解析就会失败。

我想到一个方案：**让大模型直接输出 JSON 格式**。

```python
# 我设计的 Prompt 片段
_SYSTEM_PROMPT = """
【输出格式】严格输出 JSON，不含任何注释或多余文字：
{"primary":"DATA","domain":"HR","logic":"DISTRIBUTION"}
"""
```

我在 Prompt 中明确定义了每个字段的取值范围，并要求模型"严格输出 JSON"。配合 `temperature=0.1` 和正则提取，这个方案运行得相当稳定。

后来我才知道，这种做法在行业中有一个正式的名字：**Structured Output**，而支撑它的底层机制是 **Tool Calling**（或 Function Calling）。

```python
# 行业标准写法
TOOLS = [{
    "type": "function",
    "function": {
        "name": "classify_intent",
        "parameters": {
            "type": "object",
            "properties": {
                "primary": {"type": "string", "enum": ["RAG", "DATA"]},
                "domain": {"type": "string"},
                "logic": {"type": "string"}
            }
        }
    }
}]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=TOOLS,
    tool_choice={"type": "function", "function": {"name": "classify_intent"}}
)
```

各大模型平台都原生支持这种结构化输出机制，不仅更可靠，还能与后续的工具调用无缝衔接。我的"JSON 输出"方案，本质上是在没有 Tool Calling 支持时的手动实现。

## 第三次重逢：自主判断与 Agent

项目演进到后期，我面临一个更复杂的挑战：用户的提问往往是开放式的，系统需要根据实际情况动态决定调用哪些工具、执行多少步骤、何时终止。

例如，用户问"帮我分析这份销售数据"，系统可能需要：

1. 先调用 `get_data_info` 了解数据结构
2. 发现某列是高基数分类，决定不直接用条形图
3. 调用 `analyze_data` 执行分析
4. 发现部分图表生成失败，决定重试或调整方案
5. 综合所有结果，生成最终回答

我设计了一个循环机制：让大模型在每一步都"思考"下一步该做什么，直到它认为任务完成。

```python
# 我的 ReAct 循环实现
for iteration in range(max_iterations):
    response = llm.call(messages, tools=AVAILABLE_TOOLS)
    
    if not response.tool_calls:
        # 模型认为任务完成，返回最终答案
        return response.content
    
    # 执行工具调用，将结果加入上下文
    for tool_call in response.tool_calls:
        result = execute_tool(tool_call)
        messages.append({"role": "tool", "content": result})
```

这个架构让系统具备了"自主判断"的能力——模型不再是被动执行固定流程，而是根据中间结果动态调整策略。

后来我了解到，这种架构有一个广为人知的名字：**ReAct Agent**（Reasoning + Acting）。从 AutoGPT 到 LangChain Agent，从 BabyAGI 到各大平台的 Assistant API，Agent 已经成为 AI 应用开发的核心范式。

我的"自主判断循环"，不过是 ReAct 模式的一次独立实现。

## 感悟：需求是技术之母

回顾这三次"重逢"，我有一个深刻的感悟：**这些概念本质上都源于实际需求**。

- **Skill** 源于"如何复用能力"的需求
- **Tool Calling** 源于"如何可靠地让模型输出结构化信息"的需求
- **Agent** 源于"如何让模型自主决策和执行"的需求

当我面对同样的需求时，我独立推导出了相似的解决方案。这让我意识到：**独立思考的价值不在于"第一个想到"，而在于"从真实问题出发"**。

有时候我会想：如果我早点接触大模型领域，是否能在这些概念被命名之前就提出它们？

但这个假设没有意义。更重要的是：**我从具体项目需求中独立发现了这些概念，这本身就是一种学习**。它让我对这些技术有了"从原理出发"的理解，而不是仅仅记住了一个名词。

## 结语

技术发展的规律往往是：**先有需求，后有概念，再有标准**。

当我们面对一个棘手问题时，不妨先独立思考解决方案，再去查阅行业实践。如果发现自己的方案与行业标准不谋而合，那是一种鼓励——说明我们的思维方向是正确的；如果发现自己的方案有缺陷或遗漏，那是一种学习——我们可以从成熟方案中汲取营养。

在 OpsMind 的开发过程中，这三次"重逢"让我对 AI 架构有了更深的理解。它们不再是教科书上的抽象概念，而是我曾经亲手触摸过的真实需求。

这，或许就是独立思考最大的价值。

---
