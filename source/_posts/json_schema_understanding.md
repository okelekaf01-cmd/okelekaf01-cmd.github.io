---
title: "从“让模型别乱说”开始：我对 JSON Schema 的理解"
date: 2026-03-22T20:00:00+08:00
draft: false
top_img: /img/covers/json-schema-understanding.svg
cover: /img/covers/json-schema-understanding.svg
tags: ["JSON Schema", "LLM", "Structured Output", "Tool Calling", "OpsMind"]
categories: ["技术文章"]
author: "wwxdsg"
description: "结合 OpsMind 项目的实践，记录我对 JSON Schema 的理解：它不只是给 JSON 写类型，更是在 LLM 应用里把“希望模型这样输出”变成可校验契约。"
permalink: /opsmind/json-schema-understanding/
---

![JSON Schema Cover](/img/covers/json-schema-understanding.svg)

最近在看项目里的意图分类链路时，我脑子里反复冒出来一句很朴素的话：

> 我们到底是怎么让 LLM 老老实实吐出标准字段的？

一开始我以为答案会很“AI”一点，比如 prompt 写得够好、示例给得够多、模型自己就会懂。后来发现，现实工程里并没有这么浪漫。真正让系统稳定下来的，通常不是“模型突然变听话”，而是**我们把输出要求写成了更硬的契约**。

这也是我重新理解 JSON Schema 的起点。

## JSON Schema 不是“给 JSON 写注释”

如果只从字面上看，JSON Schema 很容易被理解成一种“说明文档”：

- 这个字段是字符串
- 那个字段是数字
- 某几个字段必填

但如果只把它理解到这里，就低估它了。

在我现在的理解里，JSON Schema 更像是：

**把“我们希望返回什么”写成一份机器也能执行的契约。**

它不只是告诉人类开发者字段应该长什么样，还能直接参与校验，告诉程序：

- 这个输出合法不合法
- 缺了什么字段
- 多了什么字段
- 值是不是落在允许范围内

也就是说，它不是备注，而是规则。

## 为什么我会突然觉得它重要

因为项目里其实已经有一个很典型的现实场景：

我们并不是完全靠模型“自觉”输出标准 JSON，而是一直在做两层保护：

1. 提示词里要求“严格输出 JSON”
2. 代码里再用正则、`json.loads`、默认值和枚举校验兜底

这套做法当然能跑，而且很多项目一开始都是这么干的。但它有一个明显的问题：

**约束是分散的。**

一部分写在 prompt 里，一部分写在解析逻辑里，一部分藏在默认值里，一部分靠调用方自己脑补。

最后就会出现一种很常见的状态：

- 模型“差不多”按你想要的格式输出了
- 后端“差不多”把它解析出来了
- 前端“差不多”知道这个字段应该怎么消费

三个“差不多”叠在一起，系统就开始随机发脾气。

JSON Schema 对我最大的吸引力就在这里：  
它试图把这些零散的“差不多”，收束成一份统一的、可检查的定义。

## 一个最简单的例子

比如下面这种结构：

```json
{
  "name": "Alice",
  "age": 28,
  "role": "user"
}
```

如果不用 Schema，我们通常只会在心里默认：

- `name` 应该是字符串
- `age` 应该是整数
- `role` 应该只能是某几个值

但默认这件事本身是没有约束力的。  
只有当它被写成 Schema，它才从“大家都知道”变成“系统真的会检查”。

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["name", "age", "role"],
  "properties": {
    "name": { "type": "string", "minLength": 1 },
    "age": { "type": "integer", "minimum": 0, "maximum": 150 },
    "role": { "type": "string", "enum": ["admin", "user", "guest"] }
  }
}
```

这时候它表达的就不再是“建议”，而是：

- 你必须是对象
- 只能有这几个字段
- 这几个字段必须出现
- `role` 不能随便编

从工程角度看，这已经很接近接口契约了。

## 在 LLM 项目里，它到底解决什么问题

如果放回当前项目语境，JSON Schema 最有用的地方，其实不是“描述 JSON”，而是**收拾模型输出的不稳定性**。

### 1. 它让分类结果不再只是“看起来像 JSON”

比如意图分类这件事，本来我们只是希望模型返回：

```json
{
  "primary": "DATA",
  "domain": "HR",
  "logic": "DISTRIBUTION"
}
```

问题在于，“希望”不是约束。  
模型完全可能返回：

- 多一句解释
- 少一个字段
- 枚举值拼错
- 套一层 markdown code block
- 甚至加一个你根本没定义过的字段

而 Schema 的价值在于，它能把这件事说死：

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["primary", "domain", "logic"],
  "properties": {
    "primary": { "type": "string", "enum": ["RAG", "DATA"] },
    "domain": {
      "type": ["string", "null"],
      "enum": ["FINANCE", "GROWTH", "HR", "GOVT", null]
    },
    "logic": {
      "type": ["string", "null"],
      "enum": [
        "ACHIEVING", "COMPOSITION", "FLOW", "DISTRIBUTION",
        "TREND", "CORRELATION", "COMPARISON", "GEOSPATIAL",
        "QUALITY", "PROCESS", null
      ]
    }
  }
}
```

这时候“字段标准化”就不再只是依赖 prompt 写得够凶，而是真正有了边界。

### 2. 它让 tool calling 的参数更像函数签名

我后来越来越觉得，JSON Schema 和 tool calling 之所以搭得这么好，是因为它们本质上都在做同一件事：

**把自然语言调用，拉回到结构化函数调用。**

一个工具定义如果只有名字，没有参数约束，那模型还是有很大自由度。  
但一旦参数也由 schema 明确下来，它就更像一个真正的函数签名：

- 这个字段必填
- 这个字段必须是数字
- 这个字段只能传这几个值

这会让“调用工具”从一种带格式的文本游戏，变成真正的结构化接口调用。

### 3. 它让“后处理兜底”从救火变成补强

我现在最喜欢的一点，是 JSON Schema 并不会消灭后处理逻辑，反而会让后处理更合理。

因为一旦 schema 存在，后处理不再负责“猜模型本来想说什么”，而是负责：

- 校验
- 报错
- 默认值补齐
- 降级策略

这两者差别很大。

前者是翻译玄学，后者是执行规则。

## 我现在对它的一个核心理解

如果要用一句最简单的话概括我现在的理解，我会说：

**JSON Schema 的意义，不是让 JSON 更规范，而是让“不确定的输出”拥有一个可以被程序严格讨论的边界。**

这件事在普通后端接口里本来就重要，在 LLM 应用里则几乎是刚需。

因为 LLM 最大的魅力是灵活，最大的风险也是灵活。  
而 JSON Schema 的存在，就是在灵活和稳定之间补上一道工程护栏。

## 它不是银弹，但它能显著减少“软约束”

当然，实话也得说：JSON Schema 并不自动等于 100% 强约束。

如果模型接口本身不支持真正的 schema-level structured output，那它最终还是可能出现：

- 看起来像合法 JSON，实际字段不对
- 值类型对了，但语义错了
- 结构没问题，但内容胡说

所以 Schema 解决的不是“真伪问题”，而是**格式与结构问题**。

它让我们至少不用再把大量精力浪费在：

- 提取第一个 `{...}`
- 猜多余解释该不该删
- 担心字段名今天是 `domain` 明天变成 `area`

也就是说，它不能替代业务判断，但能帮我们先把地板铺平。

## 我觉得最容易被忽略的几个关键词

如果只记几个词，我会优先记这几个：

- `type`
  决定你到底在处理对象、数组还是字符串
- `required`
  告诉你哪些字段不是可有可无
- `enum`
  非常适合约束分类、状态、模式切换这类字段
- `additionalProperties: false`
  这个特别重要，它的作用几乎等于“别给我自由发挥新字段”
- `items`
  用来定义数组里每个元素应该长什么样
- `oneOf / anyOf / allOf`
  适合更复杂的分支结构，但也会让 schema 复杂度迅速上升

如果是 LLM 项目，我甚至会说 `additionalProperties: false` 的重要性经常被低估。  
因为模型最喜欢做的事之一，就是在你没问的时候多送你一点字段。

## 如果让我给项目提一个很实际的改进方向

我会优先把下面三类东西做成 schema：

### 1. 意图分类结果

这是最直接、收益也最高的一层。

### 2. Tool result 结构

像：

```json
{
  "success": true,
  "charts_generated": 2,
  "tables_generated": 1,
  "error": null
}
```

这种结构非常适合 schema 化，因为它天然是后端代码控制的。

### 3. SSE event payload

哪怕不是所有事件都完全 schema 化，至少 `intent`、`tool`、`done` 这类关键事件值得先规范起来。

因为一旦这些结构稳定，前端状态机会明显轻松很多。

## 结语

以前我对 JSON Schema 的理解比较偏“数据格式校验工具”。  
现在我更愿意把它理解成：

**一种把自然语言系统重新拉回工程世界的办法。**

它的价值不在于语法本身，而在于它提醒我们：  
当系统里开始出现越来越多“不完全可预测”的输出时，真正重要的不是继续寄希望于模型乖一点，而是尽快把那些关键边界写成机器也能执行的规则。

对我来说，这就是 JSON Schema 最迷人的地方。

它不是让模型更聪明。  
它只是让系统更少靠运气。
