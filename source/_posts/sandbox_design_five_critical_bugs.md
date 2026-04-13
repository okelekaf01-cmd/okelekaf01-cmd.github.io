---
title: "沙箱方案设计与五个关键 Bug：一次把复杂执行链路做顺的诊断实录"
date: 2026-04-14T21:40:00+08:00
draft: false
top_img: /img/covers/sandbox-design-five-critical-bugs.svg
cover: /img/covers/sandbox-design-five-critical-bugs.svg
tags: ["OpsMind", "Bug定位", "沙箱", "架构设计", "代码执行", "工程化"]
categories: ["技术文章"]
author: "wwxdsg"
description: "这篇文章记录我在 OpsMind 里落地代码沙箱时遇到的五个关键 Bug：它们分别卡在文件系统、接口契约、React 并发更新、SSE 事件语义和路由策略边界上。"
permalink: /opsmind/sandbox-design-five-critical-bugs/
---

![Sandbox Design Five Critical Bugs Cover](/img/covers/sandbox-design-five-critical-bugs.svg)

这篇文章想记录的，不只是“我接了一套代码沙箱”。

我更想写清楚另一件事：

**当一条分析链路开始同时跨越 LLM、执行环境、宿主机渲染、SSE 推流和前端状态管理时，复杂 Bug 往往不死在某一行代码里，而是死在边界上。**

这次在 OpsMind 里做沙箱方案落地，最后真正让我成长的，不是把 `run_code` 工具接上，而是把五个卡在不同边界层的关键问题一层层拆开、定位、修掉。

---

## 一、为什么原来的方案迟早会卡住

OpsMind 最早的数据分析链路，大致是这样：

```text
用户提问
  -> LLM 判断图表类型
  -> LLM 猜列映射
  -> 预设渲染器执行
  -> LLM 根据结果写结论
```

这套方案在简单问题上不是不能用，但它有一个很难绕开的结构性限制：

**LLM 负责猜，Python 负责跑，而“猜”和“跑”之间没有真实反馈回路。**

模型在决定图表类型和列映射时，并不知道：

- 某一列是不是高基数，根本不适合饼图
- 某一列缺失率过高，直接计算比例会失真
- 某个日期字段根本不是标准时间格式
- 用户想看的指标，原始数据里其实不存在，需要先计算

也就是说，原来的系统始终在让模型“预猜执行现场”。

你可以不断加列信息、加规则、加 prompt，但只要模型看不到真实执行结果，它就只能越来越努力地猜，而不可能真正从运行结果里修正自己。

这也是我后来决定切到沙箱方案的根本原因：

**复杂数据分析的问题，重点不是让模型更会猜，而是让模型有机会先算、再看、再改。**

---

## 二、我最后怎么收敛这套沙箱架构

我后来没有让 LLM 在沙箱里直接写 Plotly，也没有把整个渲染流程完全交给它。

最后落下来的，是一套更克制的分层：

### 沙箱内

- 用 pandas / numpy 做自由计算
- 通过 `declare_chart()` 声明要画什么图
- 通过 `declare_table()` 声明要展示什么表
- 用 `print()` 输出洞察和说明

### 宿主机

- 读取 `render_spec.json`
- 调用现有 `InteractiveChartGenerator`
- 统一渲染成 Plotly HTML
- 返回给前端

这套架构我后来把它理解成：

**LLM 负责数据计算，宿主机负责稳定渲染。**

换句话说：

- 开放给 LLM 的，是“怎么算数据”
- 不开放给 LLM 的，是“怎么稳定落成图表产物”

这样做有两个直接好处。

第一，模型不需要熟悉复杂的前端图表 API，它只需要把真正有价值的分析计算做出来。  
第二，最终产物仍然走宿主机这条可控渲染链，质量、样式和兼容性不会被沙箱代码拉散。

每次执行前，我会在用户代码前注入一段 preamble，预先放好：

- 数据读取入口
- `declare_chart`
- `declare_table`
- spec 写入逻辑
- 退出时落盘的 `atexit`

真正重要的不是 preamble 写了多长，而是它把沙箱代码和宿主机之间的契约固定住了。

---

## 三、五个关键 Bug 是怎么被拆开的

这次最有价值的部分，不是“终于能跑”，而是这五个问题让我越来越确定：复杂 Bug 的诊断，核心永远是先找对边界。

---

### Bug 1：`Permission denied: '.'`

**现象**

沙箱日志里显示执行成功，图表数量也正常，但前端一个图都看不到。宿主机日志报：

```text
PermissionError: [Errno 13] Permission denied: '.'
```

**根因**

问题出在沙箱产物收集阶段：

```python
for filename in file_list:
    src_path = output_dir / filename
    if not src_path.exists():
        continue
    shutil.copy2(src_path, charts_path / filename)
```

当 `filename` 是空字符串时，`output_dir / ""` 实际会被解释成当前目录 `.`。  
而 `Path(".").exists()` 会返回 `True`，于是系统继续往下执行，最后尝试把目录当成文件去复制。

**修复**

把判断从 `exists()` 改成 `is_file()`：

```python
if not src_path.is_file():
    continue
```

**教训**

文件系统问题经常不是“路径存不存在”，而是“这个路径到底是不是你以为的类型”。  
在文件复制、上传、收集这类链路里，`exists()` 往往是不够的，应该优先用 `is_file()` / `is_dir()` 这种带语义的判断。

---

### Bug 2：`create_table_groupby() got unexpected keyword argument 'title'`

**现象**

沙箱执行已经成功，但宿主机渲染 spec 时抛出：

```text
TypeError: create_table_groupby() got unexpected keyword argument 'title'
```

**根因**

Spec-Driven 的执行方式是把 spec 里的配置整体解包给渲染器：

```python
fig = create_fn(df, **config)
```

但 `create_table_groupby()` 和部分 `create_table_*()` 方法的签名里并没有 `title` 参数。  
这意味着：沙箱侧和宿主机侧虽然都“支持 table spec”，但它们理解的字段契约并不一致。

**修复**

给相关方法补上统一参数签名：

```python
def create_table_groupby(self, df, group_col, value_cols, agg_funcs=None, title=None):
    ...
```

**教训**

这种问题表面上像一个 Python `TypeError`，本质上却是接口契约漂移。  
只要系统是“上游生成 spec，下游消费 spec”，那所有字段都不再是某一侧的内部实现细节，而是跨模块契约。

---

### Bug 3：前端重复提交，Agent 被连续启动三次

**现象**

同一个发送动作，后端日志里却出现了三段并行启动的 Agent 流程。前端看起来像“点了一次”，系统实际发了三次。

**根因**

原来的前端提交守卫读的是 React state：

```typescript
if (streaming !== null && !streaming.error) return
```

但 React 18 的 batched updates 决定了：`setState` 不会在当前执行路径里立刻反映到闭包读取值上。  
第一次提交后，`streaming` 在当前批次里仍然可能是旧值 `null`，于是后续极短时间内的再次触发仍会穿透守卫。

**修复**

改成读取同步更新的 ref：

```typescript
if (streamingRef.current !== null && !streamingRef.current.error) return
```

并同步清理掉 `handleSend` 里对 `streaming` 的依赖。

**教训**

React state 适合驱动渲染，不适合做毫秒级竞态防护。  
任何要求“当前调用立刻可见”的保护状态，都应该优先用 ref，而不是依赖 state 何时回流到组件闭包。

---

### Bug 4：前端只显示最后一批图表

**现象**

多轮 `run_code` 明明产出了多批图表，但前端最后只剩最后一轮的结果。

**根因**

后端通过 SSE 连续发送多个 `charts` 事件，而前端 reducer 把它当成“全量替换”处理：

```typescript
case 'charts':
  return { ...prev, charts: (msg.data.files as ChartItem[]) ?? prev.charts }
```

这意味着第二轮结果一到，第一轮结果就被整体覆盖掉了。

**修复**

把事件语义改成增量追加：

```typescript
case 'charts': {
  const newCharts = (msg.data.files as ChartItem[]) ?? []
  return { ...prev, charts: [...prev.charts, ...newCharts] }
}
```

tables 同理。

**教训**

很多前后端问题看起来像“显示错了”，其实真正错的是事件语义根本没有讲清楚。  
一个 SSE 事件到底代表“当前全量状态”还是“本轮新增产物”，必须在设计阶段就定死，不然只要链路进入多轮执行，问题迟早暴露。

---

### Bug 5：LLM 始终没有机会调用 `run_code`

**现象**

明明已经接好了深度分析工具，但复杂请求还是稳定走老的 `plan_analysis -> execute_analysis` 路径，LLM 从头到尾没机会进入真正的代码执行。

**根因**

问题不在 prompt，也不在工具表本身，而在更早的路由器。

当请求没有命中知识库、数据库、连续对话这些关键词时，系统默认直接走：

```python
return "direct_analysis"
```

而 `direct_analysis` 是一条固定链路，根本不会进入 ReAct 循环。  
这意味着虽然 system prompt 写着“你可以调用 `run_code`”，但模型在执行层面从来没拿到过这个决策权。

**修复**

我最后没有继续修补规则分类器，而是把决策权显式还给用户：

- 快速模式：强制走 `direct_analysis`
- 深度模式：直接进入 ReAct，让 LLM 自主决定是否调用 `run_code`

也就是说，不再让系统偷偷替用户做“你这题适不适合深度执行”的隐式判断，而是把两种执行范式公开成一个可感知的开关。

**教训**

“规则路由优先” 和 “LLM 自主决策优先” 是两种不同的架构范式。  
如果你真的希望模型自主决策，就不能在它前面先把路线封死。

---

## 四、这五个 Bug 其实共同暴露了什么

回头看，这五个问题分布在完全不同的层：

- 文件系统
- 渲染接口
- React 并发更新
- SSE 事件协议
- 执行路由

但它们本质上都在提醒同一件事：

**复杂 Bug 很少是单点错误，它更常见的形态是：边界语义没有被定义清楚。**

具体来说，就是这五类边界：

### 1. 文件边界

你以为自己拿到的是文件名，实际可能拿到的是空路径或目录引用。

### 2. 契约边界

你以为上下游都在说“同一份 spec”，实际上它们对字段支持并不一致。

### 3. 状态边界

你以为前端守卫已经生效，实际上当前调用看到的还是旧 state。

### 4. 事件边界

你以为后端在推“新的图表”，前端却把它解释成“新的全量结果”。

### 5. 决策边界

你以为模型拥有自主权，但真正的分流决定已经在它前面被规则路由做完了。

这也是为什么我现在越来越倾向于这样看复杂排障：

**先别急着问是哪一行错了，先问这条链路的边界语义有没有被讲清楚。**

---

## 五、最终落下来的执行形态

现在这套链路，我最后把它收成了两条明确路径：

### 快速模式

```text
get_data_info
  -> plan_analysis
  -> execute_analysis
```

适合标准分析问题，成本低，结果稳定。

### 深度模式

```text
ReAct loop
  -> get_data_info
  -> run_code
  -> plan_analysis / execute_analysis（必要时回退）
```

适合需要自定义计算、复杂派生指标或标准图表路径不够表达的问题。

真正关键的是，这两条路径现在不再互相假装自己是同一种东西。

快速模式就是规则更强、产物更稳的标准链路。  
深度模式就是允许模型在真实执行结果里逐步试错、修正和扩展的探索链路。

只要路径边界清楚，很多原来看起来很“玄”的问题，都会突然变得可以诊断。

---

## 结语

这篇文章表面上是在写沙箱方案，实际上更像是在写一次复杂系统排障练习。

我后来越来越相信，复杂工程里的很多成长，并不来自“又多做了一个功能”，而来自你能不能把一个原本交织在一起的问题，真的拆开。

拆到最后，你会发现很多事情都没有那么玄：

- 图表不显示，也许不是图表的问题，而是文件语义问题
- 重复提交，也许不是按钮的问题，而是状态可见性问题
- LLM 不调用工具，也许不是 prompt 的问题，而是路由权根本没给它

对我来说，这类记录最值得留下的地方也在这里。

它不是只证明“修好了”，而是尽量把一次复杂 Bug 的形成路径、定位方法和修复原则，真正沉淀成下一次还能复用的工程经验。
