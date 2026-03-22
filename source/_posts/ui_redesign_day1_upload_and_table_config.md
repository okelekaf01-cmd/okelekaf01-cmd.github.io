---
title: "OpsMind UI 重构实录（一）：先把上传链路和宽表配置打通"
date: 2026-03-20T10:00:00+08:00
draft: false
top_img: /img/covers/ui-redesign-day1-upload-flow.svg
cover: /img/covers/ui-redesign-day1-upload-flow.svg
tags: ["OpsMind", "UI重构", "文件上传", "数据预处理", "前后端联调"]
categories: ["开发日志"]
author: "wwxdsg"
description: "记录这次 OpsMind UI 重构第一天最重要的工作：先修掉上传 500，再把宽表检测、Melt 配置和前端预览链路真正接起来。"
permalink: /opsmind/ui-redesign-day1-upload-flow/
---

![OpsMind UI Redesign Day 1 Cover](/img/covers/ui-redesign-day1-upload-flow.svg)

这次 UI 重构我最后一共做了三天。现在回头看，第一天其实没什么“高级设计感”，更多是在做一件很工程化、但也很必要的事：

**先把主链路打通。**

因为如果上传文件就 500，宽表识别不出来，前端也没有地方承接用户的配置，那后面所有关于质感、主题、排版的优化都会显得有点奢侈。

所以第一天的重点，不是把界面做漂亮，而是把“上传一个表，然后让系统真的知道该怎么处理它”这条链路做完整。

## 先修一个很不体面的 500

最先碰到的问题其实挺朴素：用户上传带年份列的 CSV，比如 `2009`、`2010` 这种表头，后端直接返回 500，而且前端拿不到有用信息。

这种问题最烦的地方不只是报错，而是**报得很没礼貌**。  
你知道它错了，但它不打算告诉你为什么。

我做的第一件事，是在 `src/api/main.py` 里临时加一个全局异常处理器，把完整 traceback 透出来。开发期这一步很值，因为它能把“纯猜”变成“可定位”。

很快就发现根因不在上传本身，而在返回 preview 的时候：

```python
df.to_dict(orient="records")
```

当 pandas 读到年份列时，列名会是整数；但 Pydantic 这边定义的是 `Dict[str, Any]`，它只接受字符串键。于是本来只是个预览数据，最后被整数 key 拖进了 ValidationError。

修法很简单：

```python
preview_df.columns = preview_df.columns.astype(str)
```

这类问题特别适合记进脑子里，因为它不是业务逻辑错了，而是**两个正常库在接口边界上没对齐**。  
以后凡是 `DataFrame.to_dict()` 后面还要进 Pydantic，我基本都会先想到这一层。

## 上传不是终点，结构识别才是

把 500 修掉之后，事情才刚刚开始。

因为这次改动的真正目标，不是“能上传文件”，而是让系统在看到一张宽表的时候，知道它需要额外处理。

所以后端又补了一层 `StructureInfo`，专门描述这张表的结构特征：

- 是否需要人工注意
- 它更像 `wide_year`、`wide_date` 还是普通表
- 哪些列像年份列
- 哪些列像文本维度
- 有没有空列、匿名列
- 以及一份建议配置

这一步我很喜欢，因为它让上传接口从“单纯接收文件”升级成了“对数据做第一轮判断”。

也就是说，文件一传上来，系统不只是说“我收到了”，而是在说：

> 我大概知道这张表长什么样了，接下来你可能需要决定怎么展开它。

这就是体验的分水岭。  
前一种更像存储接口，后一种才开始有点分析产品的味道。

## 宽表转换这件事，最好让用户先看见

光有结构检测还不够。真正让我觉得这条链路“活了”的，是前端那块配置面板。

这次做法不是把所有转换逻辑都扔给后端，而是前端先自己做一份 `meltPreview()`，直接在本地把表预演一遍。

```typescript
function meltPreview(
  data: Record<string, unknown>[],
  idVars: string[],
  valueVars: string[],
  varName: string,
  valueName: string,
): Record<string, unknown>[] {
  const result: Record<string, unknown>[] = []
  for (const row of data) {
    for (const vv of valueVars) {
      const newRow: Record<string, unknown> = {}
      for (const iv of idVars) newRow[iv] = row[iv]
      newRow[varName] = vv
      newRow[valueName] = row[vv]
      result.push(newRow)
    }
  }
  return result
}
```

这样做的好处非常直接：

- 用户点列名切换角色时，预览能立刻变化
- 不用每改一次配置就打一次后端
- “我到底在把这张表变成什么”这件事变得可见了

我后来越来越喜欢这种设计：  
**后端负责给出判断和持久化，前端负责把决策过程做得即时、透明、低摩擦。**

## 列角色系统，是这一天最像产品设计的一笔

为了让配置过程尽量轻，我没有做一堆复杂表单，而是给每一列定义了三个角色：

- `dim`
- `metric`
- `exclude`

点击一次就循环切换：

```typescript
function nextRole(r: ColRole): ColRole {
  return r === 'dim' ? 'metric' : r === 'metric' ? 'exclude' : 'dim'
}
```

这其实是个很小的交互，但我很喜欢。因为它把原本容易做得很重的一件事，压缩成了一个非常直接的操作模型：

- 这是维度
- 这是指标
- 这个不要

没有额外表单，没有复杂弹窗，没有让人读半天的设置项。  
用户只需要一边点，一边看预览怎么变。

这种感觉很像：系统不是在逼你“填写配置”，而是在和你一起把表格整理成它能理解的形状。

## 最后再把配置落回 session

等用户确认完之后，前端再调：

```http
PATCH /api/sessions/{session_id}/table-config
```

把这份 `table_config` 存进 session metadata。之后每次工作流执行前，再由后端把这个配置注入给 `data_engine`。

这一点也很关键。因为它意味着配置不是一次性的 UI 状态，而是会话语义的一部分。

也就是说，系统会记得：

- 这张表原来是宽表
- 用户上次决定怎么展开它
- 后续分析都应该沿用这份理解

这就比“用户每次上传都重新解释一遍”自然得多。

## 第一天做完后，我真正放心的不是页面，而是链路

如果只看视觉，这一天其实还谈不上惊艳。  
但它完成了一件更底层的事：把上传、识别、预览、确认、持久化这条线串起来了。

回头想，这一天的意义有点像在搭地基：

- 修掉一个不体面的 500
- 让后端对表结构有判断能力
- 让前端把转换过程可视化
- 让最终配置进入 session，而不是停留在瞬时状态

后面两天能开始认真谈设计 token、双主题、消息区排版、Markdown 渲染器这些“我很喜欢”的内容，某种意义上也是因为第一天先把产品最核心的路径扶正了。

所以如果要给这一天下个定义，我会说：

**它不是最漂亮的一天，但它把整个 UI 重构从“样式升级”拉回了“产品真正变完整”。**
