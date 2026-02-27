# 深入解析 OpsMind 三维意图识别系统

**发布日期**: 2026-02-27  
**作者**: wwxdsg 
**标签**: #意图识别 #LLM应用 #自然语言处理 #智能运营

---

## 目录

1. [背景与动机](#一背景与动机)
2. [系统架构概览](#二系统架构概览)
3. [三维分类体系详解](#三维分类体系详解)
4. [LLM 在意图识别中的核心作用](#四llm-在意图识别中的核心作用)
5. [技术实现细节](#五技术实现细节)
6. [组件协同工作机制](#六组件协同工作机制)
7. [应用场景示例](#七应用场景示例)
8. [性能优化与最佳实践](#八性能优化与最佳实践)
9. [未来展望](#九未来展望)

---

## 一、背景与动机

### 1.1 问题背景

在企业智能运营场景中，用户查询种类繁多，传统的关键词匹配或规则引擎难以应对：

```
用户查询示例：
├── "公司的请假流程是怎样的？"           → 知识库查询
├── "分析本季度各产品线的毛利率构成"      → 数据分析
├── "查看新用户注册到付费的转化漏斗"      → 数据分析
└── "各部门薪资分布和离群值分析"         → 数据分析
```

### 1.2 传统方案的局限

| 方案 | 局限性 |
|------|--------|
| **关键词匹配** | 无法理解语义，"不想要分析"会被误判为"分析" |
| **规则引擎** | 规则难以穷尽，维护成本高 |
| **简单分类** | 仅区分"查询"和"分析"，无法提供业务上下文 |

### 1.3 我们的解决方案

引入 **LLM 驱动的三维意图识别系统**，实现：

- ✅ 语义级别的深度理解
- ✅ 多维度精细分类
- ✅ 业务上下文自动注入
- ✅ 图表推荐智能引导

---

## 二、系统架构概览

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           用户查询输入                                   │
│                    "分析本季度各产品线的毛利率构成"                        │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         意图识别路由器 (Router)                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     LLM 分类引擎                                 │   │
│  │    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │   │
│  │    │   primary   │    │   domain    │    │   logic     │       │   │
│  │    │  RAG/DATA   │ ×  │ 业务领域    │ ×  │ 分析逻辑    │       │   │
│  │    └─────────────┘    └─────────────┘    └─────────────┘       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────────────────┐
│      RAG Handler        │     │              DATA Handler               │
│   ┌─────────────────┐   │     │  ┌─────────────────────────────────┐    │
│   │  MaxKB 知识库   │   │     │  │         上下文注入              │    │
│   │    查询引擎     │   │     │  │  • domain_context (业务背景)    │    │
│   └─────────────────┘   │     │  │  • logic_context (图表推荐)     │    │
└─────────────────────────┘     │  └─────────────────────────────────┘    │
                                │              │                          │
                                │              ▼                          │
                                │  ┌─────────────────────────────────┐    │
                                │  │       数据分析引擎              │    │
                                │  │  • PandasAI 数据处理            │    │
                                │  │  • Chart Recommender 图表推荐   │    │
                                │  │  • Interactive Charts 可视化    │    │
                                │  └─────────────────────────────────┘    │
                                └─────────────────────────────────────────┘
```

### 2.2 数据流图

```
用户查询
    │
    ▼
┌───────────────────┐
│ classify_query()  │ ───── LLM 三维分类
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  IntentResult     │ ───── 数据结构封装
│  ├─ primary       │
│  ├─ domain        │
│  ├─ logic         │
│  ├─ domain_context│
│  └─ logic_context │
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
  RAG路径    DATA路径
    │           │
    ▼           ▼
 知识库查询   数据分析
```

---

## 三、三维分类体系详解

### 3.1 第一维：Primary（主类别）

决定查询的基本处理路径：

| 类别 | 说明 | 典型查询 |
|------|------|----------|
| **RAG** | 知识库检索类问题 | "请假流程是什么？"、"公司保密制度" |
| **DATA** | 数据分析类任务 | "统计本月销售额"、"分析用户留存趋势" |

```python
class PrimaryType(str, Enum):
    RAG = "RAG"    # 知识库查询
    DATA = "DATA"  # 数据分析
```

### 3.2 第二维：Domain（业务领域）

识别查询所属的业务领域，为分析提供业务上下文：

```
┌─────────────────────────────────────────────────────────────────┐
│                        业务领域矩阵                              │
├─────────────┬───────────────────────────────────────────────────┤
│   FINANCE   │  财务领域：钱的来源、去向与盈亏                    │
│             │  关键词：收入、成本、利润、预算、现金流、毛利       │
│             │  业务常识：借贷平衡、毛利=收入-成本、净利润率       │
├─────────────┼───────────────────────────────────────────────────┤
│   GROWTH    │  增长领域：用户转化、留存与价值                    │
│             │  关键词：转化率、留存率、DAU/MAU、LTV、漏斗         │
│             │  业务常识：漏斗阶段顺序、留存衰减律、DAU/MAU比值    │
├─────────────┼───────────────────────────────────────────────────┤
│     HR      │  人力领域：人员分布、效能与成本                    │
│             │  关键词：员工、薪资、考勤、招聘、离职、绩效         │
│             │  业务常识：职级金字塔、薪酬分位数、离职预警         │
├─────────────┼───────────────────────────────────────────────────┤
│    GOVT     │  政务领域：事项进度、公平性与覆盖率                │
│             │  关键词：政务、审批、办事、投诉、公共服务           │
│             │  业务常识：行政区划映射、办件时效、人均资源均衡     │
└─────────────┴───────────────────────────────────────────────────┘
```

### 3.3 第三维：Logic（分析逻辑）

识别用户期望的分析方式，直接指导图表推荐：

```
┌────────────────────────────────────────────────────────────────────────┐
│                          分析逻辑模式                                   │
├──────────────┬────────────────────────┬────────────────────────────────┤
│    模式      │        分析目标        │          推荐图表              │
├──────────────┼────────────────────────┼────────────────────────────────┤
│ ACHIEVING    │ 进度与达成             │ 子弹图、仪表盘、分组柱状图      │
│              │ 实际 vs 目标对比       │                                │
├──────────────┼────────────────────────┼────────────────────────────────┤
│ COMPOSITION  │ 结构与演变             │ 瀑布图、旭日图、堆叠柱状图      │
│              │ 部分构成整体           │                                │
├──────────────┼────────────────────────┼────────────────────────────────┤
│ FLOW         │ 流转与损耗             │ 桑基图、漏斗图、路径图          │
│              │ 状态间转移             │                                │
├──────────────┼────────────────────────┼────────────────────────────────┤
│ DISTRIBUTION │ 分布与集中             │ 箱线图、小提琴图、直方图        │
│              │ 区间/离群点            │                                │
├──────────────┼────────────────────────┼────────────────────────────────┤
│ TREND        │ 趋势与波动             │ 折线图、面积图、堆叠面积图      │
│              │ 时间维度变化           │                                │
├──────────────┼────────────────────────┼────────────────────────────────┤
│ CORRELATION  │ 关联与对比             │ 散点图、热力矩阵、雷达图        │
│              │ 变量间关系             │                                │
└──────────────┴────────────────────────┴────────────────────────────────┘
```

### 3.4 三维组合示例

| 用户查询 | Primary | Domain | Logic | 推荐图表 |
|----------|---------|--------|-------|----------|
| "本季度毛利率构成分析" | DATA | FINANCE | COMPOSITION | 瀑布图、堆叠柱状图 |
| "预算执行率完成情况" | DATA | FINANCE | ACHIEVING | 子弹图、仪表盘 |
| "用户转化漏斗分析" | DATA | GROWTH | FLOW | 漏斗图、桑基图 |
| "薪资分布和离群值" | DATA | HR | DISTRIBUTION | 箱线图、小提琴图 |
| "政务办件完成率对比" | DATA | GOVT | ACHIEVING | 分组柱状图 |

---

## 四、LLM 在意图识别中的核心作用

### 4.1 LLM 的三大核心能力

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LLM 核心能力矩阵                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐      │
│   │   自然语言理解   │   │   上下文推理    │   │   分类决策      │      │
│   │                 │   │                 │   │                 │      │
│   │ • 语义解析      │   │ • 领域知识      │   │ • 多维分类      │      │
│   │ • 实体识别      │   │ • 隐含意图      │   │ • 置信度评估    │      │
│   │ • 关系抽取      │   │ • 歧义消解      │   │ • 结构化输出    │      │
│   └─────────────────┘   └─────────────────┘   └─────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 自然语言理解 (NLU)

LLM 能够超越关键词匹配，理解查询的深层语义：

```python
# 传统关键词匹配的问题
query = "我不想分析数据，只想查一下制度"
# 关键词匹配：检测到"分析" → 误判为 DATA
# LLM 理解：检测到否定词"不想" → 正确识别为 RAG

# LLM 的语义理解能力
queries = [
    ("公司的请假流程是怎样的？", "RAG"),      # 明确的制度查询
    ("帮我算一下平均工资", "DATA"),           # 隐含的计算需求
    ("对比一下各部门的绩效", "DATA"),         # 对比分析意图
    ("有没有关于加班的规定", "RAG"),          # 知识检索意图
]
```

### 4.3 上下文推理

LLM 利用预训练的领域知识进行推理：

```python
# 业务领域推理示例
query = "分析用户从注册到付费的转化情况"

# LLM 推理过程：
# 1. 识别关键词："用户"、"注册"、"付费"、"转化"
# 2. 关联领域知识：这是典型的用户增长漏斗场景
# 3. 推理结果：domain = GROWTH, logic = FLOW

# 分析逻辑推理示例
query = "各部门薪资分布情况，找出异常高或低的"

# LLM 推理过程：
# 1. 识别关键词："分布"、"异常"、"高或低"
# 2. 关联分析模式：寻找离群点、分布区间
# 3. 推理结果：logic = DISTRIBUTION
```

### 4.4 结构化分类决策

LLM 将自然语言查询转换为结构化分类结果：

```python
# 输入
user_query = "分析本季度各产品线的毛利率构成"

# LLM 输出（结构化 JSON）
{
    "primary": "DATA",
    "domain": "FINANCE",
    "logic": "COMPOSITION"
}

# 系统解析后的完整结果
IntentResult(
    primary="DATA",
    domain="FINANCE", 
    logic="COMPOSITION",
    domain_context="财务分析领域：核心关注钱的来源、去向与盈亏...",
    logic_context="结构与演变分析：优先图表：瀑布图、旭日图..."
)
```

---

## 五、技术实现细节

### 5.1 核心数据结构

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class Domain(str, Enum):
    """四大核心业务领域"""
    FINANCE = "FINANCE"   # 财务
    GROWTH  = "GROWTH"    # 增长
    HR      = "HR"        # 人力
    GOVT    = "GOVT"      # 政务

class LogicPattern(str, Enum):
    """六大通用分析逻辑模式"""
    ACHIEVING    = "ACHIEVING"      # 进度达成
    COMPOSITION  = "COMPOSITION"    # 结构占比
    FLOW         = "FLOW"           # 流转损耗
    DISTRIBUTION = "DISTRIBUTION"   # 分布集中
    TREND        = "TREND"          # 趋势波动
    CORRELATION  = "CORRELATION"    # 关联对比

@dataclass
class IntentResult:
    """路由分类结果"""
    primary: str                      # "RAG" 或 "DATA"
    domain: Optional[str] = None      # Domain 枚举值
    logic: Optional[str] = None       # LogicPattern 枚举值

    @property
    def domain_context(self) -> str:
        """获取领域背景知识字符串"""
        return DOMAIN_CONTEXT.get(self.domain, "")

    @property
    def logic_context(self) -> str:
        """获取分析逻辑背景知识字符串"""
        return LOGIC_CONTEXT.get(self.logic, "")
```

### 5.2 LLM Prompt 工程

精心设计的 System Prompt 是分类准确性的关键：

```python
_SYSTEM_PROMPT = """你是一个企业智能运营助手的意图识别专家，
负责对用户查询进行三维分类。

【分类维度 1 — primary（必填）】
- RAG  : 查询制度/流程/文档/规则/政策/话术等知识性问题
- DATA : 统计数据/分析表格/生成图表/对比数值/趋势分析等数据处理类任务

【分类维度 2 — domain（数据分析时填写，否则 null）】
- FINANCE : 财务、收入、支出、利润、成本、预算、现金流
- GROWTH  : 用户增长、转化率、留存率、DAU/MAU、漏斗
- HR      : 员工、薪资、考勤、招聘、离职、绩效
- GOVT    : 政务、政府、行政、政策、审批、办事

【分类维度 3 — logic（数据分析时填写，否则 null）】
- ACHIEVING    : KPI达成、目标完成率、实际vs计划
- COMPOSITION  : 占比构成、结构分析、来源拆解
- FLOW         : 漏斗转化、流量来源、状态转移
- DISTRIBUTION : 分布区间、离群异常值、密度集中
- TREND        : 时间趋势、月度变化、同比环比
- CORRELATION  : 变量关联、投入产出比、相关性

【输出格式】严格输出 JSON：
{"primary":"DATA","domain":"HR","logic":"DISTRIBUTION"}
"""
```

### 5.3 分类函数实现

```python
def classify_query(user_query: str) -> IntentResult:
    """
    对用户查询进行三维意图分类
    """
    try:
        # 调用 LLM API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.1,    # 低温度提高稳定性
            max_tokens=60,      # 限制输出长度
        )

        # 解析响应
        raw = response.choices[0].message.content.strip()
        
        # 提取 JSON（防止模型附加说明文字）
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"响应中未找到 JSON: {raw}")

        data = json.loads(match.group())

        # 验证并构建结果
        primary = data.get("primary", "DATA").upper()
        domain = validate_domain(data.get("domain"))
        logic = validate_logic(data.get("logic"))

        return IntentResult(primary=primary, domain=domain, logic=logic)

    except Exception as e:
        logger.error(f"分类失败，回退到 DATA: {e}")
        return IntentResult(primary="DATA", domain=None, logic=None)
```

### 5.4 错误处理与降级策略

```python
def validate_domain(value: Optional[str]) -> Optional[str]:
    """验证并规范化 domain 值"""
    if not value:
        return None
    value = value.upper()
    return value if value in Domain._value2member_map_ else None

def validate_logic(value: Optional[str]) -> Optional[str]:
    """验证并规范化 logic 值"""
    if not value:
        return None
    value = value.upper()
    return value if value in LogicPattern._value2member_map_ else None
```

---

## 六、组件协同工作机制

### 6.1 与数据分析引擎协同

```python
# 在 pandas_ai_service.py 中使用意图结果

def analyze(self, file_path: str, query: str, intent: IntentResult) -> Dict:
    """
    执行数据分析，注入业务上下文
    """
    # 1. 加载数据
    df = self.load_data(file_path)
    
    # 2. 注入业务上下文到 LLM 提示词
    analysis_prompt = f"""
    你是一个专业的{intent.domain_context or '数据'}分析师。
    
    {intent.logic_context}
    
    【数据信息】
    - 行数: {len(df)}
    - 列名: {df.columns.tolist()}
    
    【用户问题】
    {query}
    """
    
    # 3. 调用 LLM 分析
    response = self.client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": analysis_prompt}],
    )
    
    return {"response": response.choices[0].message.content}
```

### 6.2 与图表推荐引擎协同

```python
# 在 chart_recommender.py 中使用意图结果

def recommend_charts(self, df: pd.DataFrame, intent: IntentResult) -> List[str]:
    """
    基于意图识别结果推荐图表
    """
    # 根据 logic 模式直接推荐图表
    logic_to_charts = {
        "ACHIEVING": ["bullet", "gauge", "grouped_bar"],
        "COMPOSITION": ["waterfall", "sunburst", "stacked_bar"],
        "FLOW": ["sankey", "funnel", "path"],
        "DISTRIBUTION": ["boxplot", "violin", "histogram"],
        "TREND": ["line", "area", "stacked_area"],
        "CORRELATION": ["scatter", "heatmap", "radar"],
    }
    
    # 获取推荐图表列表
    recommended = logic_to_charts.get(intent.logic, [])
    
    # 根据数据特征进一步筛选
    return self._filter_by_data_features(df, recommended)
```

### 6.3 完整工作流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           完整工作流程                                   │
└─────────────────────────────────────────────────────────────────────────┘

用户输入: "分析本季度各产品线的毛利率构成"
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 1: 意图识别                                                         │
│                                                                          │
│   classify_query() → IntentResult(                                       │
│       primary = "DATA",                                                  │
│       domain  = "FINANCE",                                               │
│       logic   = "COMPOSITION"                                            │
│   )                                                                      │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 2: 路由分发                                                         │
│                                                                          │
│   if primary == "RAG":                                                   │
│       → 知识库查询引擎                                                   │
│   else:                                                                  │
│       → 数据分析引擎                                                     │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 3: 上下文注入                                                       │
│                                                                          │
│   domain_context = "财务分析领域：核心关注钱的来源..."                   │
│   logic_context  = "结构与演变分析：优先图表：瀑布图..."                 │
│                                                                          │
│   → 注入到 LLM 分析提示词                                                │
│   → 传递给图表推荐引擎                                                   │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 4: 数据分析                                                         │
│                                                                          │
│   PandasAI 分析 + LLM 业务解读                                           │
│   Chart Recommender 推荐图表                                             │
│   Interactive Charts 生成可视化                                          │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 5: 结果返回                                                         │
│                                                                          │
│   {                                                                      │
│       "analysis": "本季度毛利率构成分析...",                             │
│       "charts": ["waterfall_chart.png", "stacked_bar.png"],              │
│       "tables": ["margin_breakdown.xlsx"]                                │
│   }                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 七、应用场景示例

### 7.1 财务分析场景

```python
# 用户查询
query = "分析本季度各产品线的毛利率构成，对比去年同期"

# 意图识别结果
IntentResult(
    primary="DATA",
    domain="FINANCE",
    logic="COMPOSITION"
)

# 系统响应
{
    "analysis": "本季度毛利率构成分析...",
    "charts": ["waterfall_chart.png", "stacked_bar_yoy.png"],
    "insights": [
        "产品线A毛利率贡献最大，占比35%",
        "产品线B毛利率同比下降5个百分点"
    ]
}
```

### 7.2 用户增长场景

```python
# 用户查询
query = "查看新用户从注册到首次付费的转化漏斗"

# 意图识别结果
IntentResult(
    primary="DATA",
    domain="GROWTH",
    logic="FLOW"
)

# 系统响应
{
    "analysis": "用户转化漏斗分析...",
    "charts": ["funnel_chart.png", "sankey_diagram.png"],
    "insights": [
        "注册→激活转化率68%，高于行业均值",
        "激活→付费转化率12%，存在优化空间"
    ]
}
```

### 7.3 人力资源场景

```python
# 用户查询
query = "各部门薪资分布情况，找出异常高或低的"

# 意图识别结果
IntentResult(
    primary="DATA",
    domain="HR",
    logic="DISTRIBUTION"
)

# 系统响应
{
    "analysis": "薪资分布分析...",
    "charts": ["boxplot_salary.png", "violin_by_dept.png"],
    "insights": [
        "研发部薪资中位数最高（P50=25K）",
        "行政部存在2个离群高值（>P75+1.5IQR）"
    ]
}
```

### 7.4 政务服务场景

```python
# 用户查询
query = "各区县政务办件完成率与目标值对比"

# 意图识别结果
IntentResult(
    primary="DATA",
    domain="GOVT",
    logic="ACHIEVING"
)

# 系统响应
{
    "analysis": "政务办件完成率分析...",
    "charts": ["bullet_chart.png", "grouped_bar_target.png"],
    "insights": [
        "A区完成率92%，超额完成目标",
        "C区完成率78%，距目标差12个百分点"
    ]
}
```

---

## 八、性能优化与最佳实践

### 8.1 性能优化策略

| 策略 | 说明 | 效果 |
|------|------|------|
| **低温度采样** | temperature=0.1 | 提高输出稳定性 |
| **限制输出长度** | max_tokens=60 | 减少延迟和成本 |
| **JSON 强制提取** | 正则匹配提取 JSON | 容错处理 |
| **降级策略** | 异常时回退到 DATA | 保证系统可用 |

### 8.2 准确性优化

```python
# 1. 丰富的分类示例
_SYSTEM_PROMPT = """
【分类示例】
"请假流程是什么" → {"primary":"RAG","domain":null,"logic":null}
"分析毛利率构成" → {"primary":"DATA","domain":"FINANCE","logic":"COMPOSITION"}
"用户转化漏斗"   → {"primary":"DATA","domain":"GROWTH","logic":"FLOW"}
"""

# 2. 明确的边界条件
"""
【边界说明】
- "对比"不一定是 CORRELATION，也可能是 ACHIEVING（对比目标）
- "分布"一定是 DISTRIBUTION
- "趋势"一定涉及时间维度
"""

# 3. 领域关键词映射
DOMAIN_KEYWORDS = {
    "FINANCE": ["收入", "成本", "利润", "预算", "毛利", "现金流"],
    "GROWTH": ["用户", "转化", "留存", "DAU", "MAU", "漏斗"],
    "HR": ["员工", "薪资", "考勤", "招聘", "离职", "绩效"],
    "GOVT": ["政务", "审批", "办事", "投诉", "公共服务"],
}
```

### 8.3 监控与日志

```python
# 分类结果日志
logger.info(f"[Router] 分类结果: primary={primary}, domain={domain}, logic={logic}")

# 异常情况告警
if primary not in ("RAG", "DATA"):
    logger.warning(f"[Router] 无效的 primary 值 '{primary}'，回退到 DATA")

# 性能监控
import time
start = time.time()
result = classify_query(query)
latency = time.time() - start
logger.info(f"[Router] 分类耗时: {latency:.3f}s")
```

---

## 九、未来展望

### 9.1 短期优化

- [ ] 添加分类置信度返回
- [ ] 支持多轮对话上下文
- [ ] 增加更多业务领域（如供应链、客服）

### 9.2 中期规划

- [ ] 本地小模型替代（降低成本和延迟）
- [ ] 用户反馈学习机制
- [ ] 领域自适应微调

### 9.3 长期愿景

- [ ] 多模态意图识别（支持图表、文件上传）
- [ ] 主动式意图预测
- [ ] 跨领域知识迁移

---

## 总结

OpsMind 的三维意图识别系统通过 **Primary × Domain × Logic** 的分类体系，实现了：

1. **精准路由**：准确区分知识查询和数据分析
2. **业务理解**：识别查询所属的业务领域
3. **智能引导**：根据分析逻辑推荐最佳图表
4. **上下文增强**：为下游组件注入业务知识

这一设计充分利用了 LLM 的语义理解和推理能力，同时通过结构化输出和降级策略保证了系统的稳定性和可靠性。

---

**参考资料**：
- [DeepSeek API 文档](https://platform.deepseek.com/docs)
- [意图识别最佳实践](https://example.com)
- [Prompt Engineering Guide](https://www.promptingguide.ai)

---

*本文由 OpsMind 技术团队撰写，转载请注明出处。*
