---
title: "LLM 在 OpsMind 项目中的技术实现详解"
date: 2026-02-10T10:00:00+08:00
draft: false
tags: ["LLM", "DeepSeek", "技术实现", "代码分析"]
categories: ["技术文章"]
author: "wwxdsg"
description: "深入解析LLM在OpsMind项目中的角色定位、调用场景、Prompt工程实践及性能优化策略"
permalink: /opsmind/blog5_llm_implementation/
---

## 一、LLM 在项目中的角色定位

### 1.1 系统架构中的位置

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          OpsMind 系统架构                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                        LLM 服务层                                │   │
│   │                                                                  │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │   │
│   │   │  意图识别   │  │  数据分析   │  │  上下文压缩 │            │   │
│   │   │  (Router)   │  │ (PandasAI)  │  │ (Context)   │            │   │
│   │   └─────────────┘  └─────────────┘  └─────────────┘            │   │
│   │                                                                  │   │
│   │   ┌─────────────┐  ┌─────────────┐                             │   │
│   │   │  图表推荐   │  │  列映射推断 │                             │   │
│   │   │(Recommender)│  │(Inference)  │                             │   │
│   │   └─────────────┘  └─────────────┘                             │   │
│   │                                                                  │   │
│   │                    ↓ OpenAI API ↓                               │   │
│   │              ┌─────────────────────┐                            │   │
│   │              │   DeepSeek LLM      │                            │   │
│   │              │   (deepseek-chat)   │                            │   │
│   │              └─────────────────────┘                            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 LLM 承担的五大核心功能

| 功能模块 | 文件位置 | LLM 作用 | 调用频率 |
|----------|----------|----------|----------|
| **意图识别** | `router.py` | 三维分类 (primary/domain/logic) | 每次查询 |
| **数据分析** | `pandas_ai_service.py` | 业务洞察生成 | 每次数据分析 |
| **图表推荐增强** | `chart_recommender.py` | 用户意图理解优化 | 可选调用 |
| **列映射推断** | `chart_recommender.py` | 智能列选择 | 复杂图表时 |
| **上下文压缩** | `context_manager.py` | 对话摘要生成 | 长对话时 |

---

## 二、LLM 客户端初始化机制

### 2.1 配置管理

**配置文件**: `src/config.py`

```python
class Config:
    """应用配置类"""
    
    # DeepSeek API 配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    
    # 应用配置
    APP_ENV = os.getenv("APP_ENV", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # 可视化配置
    CHART_THEME = os.getenv("CHART_THEME", "plotly_white")
    CHART_WIDTH = int(os.getenv("CHART_WIDTH", "800"))
    CHART_HEIGHT = int(os.getenv("CHART_HEIGHT", "600"))
```

**环境变量文件**: `.env`

```dotenv
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
APP_ENV=development
LOG_LEVEL=INFO
```

### 2.2 客户端初始化模式

项目中存在两种初始化模式：

#### 模式一：模块级全局客户端（推荐用于高频调用）

```python
# src/router.py
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE"),
)
```

**特点**：
- 模块加载时初始化
- 全局复用，减少连接开销
- 适用于高频调用场景

#### 模式二：类内懒加载客户端（推荐用于可选调用）

```python
# src/services/chart_recommender.py
class ChartRecommender:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.client = None
        
        if use_llm:
            try:
                api_key = os.getenv("DEEPSEEK_API_KEY")
                api_base = os.getenv("DEEPSEEK_API_BASE")
                if api_key and api_base:
                    self.client = OpenAI(api_key=api_key, base_url=api_base)
                    logger.info("LLM 客户端初始化成功")
            except Exception as e:
                logger.warning(f"LLM 客户端初始化失败: {str(e)}")
```

**特点**：
- 按需初始化，节省资源
- 支持降级运行（LLM 不可用时仍可用规则推荐）
- 适用于可选增强场景

#### 模式三：函数内临时客户端（用于低频调用）

```python
# src/services/context_manager.py
def _generate_summary(self, conversation_text: str) -> Optional[str]:
    try:
        from openai import OpenAI
        
        api_key = Config.DEEPSEEK_API_KEY
        api_base = Config.DEEPSEEK_API_BASE
        
        if not api_key:
            logger.warning("未配置 DEEPSEEK_API_KEY，跳过摘要生成")
            return None
        
        client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        response = client.chat.completions.create(...)
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"生成摘要失败: {str(e)}")
        return None
```

**特点**：
- 函数内创建，用完即销毁
- 适用于低频、可选功能
- 便于错误隔离

---

## 三、LLM 调用场景详解

### 3.1 场景一：三维意图识别

**文件**: `src/router.py`

**调用位置**: `classify_query()` 函数

```python
def classify_query(user_query: str) -> IntentResult:
    """
    对用户查询进行三维意图分类
    
    Args:
        user_query: 用户输入的查询问题
    
    Returns:
        IntentResult 包含 primary / domain / logic 三个维度
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.1,    # 低温度提高分类稳定性
            max_tokens=60,      # 限制输出长度，降低成本
        )

        raw = response.choices[0].message.content.strip()
        
        # 提取 JSON（防止模型在 JSON 外附加说明文字）
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

**参数详解**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `model` | `"deepseek-chat"` | DeepSeek 通用对话模型 |
| `messages` | `[system, user]` | 系统提示词 + 用户查询 |
| `temperature` | `0.1` | 低温度确保输出稳定、可预测 |
| `max_tokens` | `60` | 分类结果短，限制输出节省成本 |

### 3.2 场景二：数据分析洞察生成

**文件**: `src/services/pandas_ai_service.py`

**调用位置**: `analyze()` 方法

```python
def analyze(self, file_path: str, query: str) -> Dict[str, Any]:
    """执行数据分析"""
    
    # 1. 加载数据
    df = self.load_data(file_path)
    
    # 2. 创建数据摘要
    data_summary = self.create_smart_dataframe(df)
    
    # 3. 构建分析提示
    analysis_prompt = f"""
你是一个专业的数据分析师。请根据以下数据信息，回答用户的分析问题。

【数据信息】
- 行数: {data_summary['rows']}
- 列数: {data_summary['columns']}
- 列名: {', '.join(data_summary['column_names'])}
- 数据类型: {json.dumps(data_summary['dtypes'], ensure_ascii=False)}

【数据预览】
{data_summary['preview']}

【基本统计】
{data_summary['basic_stats']}

【用户问题】
{query}

请提供详细的分析结果，包括关键发现和建议。
"""
    
    # 4. 调用 LLM API
    response = self.client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "你是一个专业的数据分析师，能够深入分析数据并提供有价值的见解。"
            },
            {
                "role": "user",
                "content": analysis_prompt
            }
        ],
        temperature=0.7,      # 较高温度增加分析多样性
        max_tokens=2000       # 允许较长的分析输出
    )
    
    result_text = response.choices[0].message.content
    
    return {
        "status": "success",
        "query": query,
        "response": result_text,
        "data_shape": f"{len(df)} 行 × {len(df.columns)} 列",
    }
```

**参数详解**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `temperature` | `0.7` | 较高温度，分析更具创造性 |
| `max_tokens` | `2000` | 允许详细分析，支持多段落输出 |

### 3.3 场景三：图表推荐增强

**文件**: `src/services/chart_recommender.py`

**调用位置**: `_enhance_with_llm()` 方法

```python
def _enhance_with_llm(self,
                      query: str,
                      features: DataFeatures,
                      recommendations: List[ChartRecommendation]) -> List[ChartRecommendation]:
    """
    使用 LLM 增强推荐
    
    Args:
        query: 用户查询
        features: 数据特征
        recommendations: 初始推荐列表
    
    Returns:
        增强后的推荐列表
    """
    if not self.client:
        return recommendations
    
    try:
        chart_options = {
            r.chart_type: self.CHART_INFO[r.chart_type]["name"] 
            for r in recommendations
        }
        
        prompt = f"""你是一个数据可视化专家。根据用户问题和数据特征，调整图表推荐顺序。

【用户问题】
{query}

【数据特征】
- 行数: {features.row_count}
- 数值列: {', '.join(features.numeric_cols) if features.numeric_cols else '无'}
- 分类列: {', '.join(features.categorical_cols) if features.categorical_cols else '无'}
- 是否有强相关性: {'是' if features.has_strong_correlations else '否'}

【候选图表】
{json.dumps(chart_options, ensure_ascii=False, indent=2)}

请根据用户问题，选择最相关的图表类型（最多3个），按重要性排序返回。

返回 JSON 格式：
{{"ordered_charts": ["chart_type1", "chart_type2", ...]}}
"""
        
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是数据可视化专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        import json
        result = json.loads(response.choices[0].message.content.strip())
        ordered_charts = result.get("ordered_charts", [])
        
        # 根据用户意图重新排序推荐
        if ordered_charts:
            chart_order = {chart: i for i, chart in enumerate(ordered_charts)}
            
            for rec in recommendations:
                if rec.chart_type in chart_order:
                    rec.score = min(1.0, rec.score + 0.2)
                    rec.priority = 10 - chart_order[rec.chart_type]
            
            recommendations.sort(key=lambda x: (x.priority, -x.score))
        
        return recommendations
    
    except Exception as e:
        logger.warning(f"LLM 增强失败: {str(e)}")
        return recommendations
```

### 3.4 场景四：对话上下文压缩

**文件**: `src/services/context_manager.py`

**调用位置**: `_generate_summary()` 方法

```python
def _generate_summary(self, conversation_text: str) -> Optional[str]:
    """
    使用 LLM 生成对话摘要
    
    Args:
        conversation_text: 对话文本
    
    Returns:
        摘要文本
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_API_BASE
        )
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个对话摘要助手。请将以下对话内容压缩为简洁的摘要，保留关键信息和上下文。摘要长度不超过500字。"
                },
                {
                    "role": "user",
                    "content": f"请总结以下对话：\n\n{conversation_text}"
                }
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content
        return summary
        
    except Exception as e:
        logger.error(f"生成摘要失败: {str(e)}")
        return None
```

### 3.5 场景五：图表列映射推断

**文件**: `src/services/chart_recommender.py`

**调用位置**: `infer_column_mapping()` 方法

```python
def infer_column_mapping(self,
                         chart_type: str,
                         df: pd.DataFrame,
                         query: Optional[str] = None) -> ChartInference:
    """
    使用 LLM 智能推断图表的列映射
    
    Args:
        chart_type: 图表类型
        df: 数据框
        query: 用户查询（可选）
    
    Returns:
        图表推断结果
    """
    features = self.analyze_data(df)
    requirements = self.get_chart_requirements(chart_type)
    
    if not self.client:
        # LLM 不可用时使用默认映射
        mapping = self._default_column_mapping(chart_type, features)
        return ChartInference(
            chart_type=chart_type,
            column_mapping=mapping,
            confidence=0.5,
            reason="使用默认列映射（LLM 不可用）",
            needs_clarification=True
        )
    
    try:
        prompt = self._build_inference_prompt(chart_type, features, query)
        
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是数据可视化专家，能够准确判断图表所需的列映射。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        
        # 解析 LLM 返回的列映射
        result = json.loads(response.choices[0].message.content)
        
        return ChartInference(
            chart_type=chart_type,
            column_mapping=ChartColumnMapping(**result),
            confidence=0.9,
            reason="LLM 智能推断"
        )
        
    except Exception as e:
        logger.warning(f"列映射推断失败: {str(e)}")
        return self._fallback_inference(chart_type, features)
```

---

## 四、核心函数参数与实现分析

### 4.1 OpenAI API 调用参数详解

```python
response = client.chat.completions.create(
    # 必需参数
    model="deepseek-chat",           # 模型标识符
    
    # 消息列表
    messages=[
        {"role": "system", "content": "..."},   # 系统提示词
        {"role": "user", "content": "..."},     # 用户输入
        {"role": "assistant", "content": "..."}, # 历史回复（多轮对话）
    ],
    
    # 可选参数
    temperature=0.1,      # 温度参数：0-2，控制随机性
    max_tokens=60,        # 最大输出 token 数
    top_p=1.0,           # 核采样参数
    frequency_penalty=0,  # 频率惩罚：-2.0 到 2.0
    presence_penalty=0,   # 存在惩罚：-2.0 到 2.0
    stop=None,           # 停止序列
    stream=False,        # 是否流式输出
)
```

### 4.2 参数选择策略

| 场景 | temperature | max_tokens | 原因 |
|------|-------------|------------|------|
| **意图分类** | 0.1 | 60 | 需要稳定、可预测的输出 |
| **数据分析** | 0.7 | 2000 | 需要创造性、详细的分析 |
| **图表推荐** | 0.3 | 200 | 需要一定灵活性但保持稳定 |
| **对话摘要** | 0.3 | 500 | 需要准确概括，适度灵活 |
| **列映射推断** | 0.2 | 300 | 需要精确匹配列名 |

### 4.3 响应解析流程

```python
def parse_llm_response(response) -> dict:
    """
    解析 LLM 响应的标准流程
    """
    # Step 1: 获取原始文本
    raw_content = response.choices[0].message.content.strip()
    
    # Step 2: 提取 JSON（处理可能的额外文本）
    import re
    match = re.search(r"\{.*?\}", raw_content, re.DOTALL)
    if not match:
        raise ValueError("响应中未找到有效的 JSON")
    
    # Step 3: 解析 JSON
    data = json.loads(match.group())
    
    # Step 4: 验证字段
    # ... 字段验证逻辑
    
    # Step 5: 返回结构化数据
    return data
```

---

## 五、Prompt 工程实践

### 5.1 意图识别 System Prompt

```python
_SYSTEM_PROMPT = """你是一个企业智能运营助手的意图识别专家，负责对用户查询进行三维分类。

【分类维度 1 — primary（必填）】
- RAG  : 查询制度/流程/文档/规则/政策/话术等知识性问题
- DATA : 统计数据/分析表格/生成图表/对比数值/趋势分析等数据处理类任务

【分类维度 2 — domain（数据分析时填写，否则 null）】
- FINANCE : 财务、收入、支出、利润、成本、预算、现金流、营收、毛利
- GROWTH  : 用户增长、转化率、留存率、DAU/MAU、注册、激活、付费、LTV
- HR      : 员工、人员、薪资、考勤、招聘、离职、绩效、职级、司龄
- GOVT    : 政务、政府、行政、政策、申请、审批、办事、投诉、公共服务

【分类维度 3 — logic（数据分析时填写，否则 null）】
- ACHIEVING    : KPI达成、目标完成率、实际vs计划、预算执行
- COMPOSITION  : 占比构成、结构分析、来源拆解、增减变化
- FLOW         : 漏斗转化、流量来源、资金流向、审批路径
- DISTRIBUTION : 分布区间、离群异常值、密度集中
- TREND        : 时间趋势、月度变化、同比环比、波动周期
- CORRELATION  : 变量关联、投入产出比、相关性分析

【输出格式】严格输出 JSON，不含任何注释或多余文字：
{"primary":"DATA","domain":"HR","logic":"DISTRIBUTION"}
"""
```

**设计要点**：
1. **角色定义**：明确 LLM 扮演的角色
2. **分类维度**：清晰定义每个维度的取值范围
3. **关键词映射**：提供关键词帮助 LLM 理解分类
4. **输出格式**：强制 JSON 格式，便于程序解析

### 5.2 数据分析 Prompt 模板

```python
def build_analysis_prompt(df: pd.DataFrame, query: str, intent: IntentResult) -> str:
    """
    构建数据分析提示词
    """
    # 获取业务上下文
    domain_context = intent.domain_context  # 如："财务分析领域：..."
    logic_context = intent.logic_context     # 如："结构与演变分析：..."
    
    prompt = f"""
你是一个专业的{domain_context or '数据'}分析师。

{logic_context}

【数据信息】
- 行数: {len(df)}
- 列数: {len(df.columns)}
- 列名: {df.columns.tolist()}
- 数据类型: {df.dtypes.to_dict()}

【数据预览】
{df.head(10).to_string()}

【基本统计】
{df.describe().to_string()}

【用户问题】
{query}

请提供详细的分析结果，包括：
1. 关键发现（最重要的 3-5 个洞察）
2. 数据特征分析
3. 建议和行动项
"""
    return prompt
```

### 5.3 Prompt 设计最佳实践

| 原则 | 说明 | 示例 |
|------|------|------|
| **角色明确** | 定义 LLM 扮演的专业角色 | "你是一个企业智能运营助手的意图识别专家" |
| **任务清晰** | 明确说明要完成的任务 | "对用户查询进行三维分类" |
| **格式约束** | 强制输出格式 | "严格输出 JSON，不含任何注释" |
| **示例引导** | 提供正确输出示例 | `{"primary":"DATA","domain":"HR","logic":"DISTRIBUTION"}` |
| **上下文注入** | 提供业务背景知识 | "财务分析领域：核心关注钱的来源、去向与盈亏" |

---

## 六、输入输出处理流程

### 6.1 完整调用流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          LLM 调用完整流程                                │
└─────────────────────────────────────────────────────────────────────────┘

用户输入
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 1: 输入预处理                                                       │
│                                                                          │
│   • 清理空白字符                                                         │
│   • 提取关键信息                                                         │
│   • 构建数据摘要                                                         │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 2: Prompt 构建                                                      │
│                                                                          │
│   • 加载 System Prompt 模板                                              │
│   • 注入业务上下文 (domain_context, logic_context)                       │
│   • 添加用户查询和数据信息                                                │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 3: API 调用                                                         │
│                                                                          │
│   client.chat.completions.create(                                        │
│       model="deepseek-chat",                                             │
│       messages=[system_prompt, user_prompt],                             │
│       temperature=0.1,                                                   │
│       max_tokens=60                                                      │
│   )                                                                      │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 4: 响应解析                                                         │
│                                                                          │
│   raw = response.choices[0].message.content                              │
│   match = re.search(r"\{.*?\}", raw, re.DOTALL)                          │
│   data = json.loads(match.group())                                       │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 5: 结果验证                                                         │
│                                                                          │
│   • 验证必需字段是否存在                                                  │
│   • 验证字段值是否合法                                                    │
│   • 类型转换和规范化                                                      │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 6: 返回结构化结果                                                   │
│                                                                          │
│   return IntentResult(                                                   │
│       primary=primary,                                                   │
│       domain=domain,                                                     │
│       logic=logic                                                        │
│   )                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 输入处理代码

```python
def preprocess_query(query: str) -> str:
    """
    预处理用户查询
    """
    # 1. 去除首尾空白
    query = query.strip()
    
    # 2. 统一换行符
    query = query.replace('\r\n', '\n')
    
    # 3. 压缩连续空白
    import re
    query = re.sub(r'\s+', ' ', query)
    
    # 4. 截断过长查询
    if len(query) > 2000:
        query = query[:2000] + "..."
    
    return query
```

### 6.3 输出处理代码

```python
def parse_intent_response(raw: str) -> IntentResult:
    """
    解析意图识别响应
    """
    # 1. 提取 JSON
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"响应中未找到 JSON: {raw}")
    
    # 2. 解析 JSON
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {e}")
    
    # 3. 验证 primary
    primary = data.get("primary", "DATA").upper()
    if primary not in ("RAG", "DATA"):
        logger.warning(f"无效的 primary 值 '{primary}'，回退到 DATA")
        primary = "DATA"
    
    # 4. 验证 domain
    domain = validate_domain(data.get("domain"))
    
    # 5. 验证 logic
    logic = validate_logic(data.get("logic"))
    
    return IntentResult(primary=primary, domain=domain, logic=logic)


def validate_domain(value: Optional[str]) -> Optional[str]:
    """验证并规范化 domain 值"""
    if not value:
        return None
    value = value.upper()
    valid_domains = {"FINANCE", "GROWTH", "HR", "GOVT"}
    return value if value in valid_domains else None


def validate_logic(value: Optional[str]) -> Optional[str]:
    """验证并规范化 logic 值"""
    if not value:
        return None
    value = value.upper()
    valid_logics = {
        "ACHIEVING", "COMPOSITION", "FLOW", 
        "DISTRIBUTION", "TREND", "CORRELATION"
    }
    return value if value in valid_logics else None
```

---

## 七、错误处理与降级策略

### 7.1 错误类型与处理

```python
def classify_query_with_retry(user_query: str, max_retries: int = 3) -> IntentResult:
    """
    带重试的意图分类
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(...)
            return parse_intent_response(response.choices[0].message.content)
            
        except json.JSONDecodeError as e:
            # JSON 解析错误，可能需要调整 prompt
            logger.warning(f"JSON 解析失败 (尝试 {attempt+1}/{max_retries}): {e}")
            
        except openai.APIConnectionError as e:
            # 网络连接错误，等待后重试
            logger.warning(f"网络连接失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(1 * (attempt + 1))  # 指数退避
            
        except openai.RateLimitError as e:
            # 速率限制，等待更长时间
            logger.warning(f"速率限制 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(5 * (attempt + 1))
            
        except openai.APIStatusError as e:
            # API 服务错误，可能需要降级
            logger.error(f"API 服务错误: {e}")
            break
            
        except Exception as e:
            logger.error(f"未知错误: {e}")
            break
    
    # 所有尝试失败，返回默认值
    logger.error("所有重试失败，返回默认分类")
    return IntentResult(primary="DATA", domain=None, logic=None)
```

### 7.2 降级策略矩阵

| 错误类型 | 降级策略 | 用户体验影响 |
|----------|----------|--------------|
| **API Key 无效** | 抛出异常，提示配置 | 无法使用 LLM 功能 |
| **网络连接失败** | 重试 3 次后降级 | 使用规则引擎替代 |
| **速率限制** | 等待后重试 | 短暂延迟 |
| **JSON 解析失败** | 返回默认值 | 分类可能不准确 |
| **超时** | 返回默认值 | 快速响应 |

### 7.3 无 LLM 降级方案

```python
class ChartRecommender:
    def recommend_charts(self, df: pd.DataFrame, ...) -> List[ChartRecommendation]:
        """
        推荐图表，支持无 LLM 降级
        """
        features = self.analyze_data(df)
        recommendations = []
        
        # 基于规则的推荐（无需 LLM）
        for chart_type in self.CHART_INFO:
            score = self.calculate_chart_score(chart_type, features)
            if score >= Config.RECOMMEND_MIN_SCORE:
                recommendations.append(...)
        
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        # LLM 增强（可选）
        if query and self.client:  # 检查 LLM 是否可用
            try:
                recommendations = self._enhance_with_llm(query, features, recommendations)
            except Exception as e:
                logger.warning(f"LLM 增强失败，使用规则推荐: {e}")
        
        return recommendations[:max_results]
```

---

## 八、性能优化策略

### 8.1 参数优化

| 参数 | 优化策略 | 效果 |
|------|----------|------|
| `temperature` | 分类任务用低值(0.1)，创作任务用高值(0.7) | 平衡稳定性和创造性 |
| `max_tokens` | 根据输出需求精确设置 | 减少不必要的 token 消耗 |
| `model` | 简单任务用轻量模型，复杂任务用高级模型 | 成本与质量平衡 |

### 8.2 缓存策略

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def classify_query_cached(query_hash: str) -> IntentResult:
    """
    缓存意图分类结果
    """
    # 实际调用 LLM
    return classify_query(query_hash)

def classify_query_with_cache(query: str) -> IntentResult:
    """
    带缓存的意图分类
    """
    # 生成查询哈希
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return classify_query_cached(query_hash)
```

### 8.3 批量处理

```python
async def batch_classify_queries(queries: List[str]) -> List[IntentResult]:
    """
    批量处理查询（异步）
    """
    import asyncio
    
    async def classify_one(query: str) -> IntentResult:
        # 异步调用 API
        ...
    
    tasks = [classify_one(q) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return [
        r if isinstance(r, IntentResult) 
        else IntentResult(primary="DATA", domain=None, logic=None)
        for r in results
    ]
```

### 8.4 性能监控

```python
import time
from contextlib import contextmanager

@contextmanager
def llm_call_monitor(func_name: str):
    """
    LLM 调用监控上下文管理器
    """
    start_time = time.time()
    try:
        yield
    finally:
        latency = time.time() - start_time
        logger.info(f"[LLM Monitor] {func_name} 耗时: {latency:.3f}s")
        
        # 告警阈值
        if latency > 5.0:
            logger.warning(f"[LLM Monitor] {func_name} 响应过慢: {latency:.3f}s")

# 使用示例
def classify_query(user_query: str) -> IntentResult:
    with llm_call_monitor("classify_query"):
        response = client.chat.completions.create(...)
        return parse_response(response)
```

---

## 总结

本文从代码层面详细分析了 LLM 在 OpsMind 项目中的技术实现：

1. **客户端初始化**：三种模式适应不同场景
2. **五大调用场景**：意图识别、数据分析、图表推荐、列映射推断、上下文压缩
3. **参数配置策略**：根据任务特点选择 temperature 和 max_tokens
4. **Prompt 工程**：角色定义、任务清晰、格式约束、示例引导
5. **错误处理**：重试机制、降级策略、无 LLM 方案
6. **性能优化**：缓存、批量处理、监控告警

通过这些技术实践，OpsMind 实现了 LLM 的稳定、高效、可控集成。

---

*本文由 OpsMind 技术团队撰写，转载请注明出处。*
