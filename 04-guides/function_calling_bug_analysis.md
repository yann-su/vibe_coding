# 智谱 GLM-4.7 Function Calling Bug 分析报告

## 问题回顾

### 测试场景
用户请求：`"推荐一部2023年的高分科幻电影"`

使用 Function Calling 让模型返回结构化电影信息：
```python
class Movie(BaseModel):
    name: str      # 电影名称
    year: int      # 上映年份
    director: str  # 导演
    rating: float  # 评分
    genres: List[str]  # 类型
```

---

## Bug 现象

### 1. 模型思考过程（reasoning_content）✅ 正确
```
用户要求推荐一部2023年的高分科幻电影...

让我推荐《流浪地球2》：
- name: 流浪地球2
- year: 2023
- director: 郭帆
- rating: 8.3
- genres: ["科幻", "动作", "剧情"]

这部是2023年中国制作的高分科幻电影，非常符合用户的要求。
```

**分析**：模型完全理解任务，知道要填什么值。

### 2. 实际调用参数（tool_calls）❌ 错误
```json
{
  "tool_calls": [{
    "function": {
      "name": "extract_movie",
      "arguments": "{\"name\": \"\"}"  // ← 只有空字符串！
    }
  }]
}
```

**分析**：虽然模型思考正确，但实际传递的参数是空的。

---

## 根本原因

```
┌─────────────────────────────────────────────────────────────┐
│                     GLM-4.7 内部处理                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  用户输入 ──▶ 模型理解 ──▶ 生成思考（reasoning_content）     │
│                              │                              │
│                              ▼                              │
│                         "推荐流浪地球2"                      │
│                         name="流浪地球2"                     │
│                         year=2023                           │
│                         ...                                 │
│                              │                              │
│                              ▼                              │
│                    生成 function calling                    │
│                              │                              │
│                    ┌─────────┴─────────┐                   │
│                    ▼                   ▼                   │
│              【预期】              【实际】                  │
│         {"name": "流浪地球2"}     {"name": ""}               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**结论**：这是 GLM-4.7 模型训练或推理引擎的 **bug**。

模型在 reasoning 阶段正确理解了任务，但在生成 function calling 参数时出现了错误，只输出了空值。

---

## 对比测试

### 测试 1: 简单参数（单个字符串）
```python
class Simple(BaseModel):
    value: str

# 结果: {"value": ""}  ← 仍然为空
```

### 测试 2: 无参数函数
```python
# 某些情况下模型可以正确调用无参数函数
```

### 测试 3: 不同模型对比

| 模型 | Function Calling | 状态 |
|------|------------------|------|
| GLM-4.7 | 参数为空 | ❌ Bug |
| OpenAI GPT-4 | 正常工作 | ✅ 正常 |
| Claude 3.5 | 正常工作 | ✅ 正常 |

---

## 解决方案

### 方案 1: 不使用 Function Calling（推荐 ✅）

使用强化 Prompt + 后处理，这是目前最可靠的方案。

```python
# output_parser_fixed.py 中的方案
def reliable_json_parser(pydantic_model, prompt_text, llm):
    # 1. 强化 Prompt 约束
    enhanced_prompt = f"""
    你必须严格按照以下 JSON Schema 格式输出：
    {format_instructions}

    ⚠️ 严格规则：
    1. 只输出纯 JSON，不要 markdown
    2. 确保所有必填字段都有值
    3. 字符串用双引号

    任务: {prompt_text}
    """

    # 2. 调用 LLM
    response = llm.invoke(enhanced_prompt)

    # 3. 后处理清理
    cleaned = clean_json_output(response.content)

    # 4. 解析验证
    return parser.parse(cleaned)
```

**优点**：
- 不依赖 API 的 function calling
- 跨平台兼容
- 可控性高

### 方案 2: 使用其他模型

如果必须使用 function calling，换用支持良好的模型：

```python
# OpenAI GPT-4
llm = ChatOpenAI(model="gpt-4", api_key="...")
structured_llm = llm.with_structured_output(Movie)

# Claude
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
```

### 方案 3: 尝试智谱其他模型版本

```python
# 尝试 glm-4-plus 或其他版本
llm = ChatOpenAI(model="glm-4-plus", ...)
```

---

## 总结

### 问题定性
**GLM-4.7 的 Function Calling 实现存在 bug**，具体表现为：
- reasoning_content 正确
- tool_calls.arguments 为空

### 当前最佳实践
**放弃 Function Calling，使用强化 Prompt + 后处理方案**（`output_parser_fixed.py`）。

### 代码选择指南

| 场景 | 推荐文件 | 说明 |
|------|----------|------|
| 智谱 GLM-4.7 | `output_parser_fixed.py` | 强化 Prompt + 后处理 |
| OpenAI GPT-4 | 可用 function calling | 原生支持良好 |
| Claude | 可用 function calling | 原生支持良好 |
| 跨平台兼容 | `output_parser_fixed.py` | 通用方案 |

---

## 相关文件

- `output_parser_fixed.py` - 推荐方案
- `output_parser_function_calling.py` - 失败测试记录
- `output_parser_function_calling_v2.py` - 改进尝试（仍可能失败）
- `zhipu_function_calling_examples.py` - 其他场景测试（某些场景可能工作）

---

## 参考链接

- 智谱官方文档：https://docs.bigmodel.cn/cn/guide/capabilities/function-calling
- Issue 讨论：可能需要在智谱社区反馈此问题
