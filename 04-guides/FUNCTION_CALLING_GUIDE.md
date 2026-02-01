# LangChain + 智谱 Function Calling 使用指南

## 关键发现

通过大量测试，发现 **LangChain 的 `with_structured_output()` 对智谱 API 有兼容性问题**，但 **`bind_tools()` 方式正常工作**。

---

## ✅ 推荐方案

### 方案 1: @tool + bind_tools（最推荐）

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# 初始化
llm = ChatOpenAI(
    model="glm-4.7",
    openai_api_key="...",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
)

# 1. 定义工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city}今天晴天，25度"

# 2. 绑定工具
llm_with_tools = llm.bind_tools([get_weather])

# 3. 使用
messages = [HumanMessage(content="北京天气怎么样？")]
response = llm_with_tools.invoke(messages)

if response.tool_calls:
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"调用: {tool_name}({tool_args})")

        # 执行工具
        if tool_name == "get_weather":
            result = get_weather.invoke(tool_args)
            messages.append(response)
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    # 获取最终回复
    final = llm_with_tools.invoke(messages)
    print(final.content)
```

### 方案 2: Pydantic 结构化输出（使用 bind_tools）

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class Movie(BaseModel):
    """电影信息"""
    name: str = Field(description="电影名称")
    year: int = Field(description="年份")
    director: str = Field(description="导演")
    rating: float = Field(description="评分")

# 使用 Pydantic 作为工具参数
@tool(args_schema=Movie)
def create_movie(name: str, year: int, director: str, rating: float) -> str:
    """创建电影信息"""
    return f"《{name}》({year})"

# 绑定并使用
llm_with_movie = llm.bind_tools([create_movie])
messages = [HumanMessage(content="推荐一部2023年的高分科幻电影")]
response = llm_with_movie.invoke(messages)

if response.tool_calls:
    # 直接转为 Pydantic 对象
    movie = Movie(**response.tool_calls[0]["args"])
    print(f"电影: {movie.name}")
```

---

## ❌ 避免使用

### 不推荐的方案

```python
# ❌ 不推荐：with_structured_output 对智谱 API 有问题
structured_llm = llm.with_structured_output(Movie)
result = structured_llm.invoke("推荐一部电影")
# 可能返回空值或缺失字段
```

**问题**：`with_structured_output` 内部实现可能对智谱 API 的适配不完善。

---

## 对比总结

| 方式 | 代码 | 智谱 GLM-4.7 | 说明 |
|------|------|--------------|------|
| **bind_tools + @tool** | `llm.bind_tools([tool])` | ✅ **正常** | **推荐** |
| **bind_tools + Pydantic** | `llm.bind_tools([tool])` | ✅ **正常** | 结构化输出 |
| with_structured_output | `llm.with_structured_output(Model)` | ❌ **失败** | 避免使用 |
| 原生 API | `requests.post(tools=[...])` | ✅ **正常** | 最底层 |

---

## 完整工具执行流程

```python
def execute_with_tools(llm, tools_dict, user_input: str) -> str:
    """完整的工具调用流程"""
    messages = [HumanMessage(content=user_input)]

    # 第一轮：获取工具调用
    response = llm.invoke(messages)
    messages.append(response)

    # 执行工具
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # 执行
            if tool_name in tools_dict:
                result = tools_dict[tool_name].invoke(tool_args)
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                ))

        # 第二轮：获取最终回复
        final = llm.invoke(messages)
        return final.content

    return response.content
```

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `langchain_function_calling_guide.py` | 推荐方案代码示例 |
| `langchain_function_calling_demo.py` | 完整演示（含对比测试） |
| `zhipu_function_calling_examples.py` | 原生 API 示例 |

---

## 结论

对于 **智谱 GLM-4.7 + LangChain**：

1. ✅ **使用 `bind_tools()` + `@tool` 装饰器** - 最稳定
2. ✅ **使用 `bind_tools()` + Pydantic 模型** - 结构化输出
3. ❌ **避免 `with_structured_output()`** - 有兼容性问题
4. ✅ **原生 API 调用** - 最底层，100% 可控

**理论 vs 实际**：
- 理论上 `with_structured_output` 应该更稳定（API 层约束）
- 实际上 LangChain 对该 API 的适配有 bug，使用 `bind_tools` 更可靠
