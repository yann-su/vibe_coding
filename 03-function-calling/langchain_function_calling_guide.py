"""
LangChain + 智谱 Function Calling 正确使用指南

关键结论：
- ✅ 使用 bind_tools() + @tool 装饰器
- ❌ 避免使用 with_structured_output()
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from typing import List, Optional
import json

# 初始化
llm = ChatOpenAI(
    model="glm-4.7",
    openai_api_key="9c575fbc0d714aa5a2ed2b1fec1359ec.Ve1Wz7QLlCHCsTiG",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
)


# ========== ✅ 推荐方案 1: @tool + bind_tools ==========
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city}今天晴天，25度"


@tool
def calculate(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except:
        return "计算错误"


# 绑定工具
llm_with_tools = llm.bind_tools([get_weather, calculate])

# 使用
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
            print(f"结果: {result}")


# ========== ✅ 推荐方案 2: 结构化输出（使用 bind_tools） ==========
from langchain_core.pydantic_v1 import BaseModel, Field

class Movie(BaseModel):
    """电影信息"""
    name: str = Field(description="电影名称")
    year: int = Field(description="年份")
    director: str = Field(description="导演")
    rating: float = Field(description="评分")

# 使用 Pydantic 模型作为工具
@tool(args_schema=Movie)
def create_movie(name: str, year: int, director: str, rating: float) -> str:
    """创建电影信息"""
    return f"电影《{name}》({year})，导演：{director}，评分：{rating}"

# 绑定
llm_with_movie = llm.bind_tools([create_movie])

# 使用
messages = [HumanMessage(content="推荐一部2023年的高分科幻电影")]
response = llm_with_movie.invoke(messages)

if response.tool_calls:
    tool_call = response.tool_calls[0]
    print(f"提取的电影信息: {tool_call['args']}")
    # 可以直接转为 Pydantic 对象
    movie = Movie(**tool_call["args"])
    print(f"电影: {movie.name}, 年份: {movie.year}")


# ========== ❌ 不推荐方案：with_structured_output ==========
"""
# 这种方式对智谱 API 有兼容性问题
structured_llm = llm.with_structured_output(Movie)  # ❌ 可能失败
result = structured_llm.invoke("推荐一部电影")  # 可能返回空值
"""


# ========== 完整工具执行流程 ==========
def execute_with_tools(llm, tools, user_input: str) -> str:
    """
    完整的工具调用流程

    Args:
        llm: 已绑定工具的 LLM
        tools: 工具字典 {name: tool}
        user_input: 用户输入

    Returns:
        最终回复
    """
    messages = [HumanMessage(content=user_input)]

    # 第一轮：获取工具调用
    response = llm.invoke(messages)
    messages.append(response)

    # 执行工具
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # 查找并执行工具
            if tool_name in tools:
                result = tools[tool_name].invoke(tool_args)
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_id
                ))

        # 第二轮：获取最终回复
        final = llm.invoke(messages)
        return final.content

    return response.content


# 使用示例
tools_map = {"get_weather": get_weather, "calculate": calculate}
llm_tools = llm.bind_tools([get_weather, calculate])

result = execute_with_tools(llm_tools, tools_map, "北京今天多少度？25乘以4呢？")
print(result)
