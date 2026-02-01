"""
LangChain + Function Calling 完整示例
结合 LangChain 的便利性和 Function Calling 的可靠性
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, tool
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import json

# ========== 初始化 LLM ==========
llm = ChatOpenAI(
    model="glm-4.7",
    openai_api_key="9c575fbc0d714aa5a2ed2b1fec1359ec.Ve1Wz7QLlCHCsTiG",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
)


# ========== 方法 1: 使用 @tool 装饰器定义工具 ==========
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 模拟天气查询
    weather_data = {
        "北京": {"temp": 25, "condition": "晴天"},
        "上海": {"temp": 28, "condition": "多云"},
        "广州": {"temp": 32, "condition": "小雨"}
    }
    data = weather_data.get(city, {"temp": 20, "condition": "未知"})
    return f"{city}今天{data['condition']}，温度{data['temp']}度"


@tool
def calculate(expression: str) -> str:
    """执行数学计算"""
    try:
        # 安全计算
        allowed_chars = set('0123456789+-*/(). ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"{expression} = {result}"
        else:
            return "表达式包含非法字符"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def search_movies(genre: str, year: Optional[int] = None) -> str:
    """搜索电影信息"""
    movies_db = {
        "科幻": ["流浪地球2", "星际穿越", "盗梦空间"],
        "动作": ["战狼2", "红海行动", "疾速追杀"],
        "喜剧": ["夏洛特烦恼", "西虹市首富", "你好李焕英"]
    }
    movies = movies_db.get(genre, [])
    year_str = f"{year}年" if year else ""
    return f"{year_str}{genre}电影推荐: {', '.join(movies)}"


def demo1_tools_decorator():
    """示例 1: 使用 @tool 装饰器"""
    print("=" * 60)
    print("示例 1: @tool 装饰器定义工具")
    print("=" * 60)

    # 绑定工具到 LLM
    tools = [get_weather, calculate, search_movies]
    llm_with_tools = llm.bind_tools(tools)

    # 测试 1: 天气查询
    print("\n--- 测试 1: 天气查询 ---")
    messages = [HumanMessage(content="北京今天天气怎么样？")]
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        print(f"🔧 工具调用: {response.tool_calls}")

        # 执行工具
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "get_weather":
                result = get_weather.invoke(tool_args)
                print(f"📊 结果: {result}")

                # 添加工具结果到对话
                messages.append(response)
                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

        # 获取最终回复
        final_response = llm_with_tools.invoke(messages)
        print(f"🤖 最终回复: {final_response.content}")

    # 测试 2: 数学计算
    print("\n--- 测试 2: 数学计算 ---")
    messages = [HumanMessage(content="计算 123 * 456")]
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        result = calculate.invoke(tool_call["args"])
        print(f"🔧 调用: {tool_call['name']}({tool_call['args']})")
        print(f"📊 结果: {result}")


# ========== 方法 2: 使用 Pydantic 定义结构化输出 ==========
class Movie(BaseModel):
    """电影信息"""
    name: str = Field(description="电影名称")
    year: int = Field(description="上映年份")
    director: str = Field(description="导演")
    rating: float = Field(description="评分")
    genres: List[str] = Field(description="类型列表")


class Book(BaseModel):
    """书籍信息"""
    title: str = Field(description="书名")
    author: str = Field(description="作者")
    pages: int = Field(description="页数")
    isbn: Optional[str] = Field(description="ISBN", default=None)


def demo2_pydantic_structured():
    """示例 2: 使用 Pydantic + with_structured_output"""
    print("\n" + "=" * 60)
    print("示例 2: Pydantic 结构化输出")
    print("=" * 60)

    # 方法 A: 使用 with_structured_output（原生 Function Calling）
    print("\n--- 方法 A: with_structured_output ---")

    # 注意：需要指定 method="function_calling" 避免警告
    structured_llm = llm.with_structured_output(Movie, method="function_calling")

    prompt = "推荐一部2023年的高分科幻电影"
    result = structured_llm.invoke(prompt)

    print(f"✅ 返回类型: {type(result)}")
    print(f"🎬 电影: {result.name}")
    print(f"📅 年份: {result.year}")
    print(f"🎭 导演: {result.director}")
    print(f"⭐ 评分: {result.rating}")
    print(f"🏷️ 类型: {result.genres}")

    # 方法 B: 使用 PydanticOutputParser（传统方式，作为对比）
    print("\n--- 方法 B: PydanticOutputParser（对比） ---")

    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import PromptTemplate

    parser = PydanticOutputParser(pydantic_object=Book)

    prompt = PromptTemplate(
        template="""推荐一本书。
{format_instructions}
要求: {requirement}""",
        input_variables=["requirement"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser
    result = chain.invoke({"requirement": "经典的科幻小说"})

    print(f"📚 书名: {result.title}")
    print(f"✍️ 作者: {result.author}")
    print(f"📖 页数: {result.pages}")


# ========== 方法 3: 复杂工具组合 ==========
def demo3_complex_tools():
    """示例 3: 复杂场景 - 多工具组合"""
    print("\n" + "=" * 60)
    print("示例 3: 复杂工具组合")
    print("=" * 60)

    @tool
    def get_current_time() -> str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @tool
    def get_user_location(user_id: str) -> str:
        """获取用户位置"""
        locations = {"user1": "北京", "user2": "上海", "user3": "广州"}
        return locations.get(user_id, "未知")

    @tool
    def send_notification(user_id: str, message: str) -> str:
        """发送通知"""
        return f"已向用户 {user_id} 发送通知: {message}"

    tools = [get_current_time, get_user_location, send_notification]
    llm_with_tools = llm.bind_tools(tools)

    # 复杂查询：需要时间、位置，然后发送通知
    print("\n--- 场景: 智能助手任务 ---")
    messages = [
        HumanMessage(content="现在几点了？我在哪里？然后给我发送一条天气提醒")
    ]

    # 第一轮：获取工具调用
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # 执行所有工具调用
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"\n🔧 调用: {tool_name}({tool_args})")

            # 执行对应的工具
            for tool in tools:
                if tool.name == tool_name:
                    result = tool.invoke(tool_args)
                    print(f"📊 结果: {result}")

                    messages.append(ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    ))

    # 第二轮：获取最终回复
    final_response = llm_with_tools.invoke(messages)
    print(f"\n🤖 最终回复:\n{final_response.content}")


# ========== 方法 4: LCEL 链式调用 + Function Calling ==========
def demo4_lcel_with_tools():
    """示例 4: LCEL 链 + Function Calling"""
    print("\n" + "=" * 60)
    print("示例 4: LCEL 链 + Function Calling")
    print("=" * 60)

    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate

    # 定义工具
    @tool
    def multiply(a: int, b: int) -> int:
        """乘法运算"""
        return a * b

    @tool
    def add(a: int, b: int) -> int:
        """加法运算"""
        return a + b

    # 创建提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个数学助手，可以使用工具进行计算。"),
        ("human", "{question}")
    ])

    # 绑定工具
    tools = [multiply, add]
    llm_with_tools = llm.bind_tools(tools)

    # 构建链
    chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | llm_with_tools
    )

    # 执行
    print("\n--- 测试: 25 乘以 4 加 10 等于多少 ---")
    result = chain.invoke("25 乘以 4 加 10 等于多少")

    print(f"📝 中间结果: {result}")

    if result.tool_calls:
        for tc in result.tool_calls:
            print(f"🔧 调用: {tc['name']}({tc['args']})")


# ========== 方法 5: 自动执行工具回调 ==========
def demo5_auto_execute():
    """示例 5: 自动执行工具并获取最终回复"""
    print("\n" + "=" * 60)
    print("示例 5: 自动工具执行")
    print("=" * 60)

    # 定义工具
    @tool
    def get_stock_price(symbol: str) -> str:
        """获取股票价格"""
        stocks = {"AAPL": "150.5", "GOOGL": "2800.3", "TSLA": "750.2"}
        price = stocks.get(symbol.upper(), "未知")
        return f"{symbol} 当前价格: ${price}"

    @tool
    def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
        """货币转换"""
        rates = {"USD": 7.2, "EUR": 7.8, "JPY": 0.05}
        rate = rates.get(to_currency.upper(), 1)
        result = amount * rate
        return f"{amount} {from_currency} = {result} {to_currency}"

    tools = [get_stock_price, convert_currency]
    tools_map = {tool.name: tool for tool in tools}

    # 创建带工具的 LLM
    llm_with_tools = llm.bind_tools(tools)

    def auto_execute_tools(user_input: str) -> str:
        """自动执行工具的完整流程"""
        messages = [HumanMessage(content=user_input)]

        # 第一轮：获取工具调用
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # 如果有工具调用，执行它们
        if response.tool_calls:
            print(f"🔧 需要调用 {len(response.tool_calls)} 个工具")

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(f"  执行: {tool_name}({tool_args})")

                # 执行工具
                if tool_name in tools_map:
                    result = tools_map[tool_name].invoke(tool_args)
                    print(f"  结果: {result}")

                    messages.append(ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    ))

            # 第二轮：获取最终回复
            final_response = llm_with_tools.invoke(messages)
            return final_response.content
        else:
            return response.content

    # 测试
    print("\n--- 测试 1: 股票价格 ---")
    result = auto_execute_tools("苹果股票价格是多少？")
    print(f"🤖 回复: {result}")

    print("\n--- 测试 2: 货币转换 ---")
    result = auto_execute_tools("100 美元等于多少人民币？")
    print(f"🤖 回复: {result}")


# ========== 主函数 ==========
def main():
    print("🚀 LangChain + Function Calling 完整示例")
    print("结合 LangChain 的便利性和 Function Calling 的可靠性\n")

    try:
        demo1_tools_decorator()
    except Exception as e:
        print(f"示例 1 错误: {e}")
        import traceback
        traceback.print_exc()

    try:
        demo2_pydantic_structured()
    except Exception as e:
        print(f"示例 2 错误: {e}")
        import traceback
        traceback.print_exc()

    try:
        demo3_complex_tools()
    except Exception as e:
        print(f"示例 3 错误: {e}")
        import traceback
        traceback.print_exc()

    try:
        demo4_lcel_with_tools()
    except Exception as e:
        print(f"示例 4 错误: {e}")
        import traceback
        traceback.print_exc()

    try:
        demo5_auto_execute()
    except Exception as e:
        print(f"示例 5 错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
