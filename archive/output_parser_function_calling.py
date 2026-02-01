"""
智谱 API Function Calling 原生实现
不使用 LangChain 的 with_structured_output，直接调用 API
"""

import requests
import json
from typing import List, Optional, Type
from pydantic import BaseModel, Field


def call_glm_with_function(
    prompt: str,
    pydantic_model: Type[BaseModel],
    api_key: str = "9c575fbc0d714aa5a2ed2b1fec1359ec.Ve1Wz7QLlCHCsTiG"
) -> Optional[BaseModel]:
    """
    使用智谱 API 的 Function Calling 能力
    """
    # 1. 从 Pydantic 模型生成 JSON Schema
    schema = pydantic_model.model_json_schema()

    # 2. 构建工具定义
    tool_name = f"extract_{pydantic_model.__name__.lower()}"
    tools = [{
        "type": "function",
        "function": {
            "name": tool_name,
            "description": f"提取{pydantic_model.__name__}信息",
            "parameters": schema
        }
    }]

    # 3. 调用智谱 API
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "glm-4.7",
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "tool_choice": {
            "type": "function",
            "function": {"name": tool_name}
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()

        print(f"📡 API 原始响应:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # 4. 提取函数调用参数
        message = result["choices"][0]["message"]

        # 检查是否有 tool_calls
        if "tool_calls" in message and message["tool_calls"]:
            tool_call = message["tool_calls"][0]
            arguments = tool_call["function"]["arguments"]

            # 解析参数并验证
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            print(f"\n📝 提取的参数:")
            print(json.dumps(arguments, indent=2, ensure_ascii=False))

            # 5. 用 Pydantic 验证并返回
            return pydantic_model(**arguments)
        else:
            # 没有 tool_calls，返回普通内容
            print(f"\n⚠️ 没有 function calling，返回普通文本:")
            print(message.get("content", "无内容"))
            return None

    except Exception as e:
        print(f"❌ 调用失败: {e}")
        if 'response' in locals():
            print(f"响应内容: {response.text}")
        return None


# ========== 示例 1: 电影信息 ==========
def demo_movie():
    """电影信息提取 - 使用 Function Calling"""
    print("=" * 60)
    print("示例 1: Function Calling - 电影信息提取")
    print("=" * 60)

    class Movie(BaseModel):
        name: str = Field(description="电影名称")
        year: int = Field(description="上映年份")
        director: str = Field(description="导演")
        rating: float = Field(description="评分 0-10")
        genres: List[str] = Field(description="电影类型列表")

    result = call_glm_with_function(
        "推荐一部2023年的高分科幻电影",
        Movie
    )

    if result:
        print(f"\n✅ 成功!")
        print(f"  电影: {result.name}")
        print(f"  年份: {result.year}")
        print(f"  导演: {result.director}")
        print(f"  评分: {result.rating}")
        print(f"  类型: {result.genres}")


# ========== 示例 2: 人物信息（嵌套模型）==========
def demo_person():
    """人物信息提取 - 嵌套模型"""
    print("\n" + "=" * 60)
    print("示例 2: Function Calling - 嵌套模型")
    print("=" * 60)

    class Address(BaseModel):
        city: str = Field(description="城市")
        street: str = Field(description="街道")
        zipcode: str = Field(description="邮编")

    class Person(BaseModel):
        name: str = Field(description="姓名")
        age: int = Field(description="年龄")
        email: str = Field(description="邮箱")
        address: Address = Field(description="地址信息")
        hobbies: List[str] = Field(description="爱好列表")

    result = call_glm_with_function(
        "生成一个住在上海的30岁软件工程师的信息",
        Person
    )

    if result:
        print(f"\n✅ 成功!")
        print(f"  姓名: {result.name}")
        print(f"  年龄: {result.age}")
        print(f"  邮箱: {result.email}")
        print(f"  地址: {result.address.city}, {result.address.street}, {result.address.zipcode}")
        print(f"  爱好: {result.hobbies}")


# ========== 对比：LangChain vs 原生 API ==========
def compare_approaches():
    """对比不同实现方式"""
    print("\n" + "=" * 60)
    print("对比：不同实现方式")
    print("=" * 60)

    comparison = """
方式 1: LangChain with_structured_output
─────────────────────────────────────────
代码: llm.with_structured_output(Movie)
问题:
  - 对智谱 API 适配不完善
  - 可能返回 "field required" 错误
  - LangChain 内部转换可能有问题

方式 2: 本文件 - 原生 API 调用
─────────────────────────────────────────
代码: requests.post(url, json={tools: [...]})
优势:
  - 直接使用智谱 API，无中间层
  - 可以看到原始响应
  - 更容易调试问题

方式 3: 强化 Prompt（output_parser_fixed.py）
─────────────────────────────────────────
代码: 复杂 Prompt + 后处理清理
优势:
  - 不依赖 API 的 function calling
  - 跨平台兼容（OpenAI、智谱、文心等）
  - 后处理修复格式问题
    """
    print(comparison)


# ========== 主函数 ==========
def main():
    print("🔧 智谱 API Function Calling 原生实现")
    print()

    demo_movie()
    demo_person()
    compare_approaches()

    print("\n" + "=" * 60)
    print("结论:")
    print("  如果智谱 API 的 function calling 正常工作，")
    print("  应该能看到 'tool_calls' 字段和完整的参数")
    print("=" * 60)


if __name__ == "__main__":
    main()
