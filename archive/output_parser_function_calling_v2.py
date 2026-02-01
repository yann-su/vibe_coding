"""
智谱 API Function Calling 改进版
通过 prompt 指导 + function calling 约束，绕过 GLM-4.7 的 bug
"""

import requests
import json
from typing import List, Optional, Type
from pydantic import BaseModel, Field


def call_glm_with_function_v2(
    prompt: str,
    pydantic_model: Type[BaseModel],
    api_key: str = "9c575fbc0d714aa5a2ed2b1fec1359ec.Ve1Wz7QLlCHCsTiG"
) -> Optional[BaseModel]:
    """
    改进版：使用 prompt 明确指导模型生成参数，同时用 function calling 约束格式
    """
    schema = pydantic_model.model_json_schema()

    # 从 schema 中提取字段信息
    properties = schema.get('properties', {})
    required = schema.get('required', [])

    # 构建字段描述
    fields_desc = []
    for field_name, field_info in properties.items():
        desc = field_info.get('description', field_name)
        field_type = field_info.get('type', 'any')
        fields_desc.append(f"  - {field_name} ({field_type}): {desc}")

    fields_text = '\n'.join(fields_desc)

    tool_name = f"extract_{pydantic_model.__name__.lower()}"

    # 构建工具定义
    tools = [{
        "type": "function",
        "function": {
            "name": tool_name,
            "description": f"提取{pydantic_model.__name__}信息。使用此函数时，必须在arguments中包含以下所有字段的完整值：\n{fields_text}",
            "parameters": schema
        }
    }]

    # 关键改进：在 prompt 中明确要求模型生成具体参数值
    enhanced_prompt = f"""{prompt}

【重要】你需要使用 {tool_name} 函数来返回结果。
请在函数调用参数中填写以下字段的具体值：
{fields_text}

请确保填写所有必填字段，不要留空。"""

    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "glm-4.7",
        "messages": [{"role": "user", "content": enhanced_prompt}],
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

        print(f"📡 API 响应:")
        message = result["choices"][0]["message"]

        # 提取参数
        if "tool_calls" in message and message["tool_calls"]:
            tool_call = message["tool_calls"][0]
            arguments = tool_call["function"]["arguments"]

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            print(f"\n📝 提取的参数:")
            print(json.dumps(arguments, indent=2, ensure_ascii=False))

            # 验证参数完整性
            missing_fields = [f for f in required if f not in arguments or not arguments[f]]
            if missing_fields:
                print(f"\n⚠️ 缺少字段: {missing_fields}")
                # 尝试用 content 补充
                if message.get("content"):
                    print(f"\n💡 尝试从 content 解析...")
                    arguments = extract_from_content(message["content"], properties, arguments)
                    missing_fields = [f for f in required if f not in arguments or not arguments[f]]
                    if not missing_fields:
                        print("✅ 补充成功!")

            if not missing_fields:
                return pydantic_model(**arguments)
            else:
                print(f"❌ 仍缺少字段: {missing_fields}")
                return None
        else:
            # 没有 tool_calls，尝试从 content 解析
            print(f"\n⚠️ 没有 function calling，尝试解析 content:")
            content = message.get("content", "")
            print(content[:500])
            return None

    except Exception as e:
        print(f"❌ 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_from_content(content: str, properties: dict, existing: dict) -> dict:
    """尝试从 content 中提取字段值"""
    result = existing.copy()

    # 简单启发式：从 reasoning_content 或 content 中提取
    # 这里可以添加更复杂的解析逻辑

    return result


# ========== 方法 2: 流式生成参数 ==========
def call_glm_streaming_params(
    prompt: str,
    pydantic_model: Type[BaseModel],
    api_key: str = "9c575fbc0d714aa5a2ed2b1fec1359ec.Ve1Wz7QLlCHCsTiG"
) -> Optional[BaseModel]:
    """
    方法 2: 不使用 tool_choice 强制，让模型自己选择
    有时模型在自由选择时表现更好
    """
    schema = pydantic_model.model_json_schema()
    tool_name = f"extract_{pydantic_model.__name__.lower()}"

    tools = [{
        "type": "function",
        "function": {
            "name": tool_name,
            "description": f"提取{pydantic_model.__name__}信息",
            "parameters": schema
        }
    }]

    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "glm-4.7",
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        # 不指定 tool_choice，让模型自己决定
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()

        message = result["choices"][0]["message"]

        print(f"🎯 Finish reason: {result['choices'][0].get('finish_reason')}")

        if "tool_calls" in message and message["tool_calls"]:
            tool_call = message["tool_calls"][0]
            arguments = tool_call["function"]["arguments"]

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            print(f"\n✅ 成功使用 function calling!")
            print(f"📝 参数: {json.dumps(arguments, indent=2, ensure_ascii=False)}")

            return pydantic_model(**arguments)
        else:
            print(f"\n⚠️ 模型选择不使用 function calling")
            print(f"Content: {message.get('content', '无')[:200]}")
            return None

    except Exception as e:
        print(f"❌ 失败: {e}")
        return None


# ========== 方法 3: 分两步走 ==========
def call_glm_two_step(
    prompt: str,
    pydantic_model: Type[BaseModel],
    api_key: str = "9c575fbc0d714aa5a2ed2b1fec1359ec.Ve1Wz7QLlCHCsTiG"
) -> Optional[BaseModel]:
    """
    方法 3: 第一步让模型思考，第二步强制格式化
    """
    schema = pydantic_model.model_json_schema()

    # 第一步：让模型生成内容（普通调用）
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 构建字段描述
    fields = '\n'.join([f"- {k}: {v.get('description', k)}"
                       for k, v in schema['properties'].items()])

    step1_prompt = f"""{prompt}

请提供以下信息：
{fields}

请以纯文本形式列出这些信息。"""

    data1 = {
        "model": "glm-4.7",
        "messages": [{"role": "user", "content": step1_prompt}]
    }

    try:
        response1 = requests.post(url, headers=headers, json=data1, timeout=30)
        response1.raise_for_status()
        content = response1.json()["choices"][0]["message"]["content"]

        print(f"📝 第一步生成内容:\n{content}\n")

        # 第二步：用 function calling 格式化
        tool_name = f"format_{pydantic_model.__name__.lower()}"
        tools = [{
            "type": "function",
            "function": {
                "name": tool_name,
                "description": "将文本信息格式化为结构化数据",
                "parameters": schema
            }
        }]

        step2_prompt = f"""将以下信息格式化为 JSON：
{content}

请使用 {tool_name} 函数返回格式化后的数据。"""

        data2 = {
            "model": "glm-4.7",
            "messages": [{"role": "user", "content": step2_prompt}],
            "tools": tools,
            "tool_choice": {
                "type": "function",
                "function": {"name": tool_name}
            }
        }

        response2 = requests.post(url, headers=headers, json=data2, timeout=30)
        response2.raise_for_status()
        result = response2.json()

        message = result["choices"][0]["message"]
        if "tool_calls" in message and message["tool_calls"]:
            arguments = message["tool_calls"][0]["function"]["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            print(f"✅ 第二步格式化成功!")
            print(f"📝 结果: {json.dumps(arguments, indent=2, ensure_ascii=False)}")

            return pydantic_model(**arguments)
        else:
            print(f"❌ 第二步失败")
            return None

    except Exception as e:
        print(f"❌ 失败: {e}")
        return None


# ========== 测试 ==========
def demo_all_methods():
    """对比三种方法"""

    class Movie(BaseModel):
        name: str = Field(description="电影名称")
        year: int = Field(description="上映年份")
        director: str = Field(description="导演")
        rating: float = Field(description="评分 0-10")
        genres: List[str] = Field(description="电影类型列表")

    prompt = "推荐一部2023年的高分科幻电影"

    print("=" * 60)
    print("方法 1: Prompt 指导 + 强制 tool_choice")
    print("=" * 60)
    result1 = call_glm_with_function_v2(prompt, Movie)
    if result1:
        print(f"\n✅ 成功: {result1.name} ({result1.year})")

    print("\n" + "=" * 60)
    print("方法 2: 让模型自主选择是否使用 tool")
    print("=" * 60)
    result2 = call_glm_streaming_params(prompt, Movie)
    if result2:
        print(f"\n✅ 成功: {result2.name} ({result2.year})")

    print("\n" + "=" * 60)
    print("方法 3: 两步走（生成 + 格式化）")
    print("=" * 60)
    result3 = call_glm_two_step(prompt, Movie)
    if result3:
        print(f"\n✅ 成功: {result3.name} ({result3.year})")


if __name__ == "__main__":
    print("🔧 智谱 API Function Calling 改进版")
    print("测试不同方法绕过 GLM-4.7 的 function calling bug\n")

    demo_all_methods()
