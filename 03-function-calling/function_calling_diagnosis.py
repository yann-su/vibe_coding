"""
Function Calling 诊断工具
排查为什么 GLM-4.7 的 Function Calling 不稳定
并尝试不同的解决方案
"""

import urllib.request
import json
from typing import Dict, List, Optional

API_KEY = "9c575fbc0d714aa5a2ed2b1fec1359ec.Ve1Wz7QLlCHCsTiG"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def call_api(messages: List[Dict], tools: Optional[List[Dict]] = None,
             tool_choice: Optional[str] = None, temperature: float = 0.1) -> Dict:
    """调用智谱 API"""
    data = {
        "model": "glm-4.7",
        "messages": messages,
        "temperature": temperature
    }

    if tools:
        data["tools"] = tools
    if tool_choice:
        data["tool_choice"] = tool_choice

    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(data).encode('utf-8'),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        method='POST'
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode('utf-8'))


# ========== 诊断 1: 最简单的 Function Calling ==========
def test1_simplest():
    """测试最简单的 function calling - 只有一个必填参数"""
    print("=" * 60)
    print("诊断 1: 最简单的 Function Calling")
    print("=" * 60)

    tools = [{
        "type": "function",
        "function": {
            "name": "say_hello",
            "description": "打个招呼",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "名字"
                    }
                },
                "required": ["name"]
            }
        }
    }]

    messages = [{"role": "user", "content": "向小明打招呼"}]

    print("\n发送请求...")
    response = call_api(messages, tools)

    message = response["choices"][0]["message"]

    if "tool_calls" in message:
        print("✅ 触发了 function calling")
        for tc in message["tool_calls"]:
            args = tc["function"]["arguments"]
            print(f"参数: {args}")
            if args and json.loads(args).get("name"):
                print("✅ 参数有值")
            else:
                print("❌ 参数为空")
    else:
        print("⚠️ 没有触发 function calling")
        print(f"直接回复: {message.get('content', '')}")


# ========== 诊断 2: 对比有无 tool_choice ==========
def test2_tool_choice():
    """对比 tool_choice="auto" vs 不传"""
    print("\n" + "=" * 60)
    print("诊断 2: tool_choice 参数影响")
    print("=" * 60)

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市"}
                },
                "required": ["city"]
            }
        }
    }]

    messages = [{"role": "user", "content": "北京天气"}]

    # 测试 A: 带 tool_choice
    print("\n测试 A: 带 tool_choice='auto'")
    resp_a = call_api(messages, tools, tool_choice="auto")
    msg_a = resp_a["choices"][0]["message"]

    if "tool_calls" in msg_a:
        args = msg_a["tool_calls"][0]["function"]["arguments"]
        print(f"参数: {args}")

    # 测试 B: 不带 tool_choice
    print("\n测试 B: 不带 tool_choice")
    resp_b = call_api(messages, tools, tool_choice=None)
    msg_b = resp_b["choices"][0]["message"]

    if "tool_calls" in msg_b:
        args = msg_b["tool_calls"][0]["function"]["arguments"]
        print(f"参数: {args}")
    else:
        print(f"直接回复: {msg_b.get('content', '')[:100]}")


# ========== 诊断 3: 温度参数影响 ==========
def test3_temperature():
    """测试不同 temperature 的影响"""
    print("\n" + "=" * 60)
    print("诊断 3: Temperature 影响")
    print("=" * 60)

    tools = [{
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "表达式"}
                },
                "required": ["expression"]
            }
        }
    }]

    messages = [{"role": "user", "content": "计算 1+1"}]

    for temp in [0.0, 0.1, 0.5, 0.9]:
        print(f"\nTemperature = {temp}")
        resp = call_api(messages, tools, temperature=temp)
        msg = resp["choices"][0]["message"]

        if "tool_calls" in msg:
            args = msg["tool_calls"][0]["function"]["arguments"]
            print(f"  参数: {args}")


# ========== 诊断 4: 参数复杂度测试 ==========
def test4_complexity():
    """测试不同参数复杂度"""
    print("\n" + "=" * 60)
    print("诊断 4: 参数复杂度测试")
    print("=" * 60)

    # 简单参数
    print("\n4.1 简单参数（1个字符串）")
    tools_simple = [{
        "type": "function",
        "function": {
            "name": "test",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"]
            }
        }
    }]
    resp = call_api([{"role": "user", "content": "测试值为 hello"}], tools_simple)
    msg = resp["choices"][0]["message"]
    if "tool_calls" in msg:
        print(f"  结果: {msg['tool_calls'][0]['function']['arguments']}")

    # 多个参数
    print("\n4.2 多个参数（3个字段）")
    tools_multi = [{
        "type": "function",
        "function": {
            "name": "test",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "integer"},
                    "c": {"type": "string"}
                },
                "required": ["a", "b", "c"]
            }
        }
    }]
    resp = call_api([{"role": "user", "content": "a=hello, b=123, c=world"}], tools_multi)
    msg = resp["choices"][0]["message"]
    if "tool_calls" in msg:
        print(f"  结果: {msg['tool_calls'][0]['function']['arguments']}")

    # 嵌套对象
    print("\n4.3 嵌套对象")
    tools_nested = [{
        "type": "function",
        "function": {
            "name": "test",
            "parameters": {
                "type": "object",
                "properties": {
                    "person": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"}
                        }
                    }
                },
                "required": ["person"]
            }
        }
    }]
    resp = call_api([{"role": "user", "content": "person name is John"}], tools_nested)
    msg = resp["choices"][0]["message"]
    if "tool_calls" in msg:
        print(f"  结果: {msg['tool_calls'][0]['function']['arguments']}")


# ========== 诊断 5: Prompt 明确指导 ==========
def test5_explicit_prompt():
    """测试 Prompt 明确指导填写参数"""
    print("\n" + "=" * 60)
    print("诊断 5: Prompt 明确指导")
    print("=" * 60)

    tools = [{
        "type": "function",
        "function": {
            "name": "create_movie",
            "description": "创建电影信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "电影名称"},
                    "year": {"type": "integer", "description": "年份"},
                    "director": {"type": "string", "description": "导演"}
                },
                "required": ["name", "year", "director"]
            }
        }
    }]

    # 测试 A: 普通 prompt
    print("\n测试 A: 普通 prompt")
    messages_a = [{"role": "user", "content": "推荐一部2023年的科幻电影"}]
    resp_a = call_api(messages_a, tools)
    msg_a = resp_a["choices"][0]["message"]
    if "tool_calls" in msg_a:
        print(f"  参数: {msg_a['tool_calls'][0]['function']['arguments']}")

    # 测试 B: 明确指导的 prompt
    print("\n测试 B: 明确指导填写参数")
    messages_b = [{"role": "user", "content": """推荐一部2023年的科幻电影。

请使用 create_movie 函数，并填写以下字段：
- name: 电影名称（如：流浪地球2）
- year: 上映年份（如：2023）
- director: 导演名字（如：郭帆）

请确保填写具体的值，不要留空。"""}]
    resp_b = call_api(messages_b, tools)
    msg_b = resp_b["choices"][0]["message"]
    if "tool_calls" in msg_b:
        print(f"  参数: {msg_b['tool_calls'][0]['function']['arguments']}")


# ========== 解决方案：模拟 Function Calling ==========
def solution_simulated_function_calling():
    """
    解决方案：用 Prompt 模拟 Function Calling 的效果
    既获得结构化输出，又避免 GLM-4.7 的 bug
    """
    print("\n" + "=" * 60)
    print("💡 解决方案：Prompt 模拟 Function Calling")
    print("=" * 60)

    # 定义工具（只在 Prompt 中描述，不传入 API）
    function_def = """
你可以使用以下工具：

工具名: create_movie
描述: 创建电影信息
参数:
  - name (string, 必填): 电影名称
  - year (integer, 必填): 上映年份
  - director (string, 必填): 导演
  - rating (number, 必填): 评分

请使用 JSON 格式调用工具：
{"tool": "create_movie", "arguments": {"name": "...", "year": ..., ...}}
"""

    messages = [
        {"role": "system", "content": "你是一个助手。" + function_def},
        {"role": "user", "content": "推荐一部2023年的高分科幻电影"}
    ]

    print("\n发送请求（使用 Prompt 模拟）...")
    response = call_api(messages)
    content = response["choices"][0]["message"]["content"]

    print(f"\n模型回复:\n{content}")

    # 解析 JSON
    try:
        # 尝试从回复中提取 JSON
        import re
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            data = json.loads(json_match.group())
            print(f"\n✅ 解析成功: {data}")
    except:
        print("\n❌ 解析失败")


# ========== 主函数 ==========
def main():
    print("🔧 Function Calling 诊断工具")
    print("排查 GLM-4.7 Function Calling 不稳定的原因\n")

    try:
        test1_simplest()
    except Exception as e:
        print(f"诊断 1 错误: {e}")

    try:
        test2_tool_choice()
    except Exception as e:
        print(f"诊断 2 错误: {e}")

    try:
        test3_temperature()
    except Exception as e:
        print(f"诊断 3 错误: {e}")

    try:
        test4_complexity()
    except Exception as e:
        print(f"诊断 4 错误: {e}")

    try:
        test5_explicit_prompt()
    except Exception as e:
        print(f"诊断 5 错误: {e}")

    try:
        solution_simulated_function_calling()
    except Exception as e:
        print(f"解决方案错误: {e}")

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
