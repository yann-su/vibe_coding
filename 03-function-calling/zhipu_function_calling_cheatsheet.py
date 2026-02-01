"""
智谱 Function Calling 速查表 / Cheat Sheet
最简化的快速参考
"""

# ========== 1. 最简定义 ==========

# 定义工具
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",           # 函数名
        "description": "获取天气",        # 函数描述
        "parameters": {                   # 参数定义 (JSON Schema)
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名"
                },
                "date": {
                    "type": "string",
                    "description": "日期"
                }
            },
            "required": ["city"]          # 必填字段
        }
    }
}]

# ========== 2. 调用 API ==========

import requests
import json

API_KEY = "your-api-key"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# 请求
data = {
    "model": "glm-4.7",
    "messages": [{"role": "user", "content": "北京天气"}],
    "tools": tools,                       # 传入工具
    "tool_choice": "auto"                 # 自动选择（目前只支持 auto）
}

response = requests.post(
    BASE_URL,
    headers={"Authorization": f"Bearer {API_KEY}"},
    json=data
).json()

# ========== 3. 处理响应 ==========

message = response["choices"][0]["message"]

# 检查是否有函数调用
if "tool_calls" in message:
    for tool_call in message["tool_calls"]:
        # 获取调用信息
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        tool_call_id = tool_call["id"]

        # 执行你的函数
        result = your_function(**arguments)

        # 返回结果给模型
        messages.append({
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call_id
        })

    # 再次调用获取最终回复
    final = requests.post(BASE_URL, headers=headers, json={
        "model": "glm-4.7",
        "messages": messages,
        "tools": tools
    }).json()

# ========== 4. 完整流程模板 ==========

def function_calling_template():
    """完整流程模板"""

    # 1. 定义工具
    tools = [{...}]

    # 2. 初始化对话
    messages = [{"role": "user", "content": "用户问题"}]

    # 3. 第一次调用（获取函数调用请求）
    response = call_api(messages, tools)
    message = response["choices"][0]["message"]
    messages.append(message)  # 保存助手回复

    # 4. 执行函数
    if "tool_calls" in message:
        for tool_call in message["tool_calls"]:
            # 解析参数
            func_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])

            # 执行实际函数
            result = execute_function(func_name, args)

            # 添加到消息历史
            messages.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call["id"]
            })

        # 5. 第二次调用（获取最终回复）
        final = call_api(messages, tools)
        return final["choices"][0]["message"]["content"]

# ========== 5. 参数类型参考 ==========

SIMPLE_PARAMETER = {
    "type": "string",           # string | integer | number | boolean | array | object
    "description": "参数描述"
}

ENUM_PARAMETER = {
    "type": "string",
    "enum": ["value1", "value2"],  # 限制可选值
    "description": "参数描述"
}

ARRAY_PARAMETER = {
    "type": "array",
    "items": {
        "type": "string"        # 数组元素类型
    },
    "description": "列表参数"
}

OBJECT_PARAMETER = {
    "type": "object",
    "properties": {
        "field1": {"type": "string"},
        "field2": {"type": "number"}
    },
    "required": ["field1"]       # 对象内的必填字段
}

# ========== 6. 常见错误 ==========

ERRORS = {
    "模型不调用函数": "检查 description 是否清晰",
    "参数缺失": "检查 required 字段标记",
    "参数格式错误": "检查参数类型的定义",
    "tool_choice 不支持": "智谱只支持 'auto'",
}

# ========== 7. 最佳实践 ==========

BEST_PRACTICES = """
✅ 函数名要清晰：get_weather, search_products
✅ 参数要有描述：每个字段都写 description
✅ 限制可选值：用 enum 限定参数范围
✅ 标记必填项：用 required 确保完整性
✅ 处理不调用：模型可能直接回复而不调用函数
✅ 安全验证：执行函数前验证参数
"""

if __name__ == "__main__":
    print("智谱 Function Calling 速查表")
    print(BEST_PRACTICES)
