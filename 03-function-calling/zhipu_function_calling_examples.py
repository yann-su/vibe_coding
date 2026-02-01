"""
智谱 GLM-4 Function Calling 完整示例
基于官方文档: https://docs.bigmodel.cn/cn/guide/capabilities/function-calling

Function Calling 流程:
1. 定义工具 (tools) - 告诉模型有哪些函数可用
2. 调用 API - 让模型决定调用哪个函数
3. 执行函数 - 在你的代码中实际执行
4. 返回结果 - 将结果传回模型生成最终回复
"""

import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# ========== 配置 ==========
API_KEY = "9c575fbc0d714aa5a2ed2b1fec1359ec.Ve1Wz7QLlCHCsTiG"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def call_glm(messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
    """
    调用智谱 GLM API
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "glm-4.7",
        "messages": messages
    }

    if tools:
        data["tools"] = tools
        data["tool_choice"] = "auto"  # 让模型自动选择

    response = requests.post(BASE_URL, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    return response.json()


# ========== 示例 1: 天气查询 ==========
def example_weather():
    """
    示例 1: 天气查询工具
    """
    print("=" * 60)
    print("示例 1: 天气查询")
    print("=" * 60)

    # 1. 定义工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的当前天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，例如：北京、上海、广州"
                        },
                        "date": {
                            "type": "string",
                            "description": "日期，格式：YYYY-MM-DD，默认为今天"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    # 2. 用户提问
    messages = [
        {"role": "user", "content": "北京今天天气怎么样？需要带伞吗？"}
    ]

    # 3. 第一次调用 API
    print("\n📝 第一次调用 - 让模型决定调用哪个函数...")
    response = call_glm(messages, tools)

    message = response["choices"][0]["message"]
    messages.append({
        "role": message["role"],
        "content": message.get("content", ""),
        "tool_calls": message.get("tool_calls", [])
    })

    # 4. 检查是否有函数调用
    if "tool_calls" in message and message["tool_calls"]:
        print(f"\n🔧 模型决定调用函数:")

        for tool_call in message["tool_calls"]:
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            tool_call_id = tool_call["id"]

            print(f"  函数名: {function_name}")
            print(f"  参数: {arguments}")

            # 5. 执行实际的函数（这里模拟）
            if function_name == "get_weather":
                city = arguments.get("city")
                # 模拟天气查询结果
                weather_result = {
                    "city": city,
                    "temperature": 25,
                    "condition": "多云",
                    "humidity": "60%",
                    "rain_probability": 20,
                    "suggestion": "不需要带伞，天气不错"
                }

                # 6. 将结果返回给模型
                messages.append({
                    "role": "tool",
                    "content": json.dumps(weather_result, ensure_ascii=False),
                    "tool_call_id": tool_call_id
                })

        # 7. 第二次调用 API，让模型生成最终回复
        print("\n📝 第二次调用 - 生成最终回复...")
        final_response = call_glm(messages, tools)
        final_message = final_response["choices"][0]["message"]

        print(f"\n✅ 最终回复:\n{final_message['content']}")
    else:
        print(f"\n📝 模型直接回复:\n{message.get('content', '')}")


# ========== 示例 2: 多工具选择 ==========
def example_multi_tools():
    """
    示例 2: 多个工具，模型自动选择
    """
    print("\n" + "=" * 60)
    print("示例 2: 多工具选择 - 计算器 + 翻译")
    print("=" * 60)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "执行数学计算",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式，例如：123 * 456"
                        }
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "translate",
                "description": "翻译文本到指定语言",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "要翻译的文本"
                        },
                        "target_language": {
                            "type": "string",
                            "description": "目标语言，例如：英语、日语、法语",
                            "enum": ["英语", "日语", "法语", "德语", "西班牙语"]
                        }
                    },
                    "required": ["text", "target_language"]
                }
            }
        }
    ]

    # 测试 1: 数学计算
    print("\n--- 测试 1: 数学计算 ---")
    messages = [
        {"role": "user", "content": "帮我算一下 123 乘以 456 等于多少"}
    ]

    response = call_glm(messages, tools)
    message = response["choices"][0]["message"]

    if "tool_calls" in message:
        for tool_call in message["tool_calls"]:
            func_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            print(f"🔧 调用: {func_name}({args})")

            if func_name == "calculate":
                # 实际执行计算
                try:
                    result = eval(args["expression"])  # 简单计算
                    print(f"📊 计算结果: {result}")
                except:
                    result = "计算错误"

    # 测试 2: 翻译
    print("\n--- 测试 2: 翻译 ---")
    messages = [
        {"role": "user", "content": "把'你好，世界'翻译成英语"}
    ]

    response = call_glm(messages, tools)
    message = response["choices"][0]["message"]

    if "tool_calls" in message:
        for tool_call in message["tool_calls"]:
            func_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            print(f"🔧 调用: {func_name}({args})")


# ========== 示例 3: 数据库查询 ==========
def example_database_query():
    """
    示例 3: 模拟数据库查询
    """
    print("\n" + "=" * 60)
    print("示例 3: 数据库查询工具")
    print("=" * 60)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "查询员工信息数据库",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "表名",
                            "enum": ["employees", "departments", "projects"]
                        },
                        "filters": {
                            "type": "object",
                            "description": "查询条件",
                            "properties": {
                                "department": {
                                    "type": "string",
                                    "description": "部门名称"
                                },
                                "name": {
                                    "type": "string",
                                    "description": "员工姓名"
                                }
                            }
                        }
                    },
                    "required": ["table"]
                }
            }
        }
    ]

    # 模拟数据库
    mock_db = {
        "employees": [
            {"name": "张三", "department": "技术部", "position": "工程师", "salary": 25000},
            {"name": "李四", "department": "销售部", "position": "经理", "salary": 30000},
            {"name": "王五", "department": "技术部", "position": "架构师", "salary": 40000},
        ]
    }

    messages = [
        {"role": "user", "content": "帮我查一下技术部有哪些员工"}
    ]

    print("\n📝 用户提问: 帮我查一下技术部有哪些员工")
    response = call_glm(messages, tools)
    message = response["choices"][0]["message"]

    if "tool_calls" in message:
        for tool_call in message["tool_calls"]:
            func_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            print(f"\n🔧 模型生成查询:")
            print(f"  表: {args.get('table')}")
            print(f"  条件: {args.get('filters', {})}")

            # 模拟查询
            if func_name == "query_database":
                table = args.get("table")
                filters = args.get("filters", {})

                results = mock_db.get(table, [])
                if "department" in filters:
                    results = [r for r in results if r["department"] == filters["department"]]

                print(f"\n📊 查询结果:")
                for r in results:
                    print(f"  - {r['name']}: {r['position']}, 薪资 {r['salary']}")


# ========== 示例 4: 复杂参数结构 ==========
def example_complex_parameters():
    """
    示例 4: 复杂参数 - 创建订单
    """
    print("\n" + "=" * 60)
    print("示例 4: 复杂参数结构 - 创建订单")
    print("=" * 60)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_order",
                "description": "创建新订单",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer": {
                            "type": "object",
                            "description": "客户信息",
                            "properties": {
                                "name": {"type": "string"},
                                "phone": {"type": "string"},
                                "address": {"type": "string"}
                            },
                            "required": ["name", "phone"]
                        },
                        "items": {
                            "type": "array",
                            "description": "订单商品列表",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "product_id": {"type": "string"},
                                    "product_name": {"type": "string"},
                                    "quantity": {"type": "integer"},
                                    "price": {"type": "number"}
                                },
                                "required": ["product_id", "quantity"]
                            }
                        },
                        "shipping_method": {
                            "type": "string",
                            "enum": ["standard", "express", "same_day"],
                            "description": "配送方式"
                        }
                    },
                    "required": ["customer", "items"]
                }
            }
        }
    ]

    messages = [
        {"role": "user", "content": "帮我创建一个订单，客户叫张三，电话 13800138000，要买 2 个 iPhone15（单价 5999）和 1 个 AirPods（单价 1999），用快递配送"}
    ]

    print("\n📝 用户创建订单请求...")
    response = call_glm(messages, tools)
    message = response["choices"][0]["message"]

    if "tool_calls" in message:
        for tool_call in message["tool_calls"]:
            args = json.loads(tool_call["function"]["arguments"])
            print(f"\n🔧 模型提取的订单信息:")
            print(f"  客户: {args['customer']['name']}, 电话: {args['customer']['phone']}")
            print(f"  商品:")
            for item in args['items']:
                print(f"    - {item.get('product_name', item['product_id'])} x {item['quantity']}")
            print(f"  配送: {args.get('shipping_method', 'standard')}")


# ========== 示例 5: 完整的对话流程 ==========
def example_full_conversation():
    """
    示例 5: 完整的对话流程，包含多次工具调用
    """
    print("\n" + "=" * 60)
    print("示例 5: 完整对话流程")
    print("=" * 60)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_products",
                "description": "搜索商品",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string"},
                        "category": {"type": "string"}
                    },
                    "required": ["keyword"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_to_cart",
                "description": "添加商品到购物车",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string"},
                        "quantity": {"type": "integer"}
                    },
                    "required": ["product_id", "quantity"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "checkout",
                "description": "结算购物车",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "payment_method": {
                            "type": "string",
                            "enum": ["alipay", "wechat", "credit_card"]
                        }
                    },
                    "required": ["payment_method"]
                }
            }
        }
    ]

    # 模拟商品数据库
    products_db = {
        "phone": [{"id": "P001", "name": "iPhone 15", "price": 5999},
                  {"id": "P002", "name": "小米 14", "price": 3999}],
        "laptop": [{"id": "L001", "name": "MacBook Pro", "price": 14999}]
    }

    # 对话历史
    messages = [
        {"role": "system", "content": "你是一个购物助手，帮助用户搜索商品、添加到购物车并结算。"}
    ]

    # 用户问题
    user_queries = [
        "我想买一部手机",
        "把 iPhone 15 加到购物车，要 1 台",
        "我要结算，用支付宝支付"
    ]

    for query in user_queries:
        print(f"\n👤 用户: {query}")
        messages.append({"role": "user", "content": query})

        # 调用 API
        response = call_glm(messages, tools)
        message = response["choices"][0]["message"]

        # 添加到历史
        msg_dict = {
            "role": message["role"],
            "content": message.get("content", "")
        }
        if "tool_calls" in message:
            msg_dict["tool_calls"] = message["tool_calls"]
        messages.append(msg_dict)

        # 处理工具调用
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])
                tool_call_id = tool_call["id"]

                print(f"  🔧 调用: {func_name}({args})")

                # 模拟执行
                if func_name == "search_products":
                    keyword = args.get("keyword", "")
                    results = products_db.get("phone", []) if "手机" in keyword else []
                    result_content = json.dumps(results, ensure_ascii=False)

                elif func_name == "add_to_cart":
                    product_id = args.get("product_id")
                    quantity = args.get("quantity")
                    result_content = json.dumps({
                        "success": True,
                        "message": f"已添加 {product_id} x {quantity} 到购物车"
                    }, ensure_ascii=False)

                elif func_name == "checkout":
                    payment = args.get("payment_method")
                    result_content = json.dumps({
                        "success": True,
                        "order_id": "ORD20240201",
                        "total": 5999,
                        "payment_method": payment
                    }, ensure_ascii=False)

                else:
                    result_content = "{}"

                # 添加工具结果到历史
                messages.append({
                    "role": "tool",
                    "content": result_content,
                    "tool_call_id": tool_call_id
                })

            # 获取最终回复
            final_response = call_glm(messages, tools)
            final_message = final_response["choices"][0]["message"]
            messages.append({
                "role": final_message["role"],
                "content": final_message.get("content", "")
            })
            print(f"  🤖 助手: {final_message.get('content', '')[:100]}...")
        else:
            print(f"  🤖 助手: {message.get('content', '')[:100]}...")


# ========== 工具定义最佳实践 ==========
def best_practices():
    """
    Function Calling 最佳实践
    """
    print("\n" + "=" * 60)
    print("Function Calling 最佳实践")
    print("=" * 60)

    tips = """
1. 函数命名
   - 使用清晰、描述性的名称
   - 例如: get_weather, search_products, create_order

2. 参数设计
   - 每个参数都要有 description
   - 使用 enum 限制可选值
   - 明确标记 required 字段

3. 描述撰写
   - 清晰说明函数用途
   - 描述参数的预期格式
   - 提供示例值

4. 安全考虑
   - 始终验证输入参数
   - 对外部 API 调用做错误处理
   - 敏感操作需要额外确认

5. 限制说明
   - tool_choice 只支持 "auto"
   - 需要处理模型不调用函数的情况
   - 复杂查询可能需要多次调用
    """
    print(tips)


# ========== 主函数 ==========
def main():
    print("🔧 智谱 GLM-4 Function Calling 示例")
    print("官方文档: https://docs.bigmodel.cn/cn/guide/capabilities/function-calling")
    print()

    try:
        example_weather()
    except Exception as e:
        print(f"示例 1 错误: {e}")

    try:
        example_multi_tools()
    except Exception as e:
        print(f"示例 2 错误: {e}")

    try:
        example_database_query()
    except Exception as e:
        print(f"示例 3 错误: {e}")

    try:
        example_complex_parameters()
    except Exception as e:
        print(f"示例 4 错误: {e}")

    try:
        example_full_conversation()
    except Exception as e:
        print(f"示例 5 错误: {e}")

    best_practices()

    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
