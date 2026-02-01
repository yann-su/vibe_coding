"""
智谱 API 可靠 JSON 输出 - 修复版
由于智谱 API 的 function calling 实现可能有差异，这里提供最可靠的方案
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import re

# ========== 初始化 LLM ==========
llm = ChatOpenAI(
    model="glm-4.7",
    openai_api_key="9c575fbc0d714aa5a2ed2b1fec1359ec.Ve1Wz7QLlCHCsTiG",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
)


# ========== 核心问题分析 ==========
"""
为什么 function calling 可能失败？

1. 智谱 API 虽然兼容 OpenAI 接口，但 function calling 实现可能有差异
2. LangChain 的 with_structured_output 对智谱 API 的适配可能不完善
3. 错误 "field required" 说明返回的 JSON 缺少字段

解决方案：使用 "强化 Prompt + 后处理" 的方式，这是跨平台最稳定的方案
"""


# ========== 最可靠的方案：强化 Prompt + 后处理修复 ==========
def reliable_json_parser(pydantic_model, prompt_text: str, llm_instance) -> Optional[BaseModel]:
    """
    可靠的 JSON 解析器
    结合强化 prompt 和自动修复
    """
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    format_instructions = parser.get_format_instructions()

    # 强化版 prompt
    enhanced_prompt = f"""你必须严格按照以下 JSON Schema 格式输出，不要添加任何其他内容：

{format_instructions}

⚠️ 严格规则：
1. 只输出纯 JSON，不要 markdown 代码块(```json)
2. 确保所有必填字段都有值
3. 字符串用双引号
4. 不要有任何解释性文字

任务: {prompt_text}

你的输出(必须是合法JSON):"""

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            # 调用 LLM
            response = llm_instance.invoke(enhanced_prompt)
            raw_output = response.content

            # 后处理：清理常见的格式问题
            cleaned = clean_json_output(raw_output)

            print(f"  尝试 {attempt + 1}:")
            print(f"    原始输出: {raw_output[:100]}...")
            print(f"    清理后: {cleaned[:100]}...")

            # 解析
            result = parser.parse(cleaned)
            return result

        except Exception as e:
            last_error = str(e)
            print(f"    ❌ 失败: {e}")

            # 让 LLM 修复错误
            if attempt < max_retries - 1:
                fix_prompt = f"""之前的 JSON 格式有误: {e}

请修正以下输出，只返回正确的 JSON：
{cleaned if 'cleaned' in locals() else raw_output}

修正后的 JSON:"""
                enhanced_prompt = fix_prompt

    raise Exception(f"解析失败，已重试 {max_retries} 次。最后错误: {last_error}")


def clean_json_output(raw_output: str) -> str:
    """清理 LLM 输出中的常见噪音"""
    text = raw_output.strip()

    # 1. 去掉 markdown 代码块标记
    text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'```\s*$', '', text)

    # 2. 去掉 "json" 前缀
    text = re.sub(r'^json\s*', '', text, flags=re.IGNORECASE)

    # 3. 提取第一个 { 到最后一个 } 之间的内容
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]

    # 4. 处理可能的多行字符串问题
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ========== 示例 1: 电影信息提取 ==========
def demo_movie():
    """电影信息提取示例"""
    print("=" * 60)
    print("示例 1: 电影信息提取")
    print("=" * 60)

    class Movie(BaseModel):
        name: str = Field(description="电影名称")
        year: int = Field(description="上映年份")
        director: str = Field(description="导演")
        rating: float = Field(description="评分 0-10")
        genres: List[str] = Field(description="电影类型列表")

    try:
        result = reliable_json_parser(
            Movie,
            "推荐一部2023年的高分科幻电影",
            llm
        )

        print(f"\n✅ 成功!")
        print(f"  电影: {result.name}")
        print(f"  年份: {result.year}")
        print(f"  导演: {result.director}")
        print(f"  评分: {result.rating}")
        print(f"  类型: {result.genres}")
        return result

    except Exception as e:
        print(f"\n❌ 最终失败: {e}")
        return None


# ========== 示例 2: 人物信息提取 ==========
def demo_person():
    """人物信息提取示例"""
    print("\n" + "=" * 60)
    print("示例 2: 人物信息提取")
    print("=" * 60)

    class Address(BaseModel):
        city: str = Field(description="城市")
        street: str = Field(description="街道")

    class Person(BaseModel):
        name: str = Field(description="姓名")
        age: int = Field(description="年龄")
        email: str = Field(description="邮箱")
        address: Address = Field(description="地址")
        hobbies: List[str] = Field(description="爱好")

    try:
        result = reliable_json_parser(
            Person,
            "生成一个住在上海的软件工程师的信息，30岁左右",
            llm
        )

        print(f"\n✅ 成功!")
        print(f"  姓名: {result.name}")
        print(f"  年龄: {result.age}")
        print(f"  邮箱: {result.email}")
        print(f"  地址: {result.address.city}, {result.address.street}")
        print(f"  爱好: {result.hobbies}")
        return result

    except Exception as e:
        print(f"\n❌ 最终失败: {e}")
        return None


# ========== 示例 3: 列表解析 ==========
def demo_list():
    """逗号分隔列表解析"""
    print("\n" + "=" * 60)
    print("示例 3: 列表解析")
    print("=" * 60)

    from langchain_core.output_parsers import CommaSeparatedListOutputParser

    parser = CommaSeparatedListOutputParser()

    prompt = f"""列出5个中国著名的旅游景点。
{parser.get_format_instructions()}
"""

    try:
        response = llm.invoke(prompt)
        result = parser.parse(response.content)

        print(f"\n✅ 成功!")
        for i, item in enumerate(result, 1):
            print(f"  {i}. {item.strip()}")
        return result

    except Exception as e:
        print(f"\n❌ 失败: {e}")
        return None


# ========== 对比：为什么不用 with_structured_output ==========
def explain_issue():
    """解释为什么 function calling 可能失败"""
    print("\n" + "=" * 60)
    print("问题分析：为什么 with_structured_output 可能失败")
    print("=" * 60)

    explanation = """
1. 智谱 API 兼容 OpenAI 接口，但 function calling 实现有差异
   - OpenAI: 返回 tool_calls 字段
   - 智谱: 可能返回格式不同，或需要特殊处理

2. LangChain 的适配问题
   - LangChain 的 with_structured_output 主要针对 OpenAI 优化
   - 对智谱 API 的适配可能不完善

3. 错误 "field required" 的含义
   - 模型返回了 JSON，但缺少某些字段
   - 说明 function calling 没有正确约束输出

4. 解决方案
   - 使用强化 Prompt + 后处理修复（本代码采用的方式）
   - 这种方式不依赖特定 API 的 function calling 实现
   - 跨平台最稳定，适用于 OpenAI、智谱、文心等
"""
    print(explanation)


# ========== 主函数 ==========
def main():
    print("🔧 智谱 API 可靠 JSON 输出 - 修复版")
    print("采用强化 Prompt + 自动修复的方案")
    print()

    demo_movie()
    demo_person()
    demo_list()
    explain_issue()

    print("\n" + "=" * 60)
    print("总结：")
    print("  对于智谱 API，推荐使用 reliable_json_parser() 函数")
    print("  它不依赖 function calling，而是通过后处理确保格式正确")
    print("=" * 60)


if __name__ == "__main__":
    main()
