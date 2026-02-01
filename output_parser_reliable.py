"""
智谱 API 可靠 JSON 输出方案
解决 LLM 输出格式不稳定的问题
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field  # 使用标准 pydantic v2
from typing import List, Optional
import json
import re

# ========== 初始化 LLM ==========
llm = ChatOpenAI(
    model="glm-4.7",
    openai_api_key="9c575fbc0d714aa5a2ed2b1fec1359ec.Ve1Wz7QLlCHCsTiG",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,  # 降低温度，让输出更确定
)


# ========== 方法 1: with_structured_output (推荐) ==========
def method1_structured_output():
    """
    方法 1: 使用 with_structured_output (基于 Function Calling)
    最可靠的方式，GLM-4.7 支持
    """
    print("=" * 60)
    print("方法 1: with_structured_output (基于 Function Calling)")
    print("=" * 60)

    class Movie(BaseModel):
        name: str = Field(description="电影名称")
        year: int = Field(description="上映年份")
        director: str = Field(description="导演")
        rating: float = Field(description="评分 0-10", ge=0, le=10)
        genres: List[str] = Field(description="电影类型列表")

    # 创建结构化输出的 LLM - 显式指定使用 function_calling
    structured_llm = llm.with_structured_output(
        Movie,
        method="function_calling"  # 显式指定方法
    )

    # 直接使用，不需要额外的 parser
    prompt = """请推荐一部2023年的科幻电影，要有高评分。直接返回电影信息。"""

    result = structured_llm.invoke(prompt)

    print(f"✅ 返回类型: {type(result)}")
    print(f"电影: {result.name}")
    print(f"年份: {result.year}")
    print(f"导演: {result.director}")
    print(f"评分: {result.rating}")
    print(f"类型: {result.genres}")

    return result


# ========== 方法 2: 强化 Prompt + 约束 ==========
def method2_enhanced_prompt():
    """
    方法 2: 强化 Prompt 约束
    通过更严格的 prompt 提高输出格式正确率
    """
    print("\n" + "=" * 60)
    print("方法 2: 强化 Prompt 约束")
    print("=" * 60)

    class Book(BaseModel):
        title: str = Field(description="书名")
        author: str = Field(description="作者")
        pages: int = Field(description="页数")

    parser = PydanticOutputParser(pydantic_object=Book)

    # 强化版 prompt，添加更多约束
    prompt = PromptTemplate(
        template="""你是一个专业的图书信息提取助手。

{format_instructions}

⚠️ 重要规则：
1. 必须严格按照上面的 JSON Schema 格式输出
2. 只输出 JSON 数据，不要添加任何其他文字、注释或 markdown 格式
3. 不要包含 ```json 或 ``` 代码块标记
4. 确保所有必填字段都有值
5. 字符串必须用双引号包裹

用户要求: {query}

请直接输出 JSON:""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke({"query": "推荐一本经典科幻小说"})
        print(f"✅ 解析成功:")
        print(f"  书名: {result.title}")
        print(f"  作者: {result.author}")
        print(f"  页数: {result.pages}")
        return result
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        return None


# ========== 方法 3: 手动重试机制 ==========
def method3_retry_parser():
    """
    方法 3: 手动重试机制
    当解析失败时，将错误信息反馈给 LLM 要求重新生成
    """
    print("\n" + "=" * 60)
    print("方法 3: 手动重试机制")
    print("=" * 60)

    class Product(BaseModel):
        name: str = Field(description="产品名称")
        price: float = Field(description="价格")
        features: List[str] = Field(description="产品特性")

    parser = PydanticOutputParser(pydantic_object=Product)
    max_retries = 2

    prompt_text = f"""请描述一个电子产品。
{parser.get_format_instructions()}

产品类型: 智能手机
"""

    for attempt in range(max_retries + 1):
        try:
            output = llm.invoke(prompt_text)
            result = parser.parse(output.content)
            print(f"✅ 解析成功 (尝试 {attempt + 1} 次):")
            print(f"  产品: {result.name}")
            print(f"  价格: {result.price}")
            print(f"  特性: {result.features}")
            return result
        except Exception as e:
            print(f"  ⚠️ 第 {attempt + 1} 次尝试失败: {e}")
            if attempt < max_retries:
                # 让 LLM 修复错误
                fix_prompt = f"""之前的输出格式有误。
错误信息: {e}

请修正以下输出，使其符合 JSON 格式要求:
{output.content}

只输出修正后的 JSON，不要其他文字。"""
                prompt_text = fix_prompt
            else:
                print(f"❌ 最终失败")
                return None


# ========== 方法 4: 后处理修复 ==========
def method4_post_processing():
    """
    方法 4: 输出后处理修复
    尝试清理和修复常见的格式错误
    """
    print("\n" + "=" * 60)
    print("方法 4: 输出后处理修复")
    print("=" * 60)

    class Person(BaseModel):
        name: str = Field(description="姓名")
        age: int = Field(description="年龄")
        city: str = Field(description="城市")

    parser = PydanticOutputParser(pydantic_object=Person)

    def clean_json_output(raw_output: str) -> str:
        """清理 LLM 输出中的常见噪音"""
        # 去掉 markdown 代码块标记
        cleaned = re.sub(r'^```json\s*', '', raw_output.strip())
        cleaned = re.sub(r'\s*```$', '', cleaned)
        # 去掉开头的 "json" 标识
        cleaned = re.sub(r'^json\s*', '', cleaned)
        # 找到第一个 { 和最后一个 }
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1:
            cleaned = cleaned[start:end+1]
        return cleaned

    prompt = PromptTemplate(
        template="""请生成一个人物信息。
{format_instructions}

要求: 一个{description}
""",
        input_variables=["description"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # 执行
    _input = prompt.format(description="30岁的北京程序员")
    raw_output = llm.invoke(_input).content

    print(f"📝 原始输出:\n{raw_output}\n")

    # 后处理
    cleaned = clean_json_output(raw_output)
    print(f"🧹 清理后:\n{cleaned}\n")

    try:
        result = parser.parse(cleaned)
        print(f"✅ 解析成功:")
        print(f"  姓名: {result.name}")
        print(f"  年龄: {result.age}")
        print(f"  城市: {result.city}")
        return result
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        return None


# ========== 方法 5: 自定义健壮解析器 ==========
class RobustJsonParser:
    """
    自定义健壮解析器
    结合多种策略提高成功率
    """

    def __init__(self, pydantic_model, llm, max_retries=2):
        self.pydantic_model = pydantic_model
        self.llm = llm
        self.max_retries = max_retries
        self.parser = PydanticOutputParser(pydantic_object=pydantic_model)

    def clean_output(self, text: str) -> str:
        """清理输出"""
        # 去掉 markdown 代码块
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        # 提取 JSON 对象
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return text.strip()

    def parse(self, text: str) -> Optional[BaseModel]:
        """尝试解析，失败则重试"""
        cleaned = self.clean_output(text)

        for attempt in range(self.max_retries + 1):
            try:
                return self.parser.parse(cleaned)
            except Exception as e:
                if attempt < self.max_retries:
                    print(f"  ⚠️ 第 {attempt + 1} 次尝试失败，请求 LLM 修复...")
                    # 让 LLM 修复错误
                    fix_prompt = f"""之前的输出格式有误，错误信息: {e}

请修正以下 JSON，使其符合要求:
{cleaned}

只输出修正后的 JSON，不要其他文字。"""
                    text = self.llm.invoke(fix_prompt).content
                    cleaned = self.clean_output(text)
                else:
                    raise e
        return None


def method5_custom_robust_parser():
    """
    方法 5: 使用自定义健壮解析器
    """
    print("\n" + "=" * 60)
    print("方法 5: 自定义健壮解析器")
    print("=" * 60)

    class Company(BaseModel):
        name: str = Field(description="公司名称")
        founded: int = Field(description="成立年份")
        employees: int = Field(description="员工数量")
        industry: str = Field(description="所属行业")

    parser = RobustJsonParser(Company, llm, max_retries=2)

    prompt = f"""描述一家知名的科技公司。
{parser.parser.get_format_instructions()}

要求: 成立超过20年，员工超过10000人
"""

    raw_output = llm.invoke(prompt).content
    print(f"📝 原始输出:\n{raw_output}\n")

    try:
        result = parser.parse(raw_output)
        print(f"✅ 解析成功:")
        print(f"  公司: {result.name}")
        print(f"  成立: {result.founded}")
        print(f"  员工: {result.employees}")
        print(f"  行业: {result.industry}")
        return result
    except Exception as e:
        print(f"❌ 最终失败: {e}")
        return None


# ========== 对比测试 ==========
def compare_methods():
    """对比不同方法的可靠性"""
    print("\n" + "=" * 60)
    print("对比测试: 执行 5 次，看成功率")
    print("=" * 60)

    class SimpleData(BaseModel):
        value: str = Field(description="一个值")
        number: int = Field(description="一个数字")

    parser = PydanticOutputParser(pydantic_object=SimpleData)

    # 方法 A: 普通方式
    print("\n方法 A: 普通 Prompt + Parser")
    success_a = 0
    for i in range(5):
        try:
            prompt = PromptTemplate(
                template="生成数据。{format_instructions}",
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            chain = prompt | llm | parser
            chain.invoke({})
            success_a += 1
            print(f"  第 {i+1} 次: ✅")
        except Exception as e:
            print(f"  第 {i+1} 次: ❌")

    # 方法 B: with_structured_output
    print("\n方法 B: with_structured_output")
    success_b = 0
    structured_llm = llm.with_structured_output(SimpleData, method="function_calling")
    for i in range(5):
        try:
            structured_llm.invoke("生成数据")
            success_b += 1
            print(f"  第 {i+1} 次: ✅")
        except Exception as e:
            print(f"  第 {i+1} 次: ❌")

    print(f"\n📊 结果:")
    print(f"  方法 A (普通): {success_a}/5 成功")
    print(f"  方法 B (结构化输出): {success_b}/5 成功")


# ========== 主函数 ==========
def main():
    print("🔧 智谱 API 可靠 JSON 输出方案")
    print()

    try:
        method1_structured_output()
    except Exception as e:
        print(f"方法 1 错误: {e}")
        import traceback
        traceback.print_exc()

    try:
        method2_enhanced_prompt()
    except Exception as e:
        print(f"方法 2 错误: {e}")
        import traceback
        traceback.print_exc()

    try:
        method3_retry_parser()
    except Exception as e:
        print(f"方法 3 错误: {e}")
        import traceback
        traceback.print_exc()

    try:
        method4_post_processing()
    except Exception as e:
        print(f"方法 4 错误: {e}")
        import traceback
        traceback.print_exc()

    try:
        method5_custom_robust_parser()
    except Exception as e:
        print(f"方法 5 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
