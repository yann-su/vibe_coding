"""
LangChain OutputParser 练习
包含多种 OutputParser 的使用示例
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import (
    PydanticOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import json

# ========== 初始化 LLM ==========
llm = ChatOpenAI(
    model="glm-4.7",
    openai_api_key="9c575fbc0d714aa5a2ed2b1fec1359ec.Ve1Wz7QLlCHCsTiG",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.7,
)


def demo_separator(title):
    """打印分隔线"""
    print(f"\n{'=' * 60}")
    print(f"🎯 {title}")
    print(f"{'=' * 60}\n")


# ========== 1. StrOutputParser - 字符串输出解析器 ==========
def demo_str_output_parser():
    """基础字符串输出解析器 - 默认使用，直接返回字符串"""
    demo_separator("1. StrOutputParser - 字符串输出解析器")

    parser = StrOutputParser()

    # 创建简单的链
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的翻译助手"),
        ("human", "请将以下中文翻译成英文: {text}")
    ])

    # 构建链: prompt -> llm -> parser
    chain = prompt | llm | parser

    result = chain.invoke({"text": "你好，世界！"})
    print(f"原文: 你好，世界！")
    print(f"翻译结果: {result}")


# ========== 2. PydanticOutputParser - 结构化数据解析 ==========
def demo_pydantic_output_parser():
    """使用 Pydantic 模型解析结构化输出 - 最常用、最强大"""
    demo_separator("2. PydanticOutputParser - 结构化数据解析")

    # 定义 Pydantic 模型
    class Movie(BaseModel):
        name: str = Field(description="电影名称")
        year: int = Field(description="上映年份")
        director: str = Field(description="导演")
        rating: float = Field(description="评分(0-10)")
        genres: List[str] = Field(description="电影类型列表")
        summary: str = Field(description="剧情简介")

        @validator('rating')
        def rating_must_be_valid(cls, v):
            if not 0 <= v <= 10:
                raise ValueError('评分必须在 0-10 之间')
            return v

    # 创建解析器
    parser = PydanticOutputParser(pydantic_object=Movie)

    # 查看格式说明
    format_instructions = parser.get_format_instructions()
    print("📋 格式说明:")
    print(format_instructions)
    print()

    # 创建提示词模板
    prompt = PromptTemplate(
        template="""请根据用户要求生成电影信息。
{format_instructions}

用户要求: {query}
""",
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions}
    )

    # 构建链
    chain = prompt | llm | parser

    # 执行
    result = chain.invoke({"query": "推荐一部2023年的科幻电影，要有高评分"})

    print(f"✅ 解析结果:")
    print(f"  电影名称: {result.name}")
    print(f"  上映年份: {result.year}")
    print(f"  导演: {result.director}")
    print(f"  评分: {result.rating}")
    print(f"  类型: {', '.join(result.genres)}")
    print(f"  简介: {result.summary}")


# ========== 3. JsonOutputParser - JSON 解析器 ==========
def demo_json_output_parser():
    """JSON 输出解析器 - 灵活解析任何 JSON 格式"""
    demo_separator("3. JsonOutputParser - JSON 解析器")

    # 定义期望的结构（可选）
    class BookInfo(BaseModel):
        title: str = Field(description="书名")
        author: str = Field(description="作者")
        pages: int = Field(description="页数")
        isbn: Optional[str] = Field(description="ISBN号", default=None)
        tags: List[str] = Field(description="标签")

    # 创建解析器，可以传入 Pydantic 模型来指定结构
    parser = JsonOutputParser(pydantic_object=BookInfo)

    format_instructions = parser.get_format_instructions()
    print("📋 格式说明:")
    print(format_instructions)
    print()

    prompt = PromptTemplate(
        template="""请推荐一本书，并以 JSON 格式返回书籍信息。
{format_instructions}

用户偏好: {preference}
""",
        input_variables=["preference"],
        partial_variables={"format_instructions": format_instructions}
    )

    chain = prompt | llm | parser

    result = chain.invoke({"preference": "喜欢科幻和哲学类书籍"})

    print(f"✅ 解析结果 (JSON):")
    print(json.dumps(result, indent=2, ensure_ascii=False))


# ========== 4. CommaSeparatedListOutputParser - 逗号分隔列表解析 ==========
def demo_comma_separated_parser():
    """逗号分隔列表解析器 - 简单实用"""
    demo_separator("4. CommaSeparatedListOutputParser - 逗号分隔列表解析")

    parser = CommaSeparatedListOutputParser()

    format_instructions = parser.get_format_instructions()
    print("📋 格式说明:")
    print(format_instructions)
    print()

    prompt = PromptTemplate(
        template="""请列出{count}个{topic}。
{format_instructions}
""",
        input_variables=["count", "topic"],
        partial_variables={"format_instructions": format_instructions}
    )

    chain = prompt | llm | parser

    result = chain.invoke({"count": "5", "topic": "中国著名的旅游景点"})

    print(f"✅ 解析结果 (列表):")
    for i, item in enumerate(result, 1):
        print(f"  {i}. {item.strip()}")


# ========== 5. 复杂示例：多模型嵌套 ==========
def demo_complex_nested():
    """复杂示例：嵌套的 Pydantic 模型"""
    demo_separator("5. 复杂示例 - 嵌套 Pydantic 模型")

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

    parser = PydanticOutputParser(pydantic_object=Person)

    prompt = PromptTemplate(
        template="""请生成一个虚构人物的完整信息。
{format_instructions}

要求: 一个生活在{location}的{occupation}
""",
        input_variables=["location", "occupation"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser

    result = chain.invoke({"location": "北京", "occupation": "软件工程师"})

    print(f"✅ 解析结果:")
    print(f"  姓名: {result.name}")
    print(f"  年龄: {result.age}")
    print(f"  邮箱: {result.email}")
    print(f"  地址: {result.address.city}, {result.address.street}, {result.address.zipcode}")
    print(f"  爱好: {', '.join(result.hobbies)}")


# ========== 6. 错误处理示例 ==========
def demo_error_handling():
    """错误处理：当输出格式不正确时"""
    demo_separator("6. 错误处理示例")

    from langchain_core.exceptions import OutputParserException

    class Product(BaseModel):
        name: str = Field(description="产品名称")
        price: float = Field(description="价格")
        in_stock: bool = Field(description="是否有库存")

    parser = PydanticOutputParser(pydantic_object=Product)

    # 创建一个可能产生错误格式的提示
    prompt = PromptTemplate(
        template="""请用指定的 JSON 格式描述一个产品。
{format_instructions}

产品描述: {description}

注意：请严格按照 JSON 格式输出，不要添加其他文字。
""",
        input_variables=["description"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    try:
        chain = prompt | llm | parser
        result = chain.invoke({"description": "苹果 iPhone 15 Pro，价格 999 美元"})
        print(f"✅ 解析成功:")
        print(f"  产品: {result.name}")
        print(f"  价格: ${result.price}")
        print(f"  库存: {'有' if result.in_stock else '无'}")
    except OutputParserException as e:
        print(f"❌ 解析错误: {e}")
        print("💡 提示: LLM 没有按照要求的格式输出")


# ========== 7. LCEL 链式调用示例 ==========
def demo_lcel_chain():
    """使用 LCEL (LangChain Expression Language) 构建复杂链"""
    demo_separator("7. LCEL 链式调用示例")

    # 解析器 1: 提取关键词
    keyword_parser = CommaSeparatedListOutputParser()

    keyword_prompt = PromptTemplate(
        template="""请从以下文本中提取 3-5 个关键词。
{format_instructions}

文本: {text}
""",
        input_variables=["text"],
        partial_variables={"format_instructions": keyword_parser.get_format_instructions()}
    )

    keyword_chain = keyword_prompt | llm | keyword_parser

    # 解析器 2: 生成摘要
    class Summary(BaseModel):
        brief: str = Field(description="一句话摘要")
        detailed: str = Field(description="详细摘要")
        sentiment: str = Field(description="情感倾向(positive/negative/neutral)")

    summary_parser = PydanticOutputParser(pydantic_object=Summary)

    summary_prompt = PromptTemplate(
        template="""请对以下文本生成摘要。
{format_instructions}

文本: {text}
""",
        input_variables=["text"],
        partial_variables={"format_instructions": summary_parser.get_format_instructions()}
    )

    summary_chain = summary_prompt | llm | summary_parser

    # 测试文本
    text = """
    人工智能（AI）正在深刻改变我们的生活和工作方式。
    从自动驾驶汽车到智能助手，AI 技术已经在多个领域展现出巨大潜力。
    然而，AI 的发展也带来了一些挑战，如隐私保护和就业问题。
    未来，我们需要在技术进步和社会责任之间找到平衡。
    """

    print("📄 原文本:")
    print(text.strip())
    print()

    # 执行关键词提取
    keywords = keyword_chain.invoke({"text": text})
    print(f"🔑 关键词: {', '.join(keywords)}")
    print()

    # 执行摘要生成
    summary = summary_chain.invoke({"text": text})
    print(f"📝 摘要结果:")
    print(f"  一句话: {summary.brief}")
    print(f"  详细版: {summary.detailed}")
    print(f"  情感倾向: {summary.sentiment}")


# ========== 主函数 ==========
def main():
    """运行所有示例"""
    print("🚀 LangChain OutputParser 练习开始！")
    print("本练习包含 7 个不同的 OutputParser 示例")

    try:
        demo_str_output_parser()
        demo_pydantic_output_parser()
        demo_json_output_parser()
        demo_comma_separated_parser()
        demo_complex_nested()
        demo_error_handling()
        demo_lcel_chain()

        print(f"\n{'=' * 60}")
        print("🎉 所有练习完成！")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
