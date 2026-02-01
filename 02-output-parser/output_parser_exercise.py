"""
OutputParser 练习挑战
请根据注释提示完成代码
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser, CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
import json

# ========== 初始化 LLM ==========
llm = ChatOpenAI(
    model="glm-4.7",
    openai_api_key="9c575fbc0d714aa5a2ed2b1fec1359ec.Ve1Wz7QLlCHCsTiG",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.7,
)


# ========== 练习 1: 基础 PydanticOutputParser ==========
def exercise_1():
    """
    练习 1: 创建一个简单的电影信息解析器
    目标：定义 Movie 模型，包含 title, year, director, rating 字段
    """
    print("=" * 50)
    print("练习 1: 基础 PydanticOutputParser")
    print("=" * 50)

    # TODO: 定义 Movie 模型
    # class Movie(BaseModel):
    #     ... 请补充字段定义

    # TODO: 创建 PydanticOutputParser
    # parser = ...

    # TODO: 创建 PromptTemplate，包含 format_instructions
    # prompt = PromptTemplate(...)

    # TODO: 构建链并执行
    # chain = ...
    # result = chain.invoke(...)

    # TODO: 打印结果
    pass  # 删除这行，开始编写代码


# ========== 练习 2: 嵌套模型 ==========
def exercise_2():
    """
    练习 2: 创建一个嵌套模型
    目标：定义包含 Address 和 Person 的嵌套结构
    """
    print("=" * 50)
    print("练习 2: 嵌套 Pydantic 模型")
    print("=" * 50)

    # TODO: 定义 Address 模型（city, street）

    # TODO: 定义 Person 模型，包含 Address 字段

    # TODO: 创建解析器和链

    # TODO: 执行并打印结果
    pass


# ========== 练习 3: 列表解析 ==========
def exercise_3():
    """
    练习 3: 使用 CommaSeparatedListOutputParser
    目标：解析 LLM 输出的逗号分隔列表
    """
    print("=" * 50)
    print("练习 3: CommaSeparatedListOutputParser")
    print("=" * 50)

    # TODO: 创建 CommaSeparatedListOutputParser

    # TODO: 创建提示词，要求 LLM 返回逗号分隔的列表

    # TODO: 执行并遍历打印列表项
    pass


# ========== 练习 4: JSON 解析 ==========
def exercise_4():
    """
    练习 4: 使用 JsonOutputParser
    目标：灵活解析 JSON 输出
    """
    print("=" * 50)
    print("练习 4: JsonOutputParser")
    print("=" * 50)

    # TODO: 定义 Book 模型

    # TODO: 创建 JsonOutputParser

    # TODO: 创建链并执行

    # TODO: 使用 json.dumps 美化输出结果
    pass


# ========== 练习 5: 实际应用场景 ==========
def exercise_5():
    """
    练习 5: 简历信息提取器
    目标：从一段简历文本中提取结构化信息
    """
    print("=" * 50)
    print("练习 5: 简历信息提取器 (综合练习)")
    print("=" * 50)

    resume_text = """
    张三，男，28岁，软件工程师，5年工作经验。
    技能：Python, Java, React, Docker, Kubernetes
    邮箱：zhangsan@example.com
    电话：138-1234-5678
    期望薪资：25k-35k
    """

    # TODO: 定义 Resume 模型，包含：
    # - name (姓名)
    # - age (年龄)
    # - skills (技能列表)
    # - email (邮箱)
    # - expected_salary (期望薪资)

    # TODO: 创建解析器

    # TODO: 创建提示词，从简历文本提取信息

    # TODO: 执行并打印结构化结果
    pass


# ========== 参考答案（在需要时参考） ==========
"""
参考答案 - exercise_1:

class Movie(BaseModel):
    title: str = Field(description="电影标题")
    year: int = Field(description="上映年份")
    director: str = Field(description="导演")
    rating: float = Field(description="评分")

parser = PydanticOutputParser(pydantic_object=Movie)

prompt = PromptTemplate(
    template="推荐一部电影。\\n{format_instructions}\\n",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
result = chain.invoke({})
print(f"电影: {result.title}, 年份: {result.year}")
"""


if __name__ == "__main__":
    print("🎯 OutputParser 练习挑战")
    print("请完成 exercise_1 到 exercise_5 的代码编写")
    print("参考 output_parser_demo.py 中的示例代码")
    print()

    # 取消注释以下行来运行对应练习
    # exercise_1()
    # exercise_2()
    # exercise_3()
    # exercise_4()
    # exercise_5()
