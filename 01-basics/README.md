# Vibe Coding - LangChain 学习项目

> 🤖 智谱 GLM-4 + LangChain 学习代码库

---

## 📁 文件索引

### 🎯 基础入门

| 文件 | 说明 | 学习重点 |
|------|------|----------|
| `langchain_hello.py` | LangChain 基础交互示例 | 流式输出、对话历史 |

### 📚 OutputParser 学习

#### 1. 基础示例
| 文件 | 说明 | 状态 |
|------|------|------|
| `output_parser_demo.py` | **7种 OutputParser 完整示例** | ✅ 推荐入门 |
| `output_parser_exercise.py` | **5个动手练习**（含参考答案） | ✅ 巩固练习 |

**内容覆盖**：
- `StrOutputParser` - 字符串解析
- `PydanticOutputParser` - 结构化数据解析 ⭐
- `JsonOutputParser` - JSON 解析
- `CommaSeparatedListOutputParser` - 列表解析
- 嵌套模型示例
- 错误处理
- LCEL 链式调用

#### 2. 智谱 API 适配方案 ⭐重要

| 文件 | 说明 | 推荐度 |
|------|------|--------|
| `output_parser_fixed.py` | **强化 Prompt + 后处理方案** | ⭐⭐⭐ 最稳定 |
| `output_parser_reliable.py` | 多种可靠方案对比 | ⭐⭐ 参考 |

**核心函数**：`reliable_json_parser()`
```python
# 跨平台最可靠的方案
# - 强化 Prompt 约束
# - 自动后处理清理
# - 失败自动重试修复
result = reliable_json_parser(Movie, "推荐一部电影", llm)
```

#### 3. Function Calling 完整示例 ⭐推荐

**重要发现**：LangChain 的 `with_structured_output()` 对智谱 API 有兼容性问题，但 `bind_tools()` 正常工作！

| 文件 | 说明 | 推荐度 |
|------|------|--------|
| `langchain_function_calling_guide.py` | **推荐方案**：`bind_tools()` + `@tool` | ⭐⭐⭐ 最稳定 |
| `FUNCTION_CALLING_GUIDE.md` | 使用指南和对比 | ⭐⭐⭐ 必读 |
| `langchain_function_calling_demo.py` | 完整演示（含对比测试） | ⭐⭐ 参考 |
| `zhipu_function_calling_examples.py` | 原生 API 示例（5个场景） | ⭐⭐ 底层实现 |

**快速开始**：
```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="glm-4.7", ...)

@tool
def get_weather(city: str) -> str:
    """获取天气"""
    return f"{city}晴天"

# ✅ 推荐：使用 bind_tools
llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.invoke("北京天气")

# ❌ 避免：with_structured_output 有兼容性问题
# structured_llm = llm.with_structured_output(Model)  # 不推荐
```

**详细对比**：

| 方式 | 智谱 GLM-4.7 | 说明 |
|------|--------------|------|
| `bind_tools() + @tool` | ✅ **正常** | **推荐** |
| `with_structured_output()` | ❌ **失败** | 避免使用 |
| 原生 API | ✅ **正常** | 最底层 |

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 进入项目目录
cd /Users/yannsu/PycharmProjects/vibe_coding

# 确保已安装依赖
pip install langchain langchain-openai pydantic
```

### 2. 运行示例

```bash
# 基础示例（推荐入门）
python output_parser_demo.py

# 动手练习
python output_parser_exercise.py

# 智谱 API 可靠方案
python output_parser_fixed.py
```

### 3. 学习顺序

```
1. output_parser_demo.py      → 了解各种 Parser
2. output_parser_exercise.py  → 动手练习
3. output_parser_fixed.py     → 掌握智谱 API 方案
```

---

## 📖 核心知识点

### OutputParser 使用步骤

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# 1. 定义 Pydantic 模型
class Movie(BaseModel):
    name: str = Field(description="电影名称")
    year: int = Field(description="上映年份")
    rating: float = Field(description="评分")

# 2. 创建 Parser
parser = PydanticOutputParser(pydantic_object=Movie)

# 3. 获取格式说明
format_instructions = parser.get_format_instructions()

# 4. 构建 Prompt（包含格式说明）
prompt = PromptTemplate(
    template="推荐一部电影\n{format_instructions}",
    partial_variables={"format_instructions": format_instructions}
)

# 5. 构建链并执行
chain = prompt | llm | parser
result = chain.invoke({})  # result 是 Movie 对象
print(result.name)
```

### 智谱 API 特殊处理

```python
# ❌ 不推荐（有兼容性问题）
structured_llm = llm.with_structured_output(Movie)

# ✅ 推荐：强化 Prompt + 后处理
def reliable_json_parser(model, prompt, llm):
    # 1. 强化 Prompt
    # 2. 后处理清理 markdown、提取 JSON
    # 3. 失败时自动重试
    pass
```

---

## 🔧 项目配置

### API 配置

代码中已配置的 API：
```python
llm = ChatOpenAI(
    model="glm-4.7",
    openai_api_key="9c575fbc0d714aa5a2ed2b1fec1359ec.Ve1Wz7QLlCHCsTiG",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
)
```

### 模型说明

- **模型**：智谱 GLM-4.7
- **特点**：支持 OpenAI 兼容接口
- **注意**：Function Calling 有 bug，建议使用 Prompt + 后处理方案

---

## 📚 Obsidian 学习笔记

本项目对应 Obsidian 笔记：

- **学习总结**：`Ai学习/04-学习笔记/OutputParser学习总结.md`
- **MOC索引**：`Ai学习/04-学习笔记/LangChain-OutputParser-MOC.md`
- **技术栈**：`Ai学习/02-技术栈/LangChain框架.md`

---

## ⚠️ 已知问题

### GLM-4.7 Function Calling Bug

**现象**：
```
模型思考：需要填写 name="流浪地球2", year=2023...
实际调用：{"name": ""}  ← 空值！
```

**原因**：模型训练/实现层面的问题

**解决方案**：
- 使用 `output_parser_fixed.py` 的强化 Prompt + 后处理方案
- 不使用 `with_structured_output()`

---

## 🎯 下一步

- [ ] 将 OutputParser 应用到 Text-to-SQL 项目
- [ ] 集成到 RAG 系统
- [ ] 尝试其他模型（OpenAI GPT-4、Claude）

---

## 📝 更新日志

| 日期 | 内容 |
|------|------|
| 2026-02-01 | 创建 OutputParser 学习代码 |
| 2026-02-01 | 添加智谱 API 适配方案 |
| 2026-02-01 | 探索 Function Calling（发现 GLM-4.7 bug） |

---

**项目路径**：`/Users/yannsu/PycharmProjects/vibe_coding/`

**Obsidian Vault**：`/Users/yannsu/Documents/ObsidianRemote/`
