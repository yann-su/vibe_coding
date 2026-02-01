"""
LangChain + 智谱 GLM-4 交互式对话示例
使用官方推荐方式（langchain-openai + OpenAI 兼容接口）
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
import sys

# 自定义流式回调处理器
class ThinkingHandler(BaseCallbackHandler):
    def __init__(self):
        self.content_started = False

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.content_started = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            if not self.content_started:
                print("\nAI: ", end="", flush=True)
                self.content_started = True
            print(token, end="", flush=True)

    def on_llm_end(self, response, **kwargs):
        if self.content_started:
            print()  # 换行

# 初始化智谱 LLM（使用 OpenAI 兼容接口）
# 注意：callbacks 需要在创建 LLM 时传入，而不是调用时传入
llm = ChatOpenAI(
    model="glm-4.7",
    openai_api_key="9c575fbc0d714aa5a2ed2b1fec1359ec.Ve1Wz7QLlCHCsTiG",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.7,
    streaming=True,
    callbacks=[ThinkingHandler()],
)

# 对话历史
messages = [
    SystemMessage(content="你是一个友好的AI助手"),
]

print("欢迎使用 GLM-4.7 对话助手！输入 'quit' 或 'exit' 退出。")
print("-" * 50)

while True:
    user_input = input("\n你: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ['quit', 'exit', 'q']:
        print("再见！")
        break

    # 添加用户消息到历史
    messages.append(HumanMessage(content=user_input))

    # 调用模型（流式输出会通过回调自动显示）
    response = llm.invoke(messages)

    # 添加AI回复到历史
    messages.append(AIMessage(content=response.content))
