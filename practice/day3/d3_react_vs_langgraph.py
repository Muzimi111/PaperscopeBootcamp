import json
import operator
import os
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()  # 从 .env 文件加载环境变量

# =====================================================================
# 准备工作：初始化统一的大模型与测试工具
# =====================================================================
def get_chat_model():
    '''获取兼容OpenAI接口的模型'''
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv('OPEN_BASE_URL')
    model_name = os.getenv("LLM_MODEL")

    if not api_key or not base_url or not model_name:
        print("请在 .env 文件中设置 OPENAI_API_KEY、OPEN_BASE_URL 和 LLM_MODEL 环境变量")
        return None
    return init_chat_model(
        model = model_name,
        model_provider="openai",
        api_key = api_key,
        base_url = base_url,
        temperature = 0.0,
    )

@tool
def search_vector_db_stats(db_name: str) -> str:
    '''查询指定向量数据库在2026年的开源星数，主要语言及适用团队规模'''
    db_clean = db_name.lower().strip()
    mock_db = {
        "qdrant": "【Qdrant】GitHub Stars: 23.5k | 语言: Rust | 适用场景: 中小团队首选，单机轻量、内存控制极佳、支持完整过滤。",
        "milvus": "【Milvus】GitHub Stars: 34.2k | 语言: Go/C++ | 适用场景: 大厂超大规模分布式场景，架构组件较多，小团队运维成本较高。",
        "chroma": "【Chroma】GitHub Stars: 19.8k | 语言: Python/TypeScript | 适用场景: 个人开发与原型实验，生产级并发下性能较弱。",
    }
    for k, v in mock_db.items():
        if k in db_clean:
            return v
    return f"未找到指定向量数据库 {db_name} 的信息，建议查询qdrant，milvus或chroma。"

# 制作工具查找字典，根据模型输出的工具名称快速路由到具体执行函数
TOOLS = [search_vector_db_stats]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# =====================================================================
# 第一部分：纯 Python 原生手写 ReAct 循环
# =====================================================================
def run_raw_python_react(user_query: str, max_steps: int = 5) -> str:
    '''
    纯 Python 原生手写 ReAct 循环
    Thought -> Action -> Observation -> 直到Final Answer
    '''
    model = get_chat_model()
    if not model:
        return "第一部分模型初始化失败，请检查环境变量设置。"
    # 将工具绑到模型上，激活Function Calling能力
    model_with_tools = model.bind_tools(TOOLS)
    # 初始化对话上下文
    messages: list[BaseMessage] = [
        SystemMessage(content="你是一名客观的技术选型专家。回答用户问题时，请先调用工具查询准确数据再回答。"),
        HumanMessage(content=user_query)
    ]
    print("\n--- 🚀 [纯 Python ReAct] 开始执行循环 ---")
    step = 0

    while step < max_steps:
        step+=1
        print(f"\n[第 {step} 轮循环] 正在请求大模型决策...")
        


if __name__ == "__main__":
    get_chat_model()
    