'''
需要构建一个处理客户指支持的ai agent。客户需求如下：
The agent should:

- Read incoming customer emails
- Classify them by urgency and topic
- Search relevant documentation to answer questions
- Draft appropriate responses
- Escalate complex issues to human agents
- Schedule follow-ups when needed

Example scenarios to handle:

1. Simple product question: "How do I reset my password?"
2. Bug report: "The export feature crashes when I select PDF format"
3. Urgent billing issue: "I was charged twice for my subscription!"
4. Feature request: "Can you add dark mode to the mobile app?"
5. Complex technical issue: "Our API integration fails intermittently with 504 errors"

'''
# Step 1. 将工作流程规划成离散的步骤
'''
Read Email： 提取并解析邮件内容
Classify Intent：使用大型语言模型（LLM）对紧迫性和主题进行分类，然后引导到合适的行动
Doc Search：查询您的知识库以获取相关信息
Bug Track：在追踪系统中创建或更新问题
Draft Reply：生成合适的回应
Human Review：升级到人工代理进行批准或处理
Send Reply：发送邮件回复
'''
# step 2. 确定每个步骤要完成的事
'''
对每个结点，确定它代表哪种类型的操作以及需要什么上下文
LLM步骤：Classify Intenr和Draft Reply
数据步骤：Read Email、Doc Search
行动步骤：Bug Track、Send Reply
用户输入步骤：Human Review
'''
# step 3. 设计状态state
from typing import TypedDict, Literal

class EmailClassification(TypedDict):
    intent: Literal["question", 'bug', 'billing', 'feature', 'complex']
    urgency: Literal['low', 'medium', 'high', 'critical']
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    # 原始邮箱和发件人信息（之后无法重建）
    email_content: str
    sender_email: str
    email_id: str

    # 分类结果（多个后续/下游节点需要）
    classification: EmailClassification | None

    # 搜索结果和客户数据（重新获取成本高）
    search_results: list[str] | None  # List of raw document chunks
    customer_history: dict | None  # Raw customer data from CRM

    # 草稿回应（需通过审核保留）,执行元数据（用于调试和恢复
    draft_response: str | None
    messages: list[str] | None

# step 4. 构建节点
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage


def read_email(state: EmailAgentState) -> dict:
    '''读取邮件内容'''
    return{
        'message': [HumanMessage]
    }


"""
Day 3 每日任务脚本：手写纯 Python ReAct 循环 vs 最小 LangGraph 状态机
运行命令：uv run python practice/d3_react_vs_langgraph.py
"""
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

# 加载 .env 中的 API Key (OPENAI_API_KEY, OPENAI_BASE_URL)
load_dotenv()


# =====================================================================
# 准备工作：初始化统一的大模型与测试工具
# =====================================================================
def get_chat_model():
    """获取兼容 OpenAI 接口的模型（如 DeepSeek、Qwen 或 gpt-4o-mini）"""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPEN_BASE_URL")
    model_name = os.getenv("LLM_MODEL")

    if not api_key:
        print("⚠️ 警告: 未在 .env 检测到 OPENAI_API_KEY，将使用模拟模式运行！")
        return None

    return init_chat_model(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        model_provider="openai",
        temperature=0.0,
    )


@tool
def search_vector_db_stats(db_name: str) -> str:
    """查询指定向量数据库在 2026 年的开源星数、主要语言及适用团队规模。"""
    db_clean = db_name.lower().strip()
    mock_db = {
        "qdrant": "【Qdrant】GitHub Stars: 23.5k | 语言: Rust | 适用场景: 中小团队首选，单机轻量、内存控制极佳、支持完整过滤。",
        "milvus": "【Milvus】GitHub Stars: 34.2k | 语言: Go/C++ | 适用场景: 大厂超大规模分布式场景，架构组件较多，小团队运维成本较高。",
        "chroma": "【Chroma】GitHub Stars: 19.8k | 语言: Python/TypeScript | 适用场景: 个人开发与原型实验，生产级并发下性能较弱。",
    }
    for k, v in mock_db.items():
        if k in db_clean:
            return v
    return f"未找到名为 '{db_name}' 的向量数据库统计信息，建议尝试查询 qdrant / milvus / chroma。"


# 制作工具查找字典：根据模型输出的工具名称快速路由到具体执行函数
TOOLS = [search_vector_db_stats]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# =====================================================================
# 第一部分：纯 Python 原生手写 ReAct 循环（约 50 行，直击本质）
# =====================================================================
def run_raw_python_react(user_query: str, max_steps: int = 5) -> str:
    """
    不依赖任何 Agent 框架，用最质朴的 while 循环实现 ReAct：
    Thought(模型决策) -> Action(调工具) -> Observation(回填ToolMessage) -> 直到 Final Answer
    """
    model = get_chat_model()
    if not model:
        return "[模拟] 纯 Python ReAct: 缺少 API Key，跳过真实调用。"

    # 将工具 Schema 绑定到模型上，激活 Function Calling 能力
    model_with_tools = model.bind_tools(TOOLS)

    # 初始化对话上下文
    messages: list[BaseMessage] = [
        SystemMessage(content="你是一名客观的技术选型专家。回答用户问题时，请先调用工具查询准确数据再回答。"),
        HumanMessage(content=user_query),
    ]

    print("\n--- 🚀 [纯 Python ReAct] 开始执行循环 ---")
    step = 0

    while step < max_steps:
        step += 1
        print(f"\n[第 {step} 轮循环] 正在请求大模型决策...")
        response: AIMessage = model_with_tools.invoke(messages)
        messages.append(response)

        # 检查模型是否发起了工具调用（Action）
        tool_calls = getattr(response, "tool_calls", None)

        if not tool_calls:
            # 模型没有调工具，说明它认为已有足够信息可以给出最终答案了（Final Answer）
            print("  -> 模型未发起工具调用，已得出最终结论！")
            return str(response.content)

        # 遍历执行模型要求调用的每一个工具（Observation）
        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            call_id = tc["id"]

            print(f"  -> 模型请求调用工具: [{tool_name}], 参数: {tool_args}")
            if tool_name in TOOLS_BY_NAME:
                tool_result = TOOLS_BY_NAME[tool_name].invoke(tool_args)
            else:
                tool_result = f"Error: 找不到名为 {tool_name} 的工具。"

            print(f"  -> 工具执行结果: {tool_result}")
            # 将工具结果构造成标准的 ToolMessage 喂回上下文
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=call_id, name=tool_name))

    return "⚠️ 达到最大步数限制（熔断），未能在规定步数内完成研究。"


# =====================================================================
# 第二部分：使用 LangGraph 状态机重构同一个需求
# =====================================================================

# 1. 定义图的全局状态 State
# 注意：messages 字段用了 Annotated[list[BaseMessage], operator.add]
# 这叫 Reducer（归约器）：节点返回的新消息不会覆盖旧消息，而是自动 append 追加！
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    research_summary: str


# 2. 定义节点 A：调用模型节点 (llm_node)
def llm_node(state: AgentState) -> dict:
    model = get_chat_model()
    model_with_tools = model.bind_tools(TOOLS)
    response = model_with_tools.invoke(state["messages"])
    # 按照约定，Node 只需返回它想要更新的 Key
    return {"messages": [response]}


# 3. 定义节点 B：工具执行节点 (tool_node)
def tool_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []
    for tc in getattr(last_message, "tool_calls", []):
        tool_fn = TOOLS_BY_NAME[tc["name"]]
        obs = tool_fn.invoke(tc["args"])
        tool_messages.append(ToolMessage(content=str(obs), tool_call_id=tc["id"], name=tc["name"]))
    return {"messages": tool_messages}


# 4. 定义条件路由边：决定下一步是继续调工具，还是结束
def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END


# 5. 组装 StateGraph
def build_minimal_graph():
    workflow = StateGraph(AgentState)

    # 注册节点
    workflow.add_node("llm_node", llm_node)
    workflow.add_node("tool_node", tool_node)

    # 连接拓扑边
    workflow.add_edge(START, "llm_node")
    workflow.add_conditional_edges(
        "llm_node",
        should_continue,
        {
            "tool_node": "tool_node",
            END: END,
        },
    )
    workflow.add_edge("tool_node", "llm_node")  # 工具执行完，形成环路重新切回模型

    return workflow.compile()


# =====================================================================
# 第三部分：结构化输出加餐（with_structured_output）
# 面试高频：如何保证最终给前端的报告一定是强类型 JSON，而不是散装自然语言？
# =====================================================================
class TechComparisonReport(BaseModel):
    title: str = Field(description="调研报告标题")
    recommended_option: str = Field(description="最终推荐的选型（如 Qdrant）")
    key_reasons: list[str] = Field(description="推荐的 3 条核心理由")
    warning_or_pitfalls: str = Field(description="潜在风险或运维警告")


def demo_structured_output(raw_text: str):
    print("\n--- 🛡️ [结构化输出测试 with_structured_output] ---")
    model = get_chat_model()
    if not model:
        return
    structured_llm = model.with_structured_output(TechComparisonReport)
    prompt = f"请根据以下调研结论，提炼为一份规范的选型报告：\n{raw_text}"
    report: TechComparisonReport = structured_llm.invoke(prompt)
    print("成功解析为强类型 Pydantic 对象：")
    print(report.model_dump_json(indent=2))


# =====================================================================
# 主函数：对比运行
# =====================================================================
if __name__ == "__main__":
    query = "我们是 5 人的初创 AI 团队，想在 Qdrant 和 Milvus 之间做个选型，哪个更合适？为什么？"

    # 1. 跑手写 Python ReAct
    result_raw = run_raw_python_react(query)
    print("\n>>> 纯 Python ReAct 最终输出结果:")
    print(result_raw)

    # 2. 跑 LangGraph 状态机
    print("\n--- 🕸️ [LangGraph 状态机] 开始执行 ---")
    if get_chat_model():
        app = build_minimal_graph()
        initial_state = {
            "messages": [
                SystemMessage(content="你是一名客观的技术选型专家。回答用户问题时，请先调用工具查询准确数据再回答。"),
                HumanMessage(content=query),
            ],
            "research_summary": "",
        }
        final_state = app.invoke(initial_state, config={"recursion_limit": 10})
        result_graph = final_state["messages"][-1].content
        print("\n>>> LangGraph 最终输出结果:")
        print(result_graph)

        # 3. 将结果结构化
        demo_structured_output(str(result_graph))