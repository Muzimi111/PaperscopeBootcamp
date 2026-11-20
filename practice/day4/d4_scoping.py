"""
Day 4 每日任务脚本：精读并复现 1_scoping.ipynb（用户澄清 + Command 动态路由 + 研究简报生成）
运行命令：uv run python practice/d4_scoping.py
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv()


# =====================================================================
# 1. 初始化模型（兼容 DeepSeek / Qwen / OpenAI）
# =====================================================================
def get_model():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPEN_BASE_URL")
    model_name = os.getenv("LLM_MODEL")
    if not api_key:
        raise RuntimeError("请先在项目根目录的 .env 文件中配置 OPENAI_API_KEY！")
    return init_chat_model(model=model_name, api_key=api_key, base_url=base_url, model_provider="openai", temperature=0.0)


def get_today_str() -> str:
    """获取当前日期字符串，注入 Prompt 防止模型时间错乱"""
    return datetime.now().strftime("%Y-%m-%d")


# =====================================================================
# 2. 定义 Scoping 阶段的 State 与 Pydantic 结构化输出 Schema
# （与 notebooks/1_scoping.ipynb 保持一致）
# =====================================================================
class AgentState(MessagesState):
    """
    继承自 LangGraph 内置的 MessagesState（自带 messages: Annotated[list[AnyMessage], add_messages]）
    在此基础上扩展 research_brief 字段，用于向后续的 Researcher 传递研究简报。
    """
    research_brief: str | None


class ClarifyWithUser(BaseModel):
    """节点 1 的结构化输出：判断是否需要向用户追问澄清"""
    need_clarification: bool = Field(
        description="是否需要向用户提出澄清问题（若问题过于宽泛缺乏核心边界则为 True，若已足够明确则为 False）"
    )
    question: str = Field(
        description="当 need_clarification=True 时，向用户提出的简短、聚焦的澄清问题（不超过 2 个小问）；若为 False 则填固定空字符串"
    )
    verification: str = Field(
        description="当 need_clarification=False 时，向用户确认即将开始调研的简短回复；若为 True 则填固定空字符串"
    )


class ResearchQuestion(BaseModel):
    """节点 2 的结构化输出：将多轮对话浓缩为结构化研究简报"""
    research_brief: str = Field(
        description="一份高度结构化、细节明确的研究简报（包含调研背景、核心对比维度、约束条件与待解答的子问题），将直接作为指令发给下游检索 Agent。"
    )


# =====================================================================
# 3. 定义图节点（使用 Command 实现“状态更新 + 路由跳转”二合一）
# =====================================================================
def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    节点 1：审视对话历史，决定是停下来追问用户（goto=END），还是进入简报生成（goto="write_research_brief"）
    """
    structured_model = get_model().with_structured_output(ClarifyWithUser)
    conversation_str = get_buffer_string(state["messages"])

    prompt = (
        f"今天是 {get_today_str()}。你是一名严谨的 AI 技术研究助手。\n"
        f"请阅读以下用户与助手的对话历史，判断用户的调研请求是否已经具备足够具体的边界：\n"
        f"<对话历史>\n{conversation_str}\n</对话历史>\n\n"
        f"判断准则：\n"
        f"1. 如果用户只丢出一个宽泛的大词（如'帮我调研向量数据库'、'看看大模型论文'），没有任何应用场景、对比对象或侧重点，且你之前还没追问过，请将 need_clarification 设为 True，并在 question 中提出简短、一针见血的澄清问题。\n"
        f"2. 如果用户的问题已经很具体，或者用户已经回答了你上一轮的澄清问题，请务必将 need_clarification 设为 False（绝不反复纠缠追问！），并在 verification 中简短确认即将启动研究。"
    )

    response: ClarifyWithUser = structured_model.invoke([HumanMessage(content=prompt)])

    if response.need_clarification:
        print(f"  🤔 [Clarify 节点判定] 需求过于宽泛 -> 触发追问并暂停图执行 (goto=END)")
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]},
        )
    else:
        print(f"  ✅ [Clarify 节点判定] 需求已明确 -> 流转至简报生成节点 (goto='write_research_brief')")
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]},
        )


def write_research_brief(state: AgentState) -> dict:
    """
    节点 2：将对话历史转化为干净、无歧义的 Research Brief
    """
    structured_model = get_model().with_structured_output(ResearchQuestion)
    conversation_str = get_buffer_string(state["messages"])

    prompt = (
        f"今天是 {get_today_str()}。请根据以下完整对话记录，提炼出一份供下游检索 Agent 使用的《研究简报 (Research Brief)》。\n"
        f"<对话历史>\n{conversation_str}\n</对话历史>\n\n"
        f"要求：\n"
        f"- 明确指出调研的核心对象、时间范围、应用场景约束；\n"
        f"- 拆解出 3~5 个需要检索核实的具体维度（如论文算法原理、GitHub 社区维护活跃度、架构性能指标等）；\n"
        f"- 严禁主观臆造对话中未提及的额外限制。"
    )

    response: ResearchQuestion = structured_model.invoke([HumanMessage(content=prompt)])
    return {"research_brief": response.research_brief}


# =====================================================================
# 4. 组装并编译带 MemorySaver 检查点的 Scoping 图
# =====================================================================
def build_scoping_graph():
    builder = StateGraph(AgentState)

    # 注册两个核心节点
    builder.add_node("clarify_with_user", clarify_with_user)
    builder.add_node("write_research_brief", write_research_brief)

    # 注意：因为 clarify_with_user 返回了 Command[Literal["write_research_brief", "__end__"]]，
    # 我们甚至不需要写 builder.add_conditional_edges(...)！LangGraph 会自动识别跳转关系！
    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("write_research_brief", END)

    # 挂载 MemorySaver 检查点，使图能够记住同一个 thread_id 下的历史对话
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# =====================================================================
# 5. 顺手攒下前 2 道评测集候选题目（写入 eval/questions_draft.jsonl）
# =====================================================================
def save_day4_eval_questions():
    eval_dir = Path("eval")
    eval_dir.mkdir(exist_ok=True)
    target_file = eval_dir / "questions_draft.jsonl"

    day4_questions = [
        {
            "id": "Q01",
            "category": "tech_selection",
            "question": "2026年针对 5 人规模的初创团队、千万级 768 维向量检索场景（要求支持混合检索与低运维成本），Qdrant 与 Milvus 应该如何选型？",
            "expected_sources": ["github", "docs", "web"],
            "rubric_notes": "需覆盖单机部署资源占用、Rust vs Go/C++架构差异、GitHub Issue响应活跃度、混合检索(Sparse+Dense)支持情况",
        },
        {
            "id": "Q02",
            "category": "paper_survey",
            "question": "梳理 GraphRAG（基于知识图谱的检索增强生成）方向的核心代表性论文，分析其相比传统向量切块 RAG 在全局摘要（Global Sensemaking）问题上的优势与 Token 成本代价。",
            "expected_sources": ["arxiv", "github"],
            "rubric_notes": "需覆盖微软 Edge et al. GraphRAG 原论文、Leiden 社区发现算法机制、建索引阶段 Token 消耗高的局限性",
        },
    ]

    with open(target_file, "w", encoding="utf-8") as f:
        for item in day4_questions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n📁 已自动将前 2 道黄金评测题写入: {target_file}")


# =====================================================================
# 主程序：测试场景 A（模糊问题多轮澄清）与 场景 B（明确问题直接放行）
# =====================================================================
if __name__ == "__main__":
    app = build_scoping_graph()

    print("=================================================================")
    print("场景 A：用户输入极度宽泛的问题 -> 触发追问中断 -> 用户补充后再生成简报")
    print("=================================================================")
    # 使用 thread_id="session_vague" 保持多轮会话状态
    config_a = {"configurable": {"thread_id": "session_vague"}}

    # 第一轮：故意输入一句含糊不清的话
    user_turn_1 = "帮我调研一下向量数据库。"
    print(f"\n👤 用户 (第 1 轮): {user_turn_1}")
    state_after_turn_1 = app.invoke({"messages": [HumanMessage(content=user_turn_1)]}, config=config_a)
    print(f"🤖 Agent 追问: {state_after_turn_1['messages'][-1].content}")
    print(f"📋 当前 research_brief 状态: {state_after_turn_1.get('research_brief')}")

    # 第二轮：带着同一个 config_a (相同 thread_id) 回答 Agent 的追问
    user_turn_2 = "我们是 5 人初创团队，大概有 1000 万条文本向量，希望对比 Qdrant 和 Milvus 的运维成本和混合检索能力。"
    print(f"\n👤 用户 (第 2 轮补充): {user_turn_2}")
    state_after_turn_2 = app.invoke({"messages": [HumanMessage(content=user_turn_2)]}, config=config_a)
    print(f"🤖 Agent 确认: {state_after_turn_2['messages'][-1].content}")
    print(f"\n📋 最终生成的 Research Brief:\n{state_after_turn_2.get('research_brief')}")

    print("\n=================================================================")
    print("场景 B：用户一开始就输入非常具体的问题 -> 无需追问，一步直达生成简报")
    print("=================================================================")
    config_b = {"configurable": {"thread_id": "session_clear"}}
    clear_query = "请调研 arXiv 上关于微软 GraphRAG 论文的核心社区聚类算法原理，并对比它与普通 Naive RAG 的优缺点。"
    print(f"\n👤 用户: {clear_query}")
    state_b = app.invoke({"messages": [HumanMessage(content=clear_query)]}, config=config_b)
    print(f"🤖 Agent 确认: {state_b['messages'][-1].content}")
    print(f"\n📋 最终生成的 Research Brief:\n{state_b.get('research_brief')}")

    # 顺手保存今天的 2 道评测题
    save_day4_eval_questions()