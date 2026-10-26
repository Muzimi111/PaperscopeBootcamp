"""
Day 1 实操脚本：PaperScope 核心数据类型与容错工具底座
运行方式：uv run python practice/d1_agent_foundation.py
"""
from dataclasses import dataclass, field
from typing import Callable, Literal, Any

# =====================================================================
# 【语法点 1】Literal 枚举类型：锁死变量的合法取值范围
# 目的：防止后续大模型或代码拼错数据源名字（比如把 "arxiv" 写成 "arXiv_paper"）
# =====================================================================
# 1. 使用 Literal 定义严格的枚举类型（锁死数据源分类）
SourceType = Literal["arxiv", "github", "web", "docs"]

# =====================================================================
# 【语法点 2】@dataclass 数据容器 + Python 3.10 联合类型 (str | None)
# 目的：把零散的字段打包成强类型对象，自带初始化(__init__)和打印格式(__repr__)
# =====================================================================
# 2. 使用 @dataclass 定义证据数据容器（W2 将升级为 Pydantic BaseModel）
@dataclass
class RawEvidence:
    source_type: SourceType #数据源格式
    title: str #论文标题
    url: str #论文链接
    snippet: str #原文逐字摘录
    published_at: str | None = None #论文发表时间，可能为空，默认值为None
    reliability: float = 0.6 #论文可靠性评分，默认值为0.6
    tags: list[str] = field(default_factory=list) #论文标签列表，默认值为空列表

    def __post_init__(self) -> None:
        '''对象创建后自动运行的校验钩子，清理首尾空格并检查数值范围'''
        self.snippet = self.snippet.strip()
        if not (0 <= self.reliability <= 1):
            #raise用来抛出异常，阻止对象创建
            raise ValueError(f"reliability 必须在 0 到 1 之间，当前值为 {self.reliability}") 

    def format_for_llm(self, idx: int) -> str:
        '''将证据对象格式化为 LLM 可读的文本块，便于后续拼接成上下文 prompt'''
        # 关键设计：如果 published_at 为空，则不显示时间字段
        pub = f"|时间：{self.published_at}" if self.published_at else ""
        return (
            f"[{idx}] 来源：{self.source_type} | 标题：{self.title}{pub}\n"
            f"链接：{self.url}\n"
            f"摘要：{self.snippet}\n"
            f"可靠性评分：{self.reliability:.2f}\n"
            f"标签：{', '.join(self.tags) if self.tags else '无'}\n"
        )


# =====================================================================
# 【语法点 3】可测的确定性规则路由（对应主项目 M1 改造点 router.py）
# 面试考点：为什么不所有问题都让大模型自己选工具？
# 答：因为规则路由 0 延迟、0 Token 成本、且 100% 可写单元测试回归！
# =====================================================================
def route_query(question: str) -> SourceType:
    '''根据用户子问题的关键词特征，决定优先调用的垂直数据源'''
    # 关键设计：使用小写化的 question 来做关键词匹配，避免大小写敏感问题
    q_lower = question.lower()
    if any(k in q_lower for k in ["论文", "arxiv", "paper", "提出", "算法原理"]):
        return "arxiv"
    if any(k in q_lower for k in ["github", "代码", "repo", "实现"]):
        return "github"
    if any(k in q_lower for k in ["文档", "docs", "api", "接口"]):
        return "docs"
    return "web" # 默认回退到通用网页搜索

# =====================================================================
# 【语法点 4】12-Factor Agents 原则 #9：把工具报错压缩进上下文，而非崩溃退出
# 目的：当外部 API 报错或大模型传错参数时，捕获异常并转成自然语言提示返回给模型，
#      让模型在下一轮 ReAct 循环中自己看到错误、自己改参数重试！
# =====================================================================
def mock_arxiv_search(query: str, limit: int = 2) -> list[RawEvidence]:
    """
    在 arXiv 论文库中检索 AI 领域的学术论文。

    Args:
        query: 英文检索关键词，不允许为空且不能包含特殊符号 '#'
        limit: 返回的最大论文数量，范围 1-5
    Returns:
        匹配的论文证据列表
    """
    if not query.strip():
        raise ValueError("检索词 query 不能为空字符串！请提供具体的英文技术关键词。")
    if "#" in query:
        raise ValueError(f"arXiv API 不支持包含 '#' 的检索词 ('{query}')，请去掉特殊符号后重试。")
    if len(query) > 50:
            raise ValueError(f"检索词过长（{len(query)}字），请提炼为 3-5 个核心英文或中文关键词后重试。")
    if limit <= 0 or limit > 5:
        raise ValueError("limit 必须在 1 到 5 之间。")

    source = route_query(query)
    # 根据来源赋予不同的基准可信度（论文 > 官方文档 > GitHub > 普通网页）
    score_map: dict[SourceType, float] = {"arxiv": 0.95, "docs": 0.90, "github": 0.85, "web": 0.60}

    # 模拟正常返回结果
    return [
            RawEvidence(
                source_type=source,
                title=f"关于『{query}』的核心检索结果 #{i}",
                url=f"https://example.com/{source}/{i}",
                snippet=f"这是从 {source} 检索到的原文逐字段落，证实了 {query} 的关键性能指标...",
                published_at="2026-09",
                reliability=score_map[source],
                tags=[source, "verified"],
            )
            for i in range(1, limit + 1)
    ]



def safe_tool_wrapper(fn: Callable[..., list[RawEvidence]], **kwargs: Any) -> str:
    '''高阶容错包装器，吞掉异常，吐出大模型能看懂的错误提示'''
    try:
        evidence = fn(**kwargs)
        # 关键设计：如果返回的 evidence 为空列表，也要给出提示，而不是让模型看到空白
        return "\n\n".join(ev.format_for_llm(i) for i, ev in enumerate(evidence, start=1))
    except Exception as e:
        return(
            f"  - 工具 [{fn.__name__}] 执行失败！\n"
            f"  - 你传入的参数: {kwargs}\n"
            f"  - 错误原因: {type(e).__name__}: {e}\n"
            f"  - 修正建议: 请根据上述错误原因调整参数后重新调用工具。"
        )

if __name__ == "__main__":
    print("========== 测试 1：测试确定性路由与正常检索 ==========")
    test_questions = [
        "哪篇 arXiv 论文提出了 GraphRAG？",
        "Qdrant 在 GitHub 上的 star 增长和 issue 维护活跃度如何？",
        "LangGraph 官方文档中 Command 的参数怎么配？",
    ]
    for q in test_questions:
        print(f"\n>>> 子问题: {q} (自动路由至 -> {route_query(q)})")
        print(safe_tool_wrapper(mock_arxiv_search, query=q, limit=1))
    print("\n========== 测试 2：模拟大模型传了超长参数（验证错误自愈反馈） ==========")
    bad_query = "我想知道2026年所有适合中小团队使用的开源向量数据库在十亿级数据量下的内存占用和QPS吞吐量对比情况"
    print(safe_tool_wrapper(mock_arxiv_search, query=bad_query, limit=2))