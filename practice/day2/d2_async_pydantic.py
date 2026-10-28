"""
Day 2 每日任务脚本：Pydantic 运行时强校验 + asyncio 异步并发四件套实战
运行命令：uv run python practice/d2_async_pydantic.py
"""
import asyncio
import time
from typing import Literal
from pydantic import BaseModel, Field, ValidationError, field_validator



# =====================================================================
# 第一部分：定义生产级 Pydantic 数据契约
# =====================================================================
SourceType = Literal["arxiv", "github", "web", "docs"]

# 将day1的静态dataclass改为BaseModel
class Evidence(BaseModel):
    source_type: SourceType = Field(
        description = "证据来源分类：arxiv(论文)、github(代码库)、web(网页)、docs(官方文档)"
    )
    title: str = Field(min_length=2, description="文献、代码库或网页的完整标题")
    url: str = Field(description="以 http:// 或 https:// 开头的完整可访问链接")
    snippet: str = Field(
        description="支持结论的原文逐字摘录（严禁模型自己改写或编造，后续用于 Citation Guard 核验）"
    )
    published_at: str|None = Field(default=None, description="发表或最后更新日期，如 2026-09")
    reliability: float = Field(
        default = 0.6,
        ge = 0.0,  # 强制要求大于等于 0.0
        le = 1.0,  # 强制要求小于等于 1
        description = "证据可信度评分，范围 0.0 ~ 1.0",
    )
    citation_count: int = Field(default = 0, ge=0)

    # 利用Pydantic的field_validator装饰器来定义自定义字段校验器
    @field_validator("url")
    @classmethod
    def validate_url_prefix(cls, value: str) -> str:
        clean_url = value.strip()
        if not clean_url.startswith(("http://", "https://")):
            raise ValueError(f"URL 必须以 http:// 或 https:// 开头，当前非法值: '{clean_url}'")
        return clean_url

    @field_validator("snippet")
    @classmethod
    def validate_snippet_not_trivial(cls, value: str) -> str:
        clean_text = value.strip()
        if len(clean_text) < 10:
            raise ValueError(f"原文摘录 snippet 过短（仅 {len(clean_text)} 字），必须包含完整的事实句。")
        return clean_text
    @field_validator("title")
    @classmethod
    def validate_title_not_test(cls, value: str) -> str:
        clean_title = value.strip()
        if "test" in clean_title :
            raise ValueError(f"标题中不能包含测试字样")
        return clean_title


class SubQueryTask(BaseModel):
    """Supervisor 拆分出的单个子课题任务"""
    query: str = Field(description="具体的检索关键词")
    target_source: SourceType = Field(description="路由到的目标垂直数据源")
    delay_seconds: float = Field(default=1.5, description="模拟该数据源的网络 I/O 响应耗时")


# =====================================================================
# 第二部分：asyncio 异步并发与容错
# =====================================================================
async def mock_fetch_source(task: SubQueryTask) -> Evidence:
    """模拟异步请求外部 API"""
    print(f"  [开始请求] ({task.target_source.upper()}) 检索: '{task.query}' (耗时 {task.delay_seconds}s)...")
    await asyncio.sleep(task.delay_seconds)  # 非阻塞挂起，交出 CPU 控制权

    # 模拟外部接口偶发 503 异常
    if "崩溃测试" in task.query:
        raise ConnectionError(f"{task.target_source} 远端服务器返回 503 Service Unavailable")

    score_map: dict[SourceType, float] = {"arxiv": 0.95, "docs": 0.90, "github": 0.85, "web": 0.60}
    print(f"  [完成响应] ({task.target_source.upper()}) 成功拿回: '{task.query}'")

    return Evidence(
        source_type=task.target_source,
        title=f"[{task.target_source.upper()}] {task.query} 核心技术调研报告",
        url=f"https://example.com/{task.target_source}/{ abs(hash(task.query)) % 10000 }",
        snippet=f"根据 {task.target_source} 官方实测数据，{task.query} 在千万级向量规模下 P99 延迟低于 12ms。",
        published_at="2026-10",
        reliability=score_map[task.target_source],
    )


async def bounded_fetch(
    task: SubQueryTask,
    semaphore: asyncio.Semaphore,
    timeout: float = 2.5,
) -> Evidence:
    """
    生产级异步包装模板：
    结合 Semaphore（限流）+ wait_for（单任务超时熔断）
    """
    async with semaphore:
        return await asyncio.wait_for(mock_fetch_source(task), timeout=timeout)


async def run_serial_vs_parallel_benchmark() -> None:
    """实验：对比传统串行 vs asyncio.gather 并发耗时"""
    tasks_data = [
        SubQueryTask(query="GraphRAG 社区发现算法原理", target_source="arxiv", delay_seconds=1.5),
        SubQueryTask(query="Qdrant vs Milvus GitHub 活跃度", target_source="github", delay_seconds=1.5),
        SubQueryTask(query="LangGraph StateGraph Command API 文档", target_source="docs", delay_seconds=1.5),
    ]
    '''
    print("\n================ 实验 A：传统串行逐个请求 ================")
    t0 = time.perf_counter()
    serial_results: list[Evidence] = []
    # 伪异步，串行执行
    for t in tasks_data:
        ev = await mock_fetch_source(t)
        serial_results.append(ev)
    serial_cost = time.perf_counter() - t0
    print(f"👉 串行总耗时: {serial_cost:.2f} 秒 (各任务耗时相加: 1.5 + 1.5 + 1.5 = 4.5s)\n")

    print("================ 实验 B：asyncio.gather 并行请求 ================")
    t1 = time.perf_counter()
    sem = asyncio.Semaphore(3)
    # 使用 asyncio.gather 并行执行
    coroutines = [bounded_fetch(t, sem, timeout=3.0) for t in tasks_data]
    parallel_results = await asyncio.gather(*coroutines)
    parallel_cost = time.perf_counter() - t1
    print(f"👉 并行总耗时: {parallel_cost:.2f} 秒 (取决于最慢的单个任务: ~1.5s，提速 {serial_cost / parallel_cost:.1f} 倍！)\n")


async def run_fault_tolerance_demo() -> None:
    """实验：Semaphore 限流排队 + 超时熔断 + 异常隔离"""
    print("================ 实验 C：并发限流(Semaphore=2) + 超时与异常隔离 ================")
    mixed_tasks = [
        SubQueryTask(query="正常论文检索 A", target_source="arxiv", delay_seconds=1.0),
        SubQueryTask(query="慢速卡死网页 B（触发超时）", target_source="web", delay_seconds=3.0),
        SubQueryTask(query="远端崩溃测试 C", target_source="github", delay_seconds=0.8),
        SubQueryTask(query="正常文档检索 D（排队等待通行证）", target_source="docs", delay_seconds=1.0),
    ]

    # 限制并发数为 2，单任务超时为 1.8 秒
    sem = asyncio.Semaphore(2)
    coros = [bounded_fetch(t, sem, timeout=1.8) for t in mixed_tasks]

    # return_exceptions=True 确保单个任务崩溃不影响整体
    raw_outcomes = await asyncio.gather(*coros, return_exceptions=True)

    valid_evidences: list[Evidence] = []
    error_logs: list[str] = []

    for task, outcome in zip(mixed_tasks, raw_outcomes):
        if isinstance(outcome, Exception):
            err_msg = f"❌ 子任务 ['{task.query}'] 失败 -> {type(outcome).__name__}: {outcome}"
            error_logs.append(err_msg)
        else:
            valid_evidences.append(outcome)

    print(f"\n📊 汇总结果：成功回收 {len(valid_evidences)} 条有效证据，平稳隔离 {len(error_logs)} 个异常：")
    for ev in valid_evidences:
        print(f"  ✅ 有效证据: {ev.title} (可信度: {ev.reliability})")
    for err in error_logs:
        print(f"  {err}")

'''
def demo_pydantic_validation() -> None:
    """实验：Pydantic 拦截大模型常见格式错误"""
    print("================ 实验 D：Pydantic 强校验展示 ================")
    # 正常数据自动转换类型（字符串 "0.92" 自动转 float）
    good_ev = Evidence(
        source_type="arxiv",
        title="Attention Is All You Need ",
        url="https://arxiv.org/abs/1706.03762",
        snippet="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
        reliability="0.92",  # type: ignore[arg-type]
        citation_count=5,
    )
    print(f"正常数据校验通过 (自动转为 float): reliability={good_ev.reliability}")
    print(f"该数据的引用书为：{good_ev.citation_count}")

    # 故意传入包含 4 处格式错误的字典，看 Pydantic 一次性拦截
    print("\n--- 模拟大模型常见的 4 处格式错误 ---")
    try:
        Evidence(
            source_type="wechat_article",  # 错误 1：非指定枚举
            title="need test",
            url="www.arxiv.org/123",       # 错误 2：缺少 https://
            snippet="太短了",               # 错误 3：不足 10 字
            reliability=1.5,               # 错误 4：超出 1.0 上限
        )
    except ValidationError as e:
        errors = e.errors()
        print(f"🛡️ Pydantic 成功拦截了 {len(errors)} 处数据错误：")
        for err in errors:
            loc = ".".join(str(x) for x in err["loc"])
            print(f"  - 字段 [{loc}]: {err['msg']}")

async def batch_verify_claims(claims: list[str]) -> list[bool]:
    '''模拟批量验证大模型生成的声明是否符合事实'''
    print(f"\n================ 实验 E：批量验证大模型声明的真实性 ================")
    sem = asyncio.Semaphore(3)
    coros = [bounded_fetch(SubQueryTask(query=claim, target_source="docs", delay_seconds=0.5), sem, timeout=2.0) for claim in claims]
    start_time = time.perf_counter() #perf_counter() 是一个高精度计时器，适合测量短时间间隔
    results = await asyncio.gather(*coros, return_exceptions=True)
    end_time = time.perf_counter()
    print(f"批量验证 {len(claims)} 条声明耗时: {end_time - start_time:.2f} 秒")

async def main() -> None:
    demo_pydantic_validation()
    await run_serial_vs_parallel_benchmark()
    # await run_fault_tolerance_demo()
    await batch_verify_claims([
        "LangGraph StateGraph Command API 文档",
        "GraphRAG 社区发现算法原理",
        "Qdrant vs Milvus GitHub 活跃度",
        "慢速卡死网页 B（触发超时）",
        "远端崩溃测试 C",
    ])

if __name__ == "__main__":
    asyncio.run(main())