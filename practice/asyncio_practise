import asyncio
import aiohttp
import time

async def fetch_url_async(session, url):
    """模拟一个耗时的网络请求（异步版本）"""
    print(f"开始异步获取: {url}")
    # 注意：这里我们使用 aiohttp 的异步 get 方法，并用 await 等待
    async with session.get(url) as response:
        # 模拟处理响应也需要时间
        await asyncio.sleep(2)  # 使用 asyncio.sleep 模拟 I/O 等待，它不会阻塞线程
        text = await response.text()
        print(f"完成异步获取: {url}")
        return f"来自 {url} 的数据 (长度: {len(text)})"

async def main_async():
    urls = ['https://httpbin.org/get', 'https://httpbin.org/delay/1', 'https://httpbin.org/headers']
    
    async with aiohttp.ClientSession() as session:  # 创建异步 HTTP 会话
        # 为每个 URL 创建一个任务（Task）
        tasks = []
        for url in urls:
            # create_task 会将协程加入事件循环，立即开始调度
            task = asyncio.create_task(fetch_url_async(session, url))
            tasks.append(task)
        
        print("所有任务已创建，开始并发执行...")
        
        # 使用 asyncio.gather 并发运行所有任务，并等待它们全部完成
        # gather 返回一个结果列表，顺序与传入的任务顺序一致
        results = await asyncio.gather(*tasks)
        
        return results

if __name__ == "__main__":
    start = time.time()
    # asyncio.run() 是启动事件循环并运行顶层协程的简便方法
    final_results = asyncio.run(main_async())
    end = time.time()
    
    print(f"\n异步版本总耗时: {end - start:.2f} 秒")
    for res in final_results:
        print(res)