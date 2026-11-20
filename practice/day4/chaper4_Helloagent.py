import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 加载.env 文件中的环境变量
load_dotenv()

class HelloAgentsLLM:
    '''
    定义一个专属的LLM客户端类。这个类将封装所有与模型服务交互的细节，让我们的主逻辑可以更专注于智能体的构建。
    '''
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        '''
        初始化客户端，优先使用传入参数，如果未提供，则从环境变量中读取。
        '''
        self.model = model or os.getenv("LLM_MODEL")
        # 不加self是因为我们希望在实例化时就确定apiKey的值，而不是在后续方法中动态获取
        apiKey = apiKey or os.getenv("OPENAI_API_KEY")
        baseUrl = baseUrl or os.getenv("OPEN_BASE_URL")
        timeout = timeout or int(os.getenv("OPENAI_TIMEOUT", 30))  # 默认超时30秒

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("请确保在 .env 文件中设置了 OPENAI_API_KEY、OPEN_BASE_URL 和 LLM_MODEL 环境变量，或者在初始化时传入这些参数。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        '''
        调用大语言模型进行思考并返回其响应
        '''
        print(f"--- 💭 [LLM] 正在思考，使用模型: {self.model} ---")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True  # 开启流式输出
            )
            # 处理流式输出
            print("--- 📝 [LLM] 响应流式输出开始 ---")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ''
                print(content, end='', flush=True)  # 实时打印输出
                collected_content.append(content)
            print("\n--- 📝 [LLM] 响应流式输出结束 ---")
            return ''.join(collected_content)

        except Exception as e:
            print(f"--- ❌ [LLM] 思考失败: {e} ---")
            return None

if __name__ == "__main__":
    try:
            llmClient = HelloAgentsLLM()
            
            exampleMessages = [
                {"role": "system", "content": "You are a helpful assistant that writes Python code."},
                {"role": "user", "content": "写一个快速排序算法"}
            ]
            
            print("--- 调用LLM ---")
            responseText = llmClient.think(exampleMessages)
            if responseText:
                print("\n\n--- 完整模型响应 ---")
                print(responseText)

    except ValueError as e:
            print(e)