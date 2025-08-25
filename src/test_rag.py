import os
import asyncio
from lightrag.lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from zhipuai import ZhipuAI


API_KEY = os.getenv("ZHIPU_API_KEY")  # 建议用环境变量存 API key
client = ZhipuAI(api_key=API_KEY)

# 自定义一个 LLM 调用函数，替代 openai_complete_if_cache
async def glm4_complete(prompt: str, **kwargs) -> str:
    loop = asyncio.get_event_loop()
    def sync_call():
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "system", "content": "You are an assistant for alloy engineering knowledge extraction."},
                {"role": "user", "content": prompt}
            ],
            temperature=kwargs.get("temperature", 0.2),
        )
        return response.choices[0].message.content
    return await loop.run_in_executor(None, sync_call)



async def main():
    # 初始化 LightRAG
    rag = LightRAG(
        working_dir="./rag_test_storage",
        llm_model_func=glm4_complete,   # 用我们自定义的函数
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,   # GLM-4 Embedding 的维度
            max_token_size=8192,
            func=lambda texts: None  # 先不做嵌入
        )
    )

    await rag.initialize_storages()

    # 测试生成
    prompt = "Please explain what is alloy?"
    result = await rag.llm_model_func(prompt)
    print("GLM-4.0 result:", result)

if __name__ == "__main__":
    asyncio.run(main())
