import asyncio
import os
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.lightrag import LightRAG
from zhipuai import ZhipuAI
from lightrag.utils import EmbeddingFunc
from raganything.modalprocessors import ImageModalProcessor, TableModalProcessor, GenericModalProcessor

# -------------------------
# set up API and paths
# -------------------------
API_KEY = os.getenv("ZHIPU_API_KEY")
RAG_STORAGE = "../data/raw_paper"
OUTPUT_DIR = "../data/processed_paper"

# -------------------------
# LightRAG initialization
# -------------------------
client = ZhipuAI(api_key=API_KEY)

# Zhipu GLM-4 async call
async def glm4_complete(prompt: str, system_prompt=None, history_messages=[], **kwargs):
    loop = asyncio.get_event_loop()
    def sync_call():
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            for msg in history_messages:
                messages.append(msg)
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="glm-4",
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
        )
        return response.choices[0].message.content
    return await loop.run_in_executor(None, sync_call)

vision_model_func = glm4_complete

async def main():
    config = RAGAnythingConfig(
        working_dir="../data/rag_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=glm4_complete,
        vision_model_func=vision_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: None  # Zhipu embedding function can be added here
        ),
    )

    await rag.process_folder_complete(
        folder_path="../data/raw_paper",     
        output_dir="../data/processed_paper",  
        file_extensions=[".pdf"], 
        recursive=True,
        max_workers=4
    )

if __name__ == "__main__":
    asyncio.run(main())