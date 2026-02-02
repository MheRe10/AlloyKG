import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.zhipu import zhipu_complete_if_cache, zhipu_embedding
from lightrag.utils import EmbeddingFunc
import os


async def main():
    # Set up API configuration
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ZHIPU_API_KEY. Set it as an environment variable (or create a local .env file)."
        )

    base_url = os.getenv("ZHIPU_BASE_URL", "")  # Optional; kept for commented-out OpenAI example

    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir="../data/end_to_end/rag_storage",
        parser="mineru",  # Parser selection: mineru or docling
        parse_method="auto",  # Parse method: auto, ocr, or txt
        enable_image_processing=False,  
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Define LLM model function
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await zhipu_complete_if_cache(
            prompt,
            model="glm-4.5-flash",  
            api_key=api_key,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )
    
    async def vision_model_func(
        prompt, system_prompt=None, history_messages=[], 
        image_data=None, messages=None, **kwargs
    ):
        if messages or image_data:
            print("Vision model function with multimodal input is not implemented yet.")
            return "Image processing is not implemented yet."
        else:
            return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)
        
        
    #TODO: Define vision model function for image processing
    
    # def vision_model_func(
    #     prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    # ):
    #     # If messages format is provided (for multimodal VLM enhanced query), use it directly
    #     if messages:
    #         return openai_complete_if_cache(
    #             "gpt-4o",
    #             "",
    #             system_prompt=None,
    #             history_messages=[],
    #             messages=messages,
    #             api_key=api_key,
    #             base_url=base_url,
    #             **kwargs,
    #         )
    #     # Traditional single image format
    #     elif image_data:
    #         return openai_complete_if_cache(
    #             "gpt-4o",
    #             "",
    #             system_prompt=None,
    #             history_messages=[],
    #             messages=[
    #                 {"role": "system", "content": system_prompt}
    #                 if system_prompt
    #                 else None,
    #                 {
    #                     "role": "user",
    #                     "content": [
    #                         {"type": "text", "text": prompt},
    #                         {
    #                             "type": "image_url",
    #                             "image_url": {
    #                                 "url": f"data:image/jpeg;base64,{image_data}"
    #                             },
    #                         },
    #                     ],
    #                 }
    #                 if image_data
    #                 else {"role": "user", "content": prompt},
    #             ],
    #             api_key=api_key,
    #             base_url=base_url,
    #             **kwargs,
    #         )
    #     # Pure text format
    #     else:
    #         return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
    
    

    # Define embedding function
    embedding_func = EmbeddingFunc(
        embedding_dim=2048,
        max_token_size=8192,
        func=lambda texts: zhipu_embedding(
            texts,
            model="embedding-3",
            api_key=api_key,
        ),
    )

    # Initialize RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # Process a document
    raw_paper_dir = "../data/raw_paper"
    for filename in os.listdir(raw_paper_dir):
        file_path = os.path.join(raw_paper_dir, filename)
        if os.path.isfile(file_path):
            print(f"Processing file: {file_path}")
            await rag.process_document_complete(
                file_path=file_path,
                output_dir="../data/end_to_end/processed_paper",
                parse_method="auto"
            )

    # Query the processed content
    # Pure text query - for basic knowledge base search
    text_result = await rag.aquery(
        "根据论文，使用TiO2和TiH2作为起始材料制备Fe-Ti合金的主要区别是什么？",  # EN: Example query in Chinese: "According to the paper, what is the main difference in preparing Fe-Ti alloys using TiO2 and TiH2 as starting materials?"
        mode="hybrid",
        enable_rerank=False
    )
    print("Text query result:", text_result)

    # Multimodal query with specific multimodal content
#     multimodal_result = await rag.aquery_with_multimodal(
#     "Explain this formula and its relevance to the document content",
#     multimodal_content=[{
#         "type": "equation",
#         "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
#         "equation_caption": "Document relevance probability"
#     }],
#     mode="hybrid",
#     enable_rerank=False
# )
#     print("Multimodal query result:", multimodal_result)

if __name__ == "__main__":
    asyncio.run(main())