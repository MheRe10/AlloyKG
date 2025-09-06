import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.zhipu import zhipu_complete_if_cache, zhipu_embedding
from lightrag.utils import EmbeddingFunc
import os


async def main():
    # Set up API configuration
    api_key = "a3d236a6017d4cfc9f15c509a3e7c786.eZIIBa9QmssnGEsv"
    base_url = "your-base-url"  # Optional

    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir="../data/end_to_end/rag_storage",
        parser="mineru",  # Parser selection: mineru or docling
        parse_method="auto",  # Parse method: auto, ocr, or txt
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Define LLM model function
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return zhipu_complete_if_cache(
            prompt,
            model="glm-4",  
            api_key=api_key,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )
    
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], 
        image_data=None, messages=None, **kwargs
    ):
        if messages or image_data:
            print("Vision model function with multimodal input is not implemented yet.")
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
        
        
    #TODO: Define vision model function for image processing
    """
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # If messages format is provided (for multimodal VLM enhanced query), use it directly
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Traditional single image format
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ],
                    }
                    if image_data
                    else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Pure text format
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
    """
    

    # Define embedding function
    embedding_func = EmbeddingFunc(
        embedding_dim=2048,  # 按 zhipu_embedding 的默认
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
        "What are the main findings shown in the figures and tables?",
        mode="hybrid"
    )
    print("Text query result:", text_result)

    # Multimodal query with specific multimodal content
    multimodal_result = await rag.aquery_with_multimodal(
    "Explain this formula and its relevance to the document content",
    multimodal_content=[{
        "type": "equation",
        "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
        "equation_caption": "Document relevance probability"
    }],
    mode="hybrid"
)
    print("Multimodal query result:", multimodal_result)

if __name__ == "__main__":
    asyncio.run(main())