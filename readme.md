#激活虚拟环境
venv_alloykg\Scripts\activate

#更新依赖 pip freeze > requirements.txt
pip install -r requirements.txt

#Enter ZHIPU API KEY

$env:ZHIPU_API_KEY="<YOUR_ZHIPU_API_KEY>"

# -------------------------

# Embedding fine-tuning (TSDAE)

# -------------------------

# Train a domain-adapted embedding model from parsed paper JSON (no labels required)

python src/train_embedding.py --input-glob "data/paper_json/**/*_content_list.json" --output-dir models/tsdae-embedding

# Use the trained embedding model in RAG

$env:LOCAL_EMBEDDING_MODEL_DIR="models/tsdae-embedding"

# -------------------------

# Offline retrieval-only (no LLM / no API key)

# -------------------------

# Print top-k matching chunks from data/end_to_end/rag_storage using the local embedding model.

python src/ragall_offline.py --query "According to the paper, what's the difference of using TiO2 or TiH2 as the initial material to prepare Fe-Ti alloy?"

python src/parse_papers.py

#Using SQlite database
#Checking the format of the table and the data of first 20 rows
python src/inspect_db.py ../data/materials.db
#Checking the format of the table and the data of first 50 rows
python src/inspect_db.py materials.db 50
#Export a table to a csv file
python src/inspect_db.py materials.db --export Materials
python src/inspect_db.py materials.db --export sqlite_sequence
python src/inspect_db.py materials.db --export Properties

#Extract triplets from database
python src/extract_triplets_db.py

#Extract triplets from parsed-paper json
python src/extract_triplets_json.py

#End to end RAG-anything
python src/ragall.py
