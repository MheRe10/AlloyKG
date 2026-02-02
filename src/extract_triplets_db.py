import asyncio
import os
import csv
import pandas as pd
from zhipuai import ZhipuAI

# -------------------------
# set up API and paths
# -------------------------
API_KEY = os.getenv("ZHIPU_API_KEY")
RAW_CSV_FOLDER = "../data/db_csv" 
OUTPUT_DIR = "../data/db_triplets"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Initialize ZhipuAI client
# -------------------------
client = ZhipuAI(api_key=API_KEY)

# -------------------------
# Zhipu GLM-4 async call
# -------------------------
async def glm4_complete(prompt: str, system_prompt=None, history_messages=[], **kwargs):
    loop = asyncio.get_event_loop()

    def sync_call():
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="glm-4",
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
        )
        return response.choices[0].message.content

    return await loop.run_in_executor(None, sync_call)

# -------------------------
# CSV -> Text
# -------------------------
def materials_csv_to_text(csv_file, N):
    df = pd.read_csv(csv_file, nrows=N)
    texts = []
    for _, row in df.iterrows():
        text = (
            f"Name: {row.get('name','')}; "
            f"Categories: {row.get('categories','')}; "
            f"Notes: {row.get('notes','')}; "
            f"Keywords: {row.get('keywords','')}"
        )
        texts.append(text)
    return "\n".join(texts)

def properties_csv_to_text(csv_file, N):
    df = pd.read_csv(csv_file, nrows = N)
    texts = []
    for _, row in df.iterrows():
        text = (
            f"Material ID: {row.get('material_id','')}; "
            f"Property Type: {row.get('property_type','')}; "
            f"Property Name: {row.get('property_name','')}; "
            f"Metric Value: {row.get('metric_value','')}; "
            f"English Value: {row.get('english_value','')}; "
            f"Comments: {row.get('comments','')}"
        )
        texts.append(text)
    return "\n".join(texts)

# -------------------------
# Extract triplets from CSV
# -------------------------
async def extract_triplets_from_csv(csv_file, N):

    df = pd.read_csv(csv_file, nrows=0)
    columns = set(df.columns)
    # Determine which conversion function to use based on CSV schema.
    if {"name", "categories", "notes", "keywords"}.issubset(columns):
        csv_text = materials_csv_to_text(csv_file, N)
    elif {"material_id", "property_type", "property_name", "metric_value", "english_value", "comments"}.issubset(columns):
        csv_text = properties_csv_to_text(csv_file, N)
    else:
        raise ValueError("未知的CSV格式")  # EN: Unknown/unrecognized CSV schema.

    # EN: The prompt is written in Chinese to instruct GLM-4 to extract KG triplets in a strict CSV-friendly format.
    prompt = f"""
            请从以下材料信息中提取制作知识图谱的三元组(entity1,  relation, entity2)，
            请保证三元组格式正确，内容完整，同时输出时每行的格式为entity1, relation, entity2，不需要序号、括号等冗余字符。
            包括材料名称、类别、特性、应用等。每行一个三元组：
            {csv_text}
            """
    triplets_text = await glm4_complete(prompt)
    
    # Simply parse the output into a list of triplets
    triplets = []
    for line in triplets_text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.strip("()")
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            triplets.append(tuple(parts))
    return triplets

# -------------------------
# Save triplets to CSV
# -------------------------
def save_triplets_csv(triplets, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Entity1", "Relation", "Entity2"])
        writer.writerows(triplets)
    print(f"Triplets saved to {output_file}")

# -------------------------
# Batch process CSV folder
# -------------------------
async def process_csv_folder(csv_folder):
    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            csv_path = os.path.join(csv_folder, filename)
            print(f"Processing {csv_path} ...")
            triplets = await extract_triplets_from_csv(csv_path, N=30) # 限制取前N行，设置为None来取全部行
            output_file = os.path.join(
                OUTPUT_DIR, os.path.splitext(filename)[0] + "_triplets.csv"
            )
            save_triplets_csv(triplets, output_file)

# -------------------------
# Main function
# -------------------------
if __name__ == "__main__":
    asyncio.run(process_csv_folder(RAW_CSV_FOLDER))
