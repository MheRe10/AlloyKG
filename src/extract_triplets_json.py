import os
import json
import pandas as pd
from zhipuai import ZhipuAI
from io import StringIO


client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))


# ========= 用智谱大模型抽取三元组 =========
def extract_triples_from_text(text: str):
    if not text.strip():
        return [] #筛掉空输入
    prompt = f"""
            请从以下材料信息中提取制作知识图谱的三元组(entity1, relation, entity2)，
            请保证三元组格式正确，内容完整，同时输出时每行的格式为entity1, relation, entity2，不需要序号、括号等冗余字符。
            包括材料名称、类别、特性、应用等，如果输入是作者、发表时间等无关信息，则可以不输出。每行一个三元组：
            "{text}"
            """
    print(prompt)
    response = client.chat.completions.create(
        model="glm-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    content = response.choices[0].message.content
    triples = []
    for line in content.splitlines():
        print("===========================zhipuResponse===========================")
        print(line)
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            triples.append(tuple(parts))
    return triples


# ========= 表格处理 =========
def extract_triples_from_table(html, caption=None):
    from io import StringIO
    dfs = pd.read_html(StringIO(html))
    triples = []
    for df in dfs:
        table_text = df.to_markdown(index=False)
        if caption:
            table_text = f"{caption}\n{table_text}"
        triples.extend(extract_triples_from_text(table_text))
    return triples


# ========= 主处理函数 =========
def process_json(json_data):
    triples = []
    for item in json_data:
        if item["type"] == "text":
            triples.extend(extract_triples_from_text(item["text"]))

        elif item["type"] == "image":
            for cap in item.get("image_caption", []):
                triples.extend(extract_triples_from_text(cap))

        elif item["type"] == "table":
            triples.extend(
                extract_triples_from_table(
                    item["table_body"],
                    caption=" ".join(item.get("table_caption", []))
                )
            )
    return triples


# ========= 示例 & 输出 CSV =========
if __name__ == "__main__":
    json_dir = "../data/paper_json"
    output_dir = "../data/json_triplets"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(json_dir, filename)
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            triples = process_json(json_data)
            df = pd.DataFrame(triples, columns=["Entity1",  "Relation", "Entity2"])
            output_path = os.path.join(output_dir, filename.replace(".json", "_triples.csv"))
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"已处理并保存: {output_path}")

