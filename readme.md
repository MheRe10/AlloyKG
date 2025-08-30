#激活虚拟环境
venv_alloykg\Scripts\activate

$env:ZHIPU_API_KEY="a3d236a6017d4cfc9f15c509a3e7c786.eZIIBa9QmssnGEsv"


python parse_papers.py

#Using SQlite database
#Checking the format of the table and the data of first 20 rows
python inspect_db.py materials.db
#Checking the format of the table and the data of first 50 rows
python inspect_db.py materials.db 50
#Export a table to a csv file
python inspect_db.py materials.db --export Materials
python inspect_db.py materials.db --export sqlite_sequence
python inspect_db.py materials.db --export Properties

#Extract triplets from database
python extract_triplets_db.py