import sqlite3
import sys
import os
import csv

def inspect_db(filepath, dump_rows=20, export_table=None):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    # 检查文件开头是否包含 "SQLite format 3"
    with open(filepath, "rb") as f:
        header = f.read(16)
    if b"SQLite format 3" not in header:
        print("This file does not look like a SQLite database.")
        return

    try:
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()

        # 获取所有表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]

        if not tables:
            print("No tables found in this database.")
        else:
            print("Tables in database:", ", ".join(tables))

            for table_name in tables:
                print(f"\n=== {table_name} ===")
                
                # 获取表的列结构
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                col_names = [col[1] for col in columns]
                print("Columns:", ", ".join([f"{col[1]} ({col[2]})" for col in columns]))

                # 导出前 N 行
                cursor.execute(f"SELECT * FROM {table_name} LIMIT {dump_rows};")
                rows = cursor.fetchall()
                if rows:
                    print(f"First {len(rows)} rows:")
                    for row in rows:
                        print(dict(zip(col_names, row)))
                else:
                    print("No data in this table.")

                # 导出整张表为 CSV
                if export_table and table_name == export_table:
                    cursor.execute(f"SELECT * FROM {table_name};")
                    all_rows = cursor.fetchall()
                    csv_filename = f"{table_name}.csv"
                    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(col_names)  # 写入列名
                        writer.writerows(all_rows)  # 写入所有数据
                    print(f"Table {table_name} exported to {csv_filename}")

        conn.close()

    except sqlite3.DatabaseError as e:
        print(f"Error reading database: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_db.py <yourfile.db> [rows_to_dump] [--export table_name]")
    else:
        filepath = sys.argv[1]
        dump_rows = 20
        export_table = None

        # 解析参数
        if len(sys.argv) >= 3 and sys.argv[2].isdigit():
            dump_rows = int(sys.argv[2])

        if "--export" in sys.argv:
            idx = sys.argv.index("--export")
            if idx + 1 < len(sys.argv):
                export_table = sys.argv[idx + 1]

        inspect_db(filepath, dump_rows, export_table)
