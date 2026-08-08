"""
初始化示例数据库 —— 创建销售记录表并插入测试数据
运行一次即可：python init_db.py
"""
import sqlite3
import os

DB_PATH = "company.db"

# 删掉旧数据库（如果有的话）
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    amount REAL NOT NULL,
    quantity INTEGER NOT NULL,
    sale_date TEXT NOT NULL,
    region TEXT NOT NULL,
    salesperson TEXT NOT NULL
)
""")

sample_data = [
    ("iPhone 16", "手机", 7999, 120, "2026-07-01", "华东", "张三"),
    ("MacBook Pro", "电脑", 14999, 45, "2026-07-02", "华东", "李四"),
    ("iPad Air", "平板", 4999, 80, "2026-07-03", "华南", "王五"),
    ("AirPods Pro", "耳机", 1899, 200, "2026-07-05", "华北", "张三"),
    ("Apple Watch", "穿戴", 3199, 65, "2026-07-06", "华东", "赵六"),
    ("iPhone 16", "手机", 7999, 150, "2026-07-08", "华南", "李四"),
    ("MacBook Pro", "电脑", 14999, 38, "2026-07-10", "华北", "王五"),
    ("iPad Air", "平板", 4999, 55, "2026-07-12", "华东", "张三"),
    ("AirPods Pro", "耳机", 1899, 175, "2026-07-14", "华南", "赵六"),
    ("Apple Watch", "穿戴", 3199, 50, "2026-07-15", "华北", "李四"),
    ("iPhone 16", "手机", 7999, 100, "2026-07-16", "华东", "王五"),
    ("MacBook Pro", "电脑", 14999, 42, "2026-07-18", "华南", "张三"),
    ("iPad Air", "平板", 4999, 70, "2026-07-20", "华北", "赵六"),
    ("AirPods Pro", "耳机", 1899, 190, "2026-07-22", "华东", "李四"),
    ("Apple Watch", "穿戴", 3199, 55, "2026-07-24", "华南", "王五"),
    ("iPhone 16", "手机", 7999, 130, "2026-07-25", "华北", "张三"),
    ("MacBook Pro", "电脑", 14999, 35, "2026-07-27", "华东", "赵六"),
    ("iPad Air", "平板", 4999, 60, "2026-07-28", "华南", "李四"),
    ("AirPods Pro", "耳机", 1899, 210, "2026-07-29", "华北", "王五"),
    ("Apple Watch", "穿戴", 3199, 45, "2026-07-30", "华东", "张三"),
]

cursor.executemany(
    "INSERT INTO sales (product_name, category, amount, quantity, sale_date, region, salesperson) VALUES (?, ?, ?, ?, ?, ?, ?)",
    sample_data,
)

conn.commit()
conn.close()
print("✅ 数据库 company.db 创建成功！已插入 20 条销售记录。")
