"""
dedup_csv.py — 去除所有 CSV 文件中的重复日期行
保留每个日期第一次出现的行（文件是倒序的，第一次 = 最新/最完整的数据）
"""
import pandas as pd
import os
from tqdm import tqdm

DATA_DIR = r'all_stock_data'

all_files = []
for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith('.csv') and f[:6].isdigit():
            all_files.append(os.path.join(root, f))

fixed = 0
clean = 0

for fp in tqdm(all_files, desc="去重进度"):
    try:
        df = pd.read_csv(fp, encoding='utf-8-sig', dtype=str)
        col = '交易日期'
        if col not in df.columns:
            clean += 1
            continue
        before = len(df)
        df = df.drop_duplicates(subset=[col], keep='first')
        after = len(df)
        if after < before:
            df.to_csv(fp, index=False, encoding='utf-8-sig')
            fixed += 1
        else:
            clean += 1
    except Exception as e:
        pass

print(f'\n去重完成: 修复 {fixed} 个文件 | 无重复 {clean} 个文件')
