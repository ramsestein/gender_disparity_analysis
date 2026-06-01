import sys
sys.path.insert(0, "src_stat")
from scholar_enrich_public import build_public_list

df = build_public_list()
uniq = df.drop_duplicates("name").reset_index(drop=True)
print(f"Total entradas: {len(df)}  |  Nombres únicos: {len(uniq)}")
print()
print(uniq[["name", "source"]].to_string())
