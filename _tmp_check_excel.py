import pandas as pd
xl = pd.read_excel('esicm_talks_grouped_9_groups_new.xlsx', sheet_name='Grouped participants')
names = ['Elena Sancho Ferrando','Eleonora Balzani','Ana-Maria Ioan','Luigi Zattera']
for n in names:
    r = xl[xl['Name of the person'] == n]
    if r.empty:
        print(f'NOT FOUND: {n}')
        first = n.split()[0]
        hits = xl[xl['Name of the person'].str.contains(first, case=False, na=False)]
        for _, row in hits.iterrows():
            print(f'  Similar: {row["Name of the person"]} | oldMap={row["oldNamesMapping"]}')
    else:
        for _, row in r.iterrows():
            print(f'FOUND: {n} | oldMap={row["oldNamesMapping"]} | Role={row["Role"]}')
