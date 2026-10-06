import os
import json
import pandas as pd
import kagglehub

# 1. Скачиваем датасет
path = kagglehub.dataset_download("saugataroyarghya/resume-dataset")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
full_csv_path = os.path.join(path, csv_files[0])

# 2. Читаем CSV
df = pd.read_csv(full_csv_path)

df_clean = pd.DataFrame()

# 3. Делаем короткий title (первые 40 символов из цели)
df_clean['title'] = df['career_objective'].fillna('IT Specialist').astype(str).str.slice(0, 40) + "..."

# 4. Сохраняем навыки
df_clean['skills_raw'] = df['skills'].fillna('')

# 5. Собираем описание
desc_cols = ['career_objective', 'major_field_of_studies', 'degree_names', 'professional_company_names']
existing_cols = [col for col in desc_cols if col in df.columns]
df_clean['description'] = df[existing_cols].fillna('').astype(str).agg('. '.join, axis=1)

# 6. Сохраняем в JSON
df_clean.to_json("resumes_en_ready.json", orient="records", force_ascii=False, indent=2)

# 7. Читаем и выводим ПЕРВУЮ запись из итогового JSON для проверки
with open("resumes_en_ready.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Первая запись из итогового файла resumes_en_ready.json:\n")
print(json.dumps(data[0], ensure_ascii=False, indent=2))