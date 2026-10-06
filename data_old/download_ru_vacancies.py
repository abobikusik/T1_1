import os
import pandas as pd
import kagglehub

# 1. Скачиваем датасет с Kaggle (возвращает путь к локальной папке)
path = kagglehub.dataset_download("vyacheslavpanteleev1/hhru-it-vacancies-from-20211025-to-20211202")
print("Файлы скачаны в:", path)

# 2. Находим CSV-файл в скачанной папке
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
full_csv_path = os.path.join(path, csv_files[0])

# 3. Читаем CSV через Pandas
df = pd.read_csv(full_csv_path)

# 4. Оставляем только нужные колонки (название, навыки, описание)
df_clean = df.iloc[:, [1, 8, 9]].copy()
df_clean.columns = ["title", "skills_raw", "description"]

# 5. Сохраняем в готовую базу JSON
df_clean.to_json("dataset_ru_ready.json", orient="records", force_ascii=False, indent=2)

print("Готово! Первые 3 вакансии из датасета:")
print(df_clean.head(3))