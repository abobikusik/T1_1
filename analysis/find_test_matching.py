import json
import math
import re
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

DICT_PATH = BASE_DIR / "datasets_and_skills" / "skills" / "skills_dict_v3.json"
RESUMES_PATH = BASE_DIR / "datasets_and_skills" / "datasets" / "resumes_ru.csv"
VACANCIES_PATH = BASE_DIR / "datasets_and_skills" / "datasets" / "vacancies_ru.csv"


def prepare_skill_patterns(skills_dict):
    """Компилирует алиасы навыков в быстрые регулярные выражения."""
    compiled_skills = []
    for s in skills_dict:
        name = s["name"]
        aliases = s["aliases"]
        escaped_aliases = [re.escape(a.lower()) for a in aliases]
        pattern_str = (
            r"(?<=[\s,.;:()!?-])(?:"
            + "|".join(escaped_aliases)
            + r")(?=[\s,.;:()!?-])"
        )
        compiled_skills.append((name, re.compile(pattern_str)))
    return compiled_skills


def fast_extract_skills(text, compiled_skills):
    if not isinstance(text, str):
        return set()
    t = " " + text.lower() + " "
    found = set()
    for name, pattern in compiled_skills:
        if pattern.search(t):
            found.add(name)
    return found


def main():
    print("► Загрузка словаря навыков и датасетов...")
    with open(DICT_PATH, encoding="utf-8") as f:
        skills_dict = json.load(f)

    compiled_skills = prepare_skill_patterns(skills_dict)

    # Читаем 1000 вакансий и 3000 резюме
    print("► Загружаем 1000 вакансий и 3000 резюме...")
    vacancies_df = pd.read_csv(VACANCIES_PATH, nrows=1000)
    resumes_df = pd.read_csv(RESUMES_PATH, nrows=3000)

    # 1. Извлекаем навыки из всех 3000 резюме и вычисляем IDF веса
    print("► Извлечение навыков из выборки резюме и расчёт IDF весов...")
    resume_skills_list = []
    df_counts = {}
    N = len(resumes_df)

    for idx, row in resumes_df.iterrows():
        skills = fast_extract_skills(row["text"], compiled_skills)
        resume_skills_list.append(skills)
        for s in skills:
            df_counts[s] = df_counts.get(s, 0) + 1

    resumes_df["extracted_skills"] = resume_skills_list

    idf_weights = {}
    for s in skills_dict:
        s_name = s["name"]
        df_val = df_counts.get(s_name, 0)
        idf_weights[s_name] = math.log((N + 1) / (df_val + 1)) + 1.0

    # 2. Извлекаем навыки из всех 1000 вакансий
    print("► Извлечение навыков из 1000 вакансий...")
    vacancies_skills_list = []
    for idx, row in vacancies_df.iterrows():
        vac_skills = fast_extract_skills(row["text"], compiled_skills)
        vacancies_skills_list.append(vac_skills)
    vacancies_df["extracted_skills"] = vacancies_skills_list

    # 3. Поиск вакансии, под которую идеально находятся ровно нужные 5 резюме
    print("► Поиск идеальной вакансии с нужным набором кандидатов...")

    best_vacancy = None
    best_sample = None

    for idx, vac_row in vacancies_df.iterrows():
        v_skills = vac_row["extracted_skills"]
        if len(v_skills) < 3:
            continue

        max_vac_weight = sum(idf_weights.get(s, 1.0) for s in v_skills)

        # Оцениваем все 3000 резюме под текущую вакансию
        scored = []
        for r_idx, r_row in resumes_df.iterrows():
            r_skills = r_row["extracted_skills"]
            matched = v_skills.intersection(r_skills)

            base_score = (len(matched) / len(v_skills)) * 100
            matched_weight = sum(idf_weights.get(s, 1.0) for s in matched)
            tfidf_score = (
                (matched_weight / max_vac_weight) * 100
                if max_vac_weight > 0
                else 0
            )

            scored.append(
                {
                    "id": r_row["id"],
                    "text": str(r_row["text"]),
                    "source": r_row.get("source", "N/A"),
                    "extracted_skills": list(r_skills),
                    "matched_skills": list(matched),
                    "missing_skills": list(v_skills - matched),
                    "base_score": round(base_score, 1),
                    "tfidf_score": round(tfidf_score, 1),
                }
            )

        # Категоризируем кандидатов под эту вакансию
        hundreds = [r for r in scored if r["base_score"] == 100.0]
        fifties = [r for r in scored if 40.0 <= r["tfidf_score"] <= 60.0]
        lows = [r for r in scored if 1.0 <= r["tfidf_score"] <= 25.0]
        zeros = [r for r in scored if r["tfidf_score"] == 0.0]

        # Идеальное условие: есть минимум 2 со 100%, минимум 1 с ~50%, и есть с 0%
        if len(hundreds) >= 2 and len(fifties) >= 1 and len(zeros) >= 1:
            best_vacancy = vac_row
            best_sample = [
                hundreds[0],
                hundreds[1],
                fifties[0],
                lows[0] if lows else zeros[1],
                zeros[0],
            ]
            break

    # 4. Вывод результатов
    if best_vacancy is None:
        print("Не удалось найти вакансию с ровно двумя 100% совпадениями.")
        return

    print("\n" + "=" * 80)
    print("НАЙДЕННАЯ ВАКАНСИЯ ИЗ 1000 ПРОСМОТРЕННЫХ:")
    print(f"ID Вакансии: {best_vacancy['id']}")
    print(
        f"Требования вакансии ({len(best_vacancy['extracted_skills'])}): {list(best_vacancy['extracted_skills'])}"
    )
    print(
        "Текст вакансии (фрагмент):",
        str(best_vacancy["text"])[:200].replace("\n", " "),
        "...",
    )
    print("=" * 80)

    # Сортируем отбранную 5-ку кандидатов алгоритмом
    best_sample.sort(
        key=lambda x: (
            x["tfidf_score"],
            len(x["matched_skills"]),
            len(x["extracted_skills"]),
        ),
        reverse=True,
    )

    print("\n" + "=" * 80)
    print("ИТОГОВОЕ РАНЖИРОВАНИЕ КОНТРОЛЬНОЙ ПЯТЕРКИ КАНДИДАТОВ:")
    print("=" * 80 + "\n")

    for rank, res in enumerate(best_sample, start=1):
        if res["base_score"] == 100.0 or res["tfidf_score"] == 100.0:
            category_label = "100% ПОДХОДЯЩЕЕ РЕЗЮМЕ"
        elif 35.0 <= res["tfidf_score"] < 80.0:
            category_label = "СРЕДНЕЕ РЕЗЮМЕ (~50%)"
        else:
            category_label = "НЕПОДХОДЯЩЕЕ РЕЗЮМЕ"

        print(
            f"МЕСТО #{rank} | [{category_label}] (ID: {res['id']}, Источник: {res['source']})"
        )
        print(
            f"  ► TF-IDF Score: {res['tfidf_score']}%  (Базовое покрытие: {res['base_score']}%)"
        )
        print(
            f"  ► Совпали навыки ({len(res['matched_skills'])}): {res['matched_skills']}"
        )
        print(f"  ► Не хватило навыков: {res['missing_skills']}")
        print(
            f"  ► Фрагмент резюме: {res['text'][:140].replace('\n', ' ')}..."
        )
        print("-" * 80)


if __name__ == "__main__":
    main()