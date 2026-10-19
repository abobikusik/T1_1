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

# Текст указанной вакансии SEL
VACANCY_TEXT = """
2,"Schweitzer Engineering Laboratories (SEL) — подразделение защиты инфраструктуры ищет талантливого специалиста на должность (ведущего) инженера-программиста с фокусом на кибербезопасность. Работа, которую мы выполняем, является одной из самых увлекательных, сложных и полезных, какую можно найти в сообществе инженеров энергетики.

Руководите командой, проводя white box и black box тестирование устройств и ПО на наличие уязвимостей и недостатков безопасности. Выявляйте проблемные зоны и давайте рекомендации по улучшению позиции в области кибербезопасности. Потребуется следить за актуальными уязвимостями безопасности и процессами их устранения, а также обучать этому других.

Примечание: требуется гражданство США. Как ведущий инженер-программист, типичный день может включать следующее: руководство разработкой спецификаций, проектированием, реализацией и тестированием систем firmware/ПО. Понимание взаимодействия аппаратного и программного обеспечения, сотрудничество с инженерами-электронщиками. Руководство разработкой проектной и тестовой документации. Наставничество инженеров, младших инженеров и техников.

Эта работа может подойти вам, если: степень бакалавра в области электротехники, computer science, кибербезопасности, информационных технологий, программной инженерии, информационных систем, компьютерной инженерии или эквивалент. Обычно 6+ лет опыта в одной или нескольких из следующих областей: объектно-ориентированное и/или встроенное программирование (C, C++, C#, Python), сетевые коммуникации (протоколы IS и/или ICS), цифровая криминалистика, reverse engineering. Опыт работы с промышленными системами управления, информационной безопасностью и киберанализом/операциями. См. NICCS Work Role ID: PR-VAM-001 и AN-EXP-001.

Информация о локации
Pullman, WA — корпоративный офис SEL расположен в Восточном Вашингтоне, где вы сможете насладиться непревзойдённым качеством жизни. Наслаждайтесь жизнью в небольшом городке: сельское пространство, отсутствие пробок, лёгкий доступ к рекреационным активностям в близлежащих горах, реках и лесах, а также отличные школы и университеты.

Конкурентная зарплата. Превосходные льготы. Вдохновляющая работа. Люди — основа нашей компании, и мы нанимаем сотрудников-собственников. Приглашаем вас узнать больше о том, как мы поддерживаем сотрудников. Мы на 100% находимся в собственности сотрудников. Пенсионные льготы включают финансируемый работодателем план владения акциями (ESOP) и персональные опции 401K. Мы предлагаем медицинскую страховку высшего уровня, страхование рецептурных препаратов, стоматологическую, зрительную, жизненную и страхование от нетрудоспособности. Мы осознаём важность здорового баланса между работой и личной жизнью и предлагаем: 10 оплачиваемых праздников, ежегодный отпуск, начинающийся с 12 дней, 9 оплачиваемых больничных дней, а также оплачиваемый отпуск по семейным обстоятельствам и по болезни, покрывающий 90% вашей зарплаты. Мы способствуем росту и развитию наших сотрудников через такие каналы, как STEM-курсы, ученичества, помощь в оплате обучения и программы инженерного развития. Спросите нашу команду о других льготах, включая wellness, фертильность, усыновление и гибкие расходные льготы.

Диапазон зарплаты: $91,000–$112,000 в год. Мы формируем наше начальное предложение по оплате на основе локации и связанных с работой факторов, таких как опыт кандидата, обучение, знания и навыки.

Коммуникация с кандидатами
Мы связываемся со всеми кандидатами. Если вы не получили ответ по поводу вашей заявки, пожалуйста, проверьте SPAM-фильтр или свяжитесь с нами по адресу careers@selinc.com.

SEL — работодатель, обеспечивающий равные возможности: ветераны/инвалиды.",natural,hf_resume-job-description-fit-ru"
"""


def prepare_skill_patterns(skills_dict):
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
    print("► Загрузка словаря навыков...")
    with open(DICT_PATH, encoding="utf-8") as f:
        skills_dict = json.load(f)

    compiled_skills = prepare_skill_patterns(skills_dict)

    # Извлекаем навыки вакансии SEL
    vacancy_skills = fast_extract_skills(VACANCY_TEXT, compiled_skills)

    print("\n" + "=" * 80)
    print("ВАКАНСИЯ: SEL — Инженер-программист (Кибербезопасность)")
    print(
        f"Извлеченные навыки вакансии ({len(vacancy_skills)}): {list(vacancy_skills)}"
    )
    print("=" * 80)

    # Быстрый анализ по 3 000 резюме
    print("\n► Анализируем выборку резюме и вычисляем IDF веса...")
    resumes_df = pd.read_csv(RESUMES_PATH, nrows=3000)

    resume_skills_list = []
    df_counts = {}
    N = len(resumes_df)

    for idx, row in resumes_df.iterrows():
        skills = fast_extract_skills(row["text"], compiled_skills)
        resume_skills_list.append(skills)
        for s in skills:
            df_counts[s] = df_counts.get(s, 0) + 1

    resumes_df["extracted_skills"] = resume_skills_list

    # IDF веса для каждого навыка
    idf_weights = {}
    for s in skills_dict:
        s_name = s["name"]
        df_val = df_counts.get(s_name, 0)
        idf_weights[s_name] = math.log((N + 1) / (df_val + 1)) + 1.0

    max_vac_weight = (
        sum(idf_weights.get(s, 1.0) for s in vacancy_skills)
        if vacancy_skills
        else 1.0
    )

    # Оценка резюме
    scored_resumes = []
    for idx, row in resumes_df.iterrows():
        r_skills = row["extracted_skills"]
        matched = vacancy_skills.intersection(r_skills)

        base_score = (
            (len(matched) / len(vacancy_skills)) * 100 if vacancy_skills else 0
        )
        matched_weight = sum(idf_weights.get(s, 1.0) for s in matched)
        tfidf_score = (
            (matched_weight / max_vac_weight) * 100
            if max_vac_weight > 0
            else 0
        )

        scored_resumes.append(
            {
                "id": row["id"],
                "text": str(row["text"]),
                "source": row.get("source", "N/A"),
                "extracted_skills": list(r_skills),
                "matched_skills": list(matched),
                "missing_skills": list(vacancy_skills - matched),
                "base_score": round(base_score, 1),
                "tfidf_score": round(tfidf_score, 1),
            }
        )

    # Сортировка по TF-IDF
    scored_resumes.sort(
        key=lambda x: (
            x["tfidf_score"],
            len(x["matched_skills"]),
            len(x["extracted_skills"]),
        ),
        reverse=True,
    )

    # Формируем выборку: 2 подходящих, 1 среднее, 2 неподходящих
    suitable_1 = scored_resumes[0]
    suitable_2 = (
        scored_resumes[1] if len(scored_resumes) > 1 else scored_resumes[0]
    )

    medium = next(
        (r for r in scored_resumes if 25.0 <= r["tfidf_score"] <= 55.0),
        scored_resumes[len(scored_resumes) // 2],
    )
    unsuitable_1 = next(
        (r for r in scored_resumes if 1.0 <= r["tfidf_score"] < 25.0),
        scored_resumes[-10],
    )
    unsuitable_2 = next(
        (r for r in reversed(scored_resumes) if r["tfidf_score"] == 0),
        scored_resumes[-1],
    )

    sample = [suitable_1, suitable_2, medium, unsuitable_1, unsuitable_2]
    sample.sort(
        key=lambda x: (
            x["tfidf_score"],
            len(x["matched_skills"]),
            len(x["extracted_skills"]),
        ),
        reverse=True,
    )

    print("\n" + "=" * 80)
    print("ОБРАБОТКА И РАНЖИРОВАНИЕ КАНДИДАТОВ ДЛЯ SEL:")
    print("=" * 80 + "\n")

    for rank, res in enumerate(sample, start=1):
        if res["tfidf_score"] >= 50.0:
            category_label = "ПОДХОДЯЩЕЕ РЕЗЮМЕ"
        elif 20.0 <= res["tfidf_score"] < 50.0:
            category_label = "СРЕДНЕЕ РЕЗЮМЕ"
        else:
            category_label = "НЕПОДХОДЯЩЕЕ РЕЗЮМЕ"

        print(
            f"МЕСТО #{rank} | [{category_label}] (ID: {res['id']}, Источник: {res['source']})"
        )
        print(
            f"  ► TF-IDF Score: {res['tfidf_score']}%  (Базовый процент совпадения: {res['base_score']}%)"
        )
        print(
            f"  ► Совпали навыки ({len(res['matched_skills'])}): {res['matched_skills']}"
        )
        print(f"  ► Не хватило навыков: {res['missing_skills']}")
        print(
            f"  ► Текст резюме: {res['text'][:140].replace('\n', ' ')}..."
        )
        print("-" * 80)


if __name__ == "__main__":
    main()