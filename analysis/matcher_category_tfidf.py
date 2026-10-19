#Добавлены 3 аналитических блока:
# Глобальный анализ весов по категориям: Средний, минимальный и максимальный вес IDF навыков в каждой категории.
# Анализ весов вакансии по категориям: Какие категории навыков требуются в вакансии и какова их весовая ценность.
# Анализ весов кандидата по категориям: Какими категориями навыков обладает каждый кандидат и какова суммарная «ценность» его навыков в каждой категории.

from collections import defaultdict
import json
import math
from pathlib import Path
import re
import pandas as pd

# ----------------------------------------------------------------------
# 1. КОНФИГУРАЦИЯ И ПУТИ К ФАЙЛАМ
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

SKILLS_DICT_PATH = BASE_DIR / "skills_dict_v3.json"
VACANCIES_PATH = BASE_DIR / "text_vacancy.txt"
RESUMES_PATH = BASE_DIR / "text_resume.txt"


# ----------------------------------------------------------------------
# 2. ЗАГРУЗКА И НОРМАЛИЗАЦИЯ С УЧЕТОМ КАТЕГОРИЙ
# ----------------------------------------------------------------------
def load_dataset(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден по пути: {file_path}")

    ext = file_path.suffix.lower()

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]

    elif ext == ".csv":
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()

        raw_blocks = re.split(r"(?:\r?\n)+(?=\d+\s*,\s*)", raw_text)

        items = []
        for idx, block in enumerate(raw_blocks, start=1):
            block_clean = block.strip()
            if not block_clean:
                continue

            match = re.match(r"^(\d+)\s*,\s*\"?(.*?)\"?$", block_clean, re.DOTALL)
            if match:
                item_id = match.group(1)
                item_text = match.group(2).strip()
            else:
                item_id = str(idx)
                item_text = block_clean

            item_text = re.sub(
                r'\",\s*natural\s*,\s*[\w-]+$', "", item_text
            ).strip()

            items.append({"id": item_id, "text": item_text})

        return items
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}")


def normalize_skills_dict(skills_data):
    """
    Нормализует словарь навыков.
    Возвращает:
      - skills_dict: { canonical_name: [aliases] }
      - skill_categories: { canonical_name: category_string }
    """
    skills_dict = {}
    skill_categories = {}

    if isinstance(skills_data, list):
        for item in skills_data:
            if isinstance(item, dict):
                name = item.get("name") or item.get("skill") or item.get("title")
                aliases = item.get("aliases") or item.get("synonyms") or []
                category = item.get("category", "Без категории")

                if name:
                    skills_dict[name] = aliases if isinstance(aliases, list) else [str(aliases)]
                    skill_categories[name] = category
    elif isinstance(skills_data, dict):
        for k, v in skills_data.items():
            skills_dict[k] = v if isinstance(v, list) else [str(v)]
            skill_categories[k] = "Без категории"

    return skills_dict, skill_categories


def extract_title(entity: dict, default_prefix: str = "Объект") -> str:
    title = entity.get("title") or entity.get("position")
    if title and pd.notna(title) and str(title).strip():
        return str(title).strip()

    text = entity.get("text", "")
    if text and pd.notna(text):
        match = re.search(
            r"(?:Должность|Position|Роль|Role):\s*(.+)",
            str(text),
            re.IGNORECASE,
        )
        if match:
            extracted = match.group(1).split("\n")[0].strip()
            if extracted:
                return extracted

        first_line = str(text).strip().split("\n")[0][:50]
        if first_line:
            return first_line + "..."

    entity_id = entity.get("id")
    if entity_id is not None and pd.notna(entity_id):
        return f"{default_prefix} #{entity_id}"
    return f"{default_prefix} без названия"


def get_entity_text(entity: dict) -> str:
    text_parts = []
    if "text" in entity and pd.notna(entity["text"]):
        text_parts.append(str(entity["text"]))

    fields = [
        "title",
        "position",
        "skills_raw",
        "skills",
        "description",
        "duties",
        "requirements",
    ]
    for field in fields:
        val = entity.get(field)
        if val and pd.notna(val):
            if isinstance(val, list):
                text_parts.append(" ".join(map(str, val)))
            else:
                text_parts.append(str(val))

    return " ".join(text_parts).strip()


def load_data():
    skills_raw = load_dataset(SKILLS_DICT_PATH)
    skills_dict, skill_categories = normalize_skills_dict(skills_raw)

    vacancies = load_dataset(VACANCIES_PATH)
    resumes = load_dataset(RESUMES_PATH)

    return skills_dict, skill_categories, vacancies, resumes


# ----------------------------------------------------------------------
# 3. ИЗВЛЕЧЕНИЕ НАВЫКОВ И РАСЧЕТ TF-IDF
# ----------------------------------------------------------------------
def extract_skills(text: str, skills_dict: dict) -> set:
    if not text:
        return set()

    text_clean = str(text).lower()
    found_skills = set()

    for canonical_skill, aliases in skills_dict.items():
        all_forms = [canonical_skill] + aliases
        for form in all_forms:
            pattern = (
                r"(?:^|[\s,.;:()!?-])"
                + re.escape(str(form).lower())
                + r"(?:$|[\s,.;:()!?-])"
            )
            if re.search(pattern, text_clean):
                found_skills.add(canonical_skill)
                break

    return found_skills


def compute_skills_idf(resumes: list, skills_dict: dict) -> dict:
    N = len(resumes)
    doc_freq = defaultdict(int)

    for resume in resumes:
        res_text = get_entity_text(resume)
        skills_in_doc = extract_skills(res_text, skills_dict)
        for skill in skills_in_doc:
            doc_freq[skill] += 1

    idf_weights = {}
    for skill in skills_dict.keys():
        df = doc_freq[skill]
        idf_weights[skill] = math.log((N + 1) / (df + 1)) + 1.0

    return idf_weights


# ----------------------------------------------------------------------
# 4. ФУНКЦИИ ВЫВОДА ВЕСОВ ПО КАТЕГОРИЯМ
# ----------------------------------------------------------------------
def print_category_idf_summary(idf_weights: dict, skill_categories: dict):
    """Выводит средние и максимальные веса TF-IDF по каждой категории."""
    cat_weights = defaultdict(list)
    for skill, weight in idf_weights.items():
        cat = skill_categories.get(skill, "Без категории")
        cat_weights[cat].append((skill, weight))

    stats = []
    for cat, items in cat_weights.items():
        weights = [w for _, w in items]
        avg_w = sum(weights) / len(weights)
        max_w = max(weights)
        min_w = min(weights)
        stats.append({
            "Категория": cat,
            "Кол-во навыков": len(items),
            "Средний IDF": round(avg_w, 3),
            "Мин IDF": round(min_w, 3),
            "Макс IDF": round(max_w, 3)
        })

    df_stats = pd.DataFrame(stats).sort_values(by="Средний IDF", ascending=False)
    
    print("\n" + "=" * 75)
    print(" 📊 ВЫВОД ВЕСОВ TF-IDF ПО КАТЕГОРИЯМ (ГЛОБАЛЬНО ПО ВСЕМУ СЛОВАРЮ)")
    print("=" * 75)
    print(df_stats.to_string(index=False))
    print("=" * 75 + "\n")


def print_entity_category_breakdown(entity_name: str, skills: set, idf_weights: dict, skill_categories: dict):
    """Выводит раскладку весов навыков конкретной вакансии или резюме по категориям."""
    cat_map = defaultdict(list)
    total_weight = 0.0

    for skill in skills:
        w = idf_weights.get(skill, 1.0)
        cat = skill_categories.get(skill, "Без категории")
        cat_map[cat].append((skill, round(w, 3)))
        total_weight += w

    print(f"АНАЛИЗ ВЕСОВ ПО КАТЕГОРИЯМ ДЛЯ: {entity_name}")
    print(f"Суммарный вес всех извлеченных навыков: {round(total_weight, 3)}")
    
    rows = []
    for cat, skill_list in cat_map.items():
        cat_weight = sum(w for _, w in skill_list)
        pct = (cat_weight / total_weight * 100) if total_weight > 0 else 0.0
        skills_formatted = ", ".join([f"{s} ({w})" for s, w in skill_list])
        rows.append({
            "Категория": cat,
            "Кол-во": len(skill_list),
            "Сумма весов": round(cat_weight, 3),
            "Доля %": f"{round(pct, 1)}%",
            "Навыки (с весом IDF)": skills_formatted
        })

    df_entity = pd.DataFrame(rows).sort_values(by="Сумма весов", ascending=False)
    print(df_entity.to_string(index=False))
    print("-" * 75 + "\n")


# ----------------------------------------------------------------------
# 5. ОСНОВНОЙ ПРОЦЕСС МЭТЧИНГА
# ----------------------------------------------------------------------
def run_matching():
    skills_dict, skill_categories, vacancies, resumes = load_data()

    print(
        f"Успешно загружено из локальной папки: {len(vacancies)} вакансия, {len(resumes)} резюме.\n"
    )

    print("Расчет TF-IDF весов по корпусу резюме...")
    idf_weights = compute_skills_idf(resumes, skills_dict)

    # 1. Выводим сводную таблицу весов по категориям
    print_category_idf_summary(idf_weights, skill_categories)

    vacancy = vacancies[0]
    vac_text = get_entity_text(vacancy)
    vac_skills = extract_skills(vac_text, skills_dict)
    vac_title = extract_title(vacancy, default_prefix="Вакансия")

    print("=" * 75)
    print(f"ЦЕЛЕВАЯ ВАКАНСИЯ: {vac_title}")
    print("=" * 75)
    # 2. Выводим веса вакансии по категориям
    print_entity_category_breakdown(f"Вакансия [{vac_title}]", vac_skills, idf_weights, skill_categories)

    max_vac_weight = sum(idf_weights.get(skill, 1.0) for skill in vac_skills)
    results = []

    for idx, resume in enumerate(resumes):
        res_text = get_entity_text(resume)
        res_skills = extract_skills(res_text, skills_dict)

        matched_skills = vac_skills.intersection(res_skills)
        missing_skills = vac_skills - res_skills

        base_score = (
            (len(matched_skills) / len(vac_skills) * 100) if vac_skills else 0.0
        )
        matched_weight = sum(
            idf_weights.get(skill, 1.0) for skill in matched_skills
        )
        tfidf_score = (
            (matched_weight / max_vac_weight * 100) if max_vac_weight > 0 else 0.0
        )

        res_id = resume.get("id") if pd.notna(resume.get("id")) else idx + 1
        res_title = extract_title(resume, default_prefix="Резюме")

        results.append(
            {
                "resume_id": res_id,
                "title": res_title,
                "base_score": round(base_score, 1),
                "tfidf_score": round(tfidf_score, 1),
                "matched_skills": sorted(list(matched_skills)),
                "missing_skills": sorted(list(missing_skills)),
                "total_candidate_skills": res_skills,
            }
        )

    results.sort(
        key=lambda x: (
            x["tfidf_score"],
            len(x["matched_skills"]),
            len(x["total_candidate_skills"]),
        ),
        reverse=True,
    )

    print("=" * 75)
    print("РЕЗУЛЬТАТЫ РАНЖИРОВАНИЯ И АНАЛИЗ КАНДИДАТОВ ПО КАТЕГОРИЯМ")
    print("=" * 75 + "\n")

    for rank, res in enumerate(results, start=1):
        print(
            f"МЕСТО #{rank} | Резюме ID: {res['resume_id']} | Должность: {res['title']}"
        )
        print(
            f"  ► TF-IDF Score: {res['tfidf_score']}%  (Базовый процент: {res['base_score']}%)"
        )
        print(
            f"  ► Совпали навыки ({len(res['matched_skills'])}): {res['matched_skills']}"
        )
        print(f"  ► Недостающие навыки: {res['missing_skills']}\n")

        # 3. Выводим веса каждого резюме по категориям
        print_entity_category_breakdown(
            f"Кандидат ID #{res['resume_id']} ({res['title']})",
            res['total_candidate_skills'],
            idf_weights,
            skill_categories
        )


if __name__ == "__main__":
    run_matching()