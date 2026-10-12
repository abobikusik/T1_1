import json
import re
import math
import pandas as pd
from pathlib import Path
from collections import defaultdict

# 1. КОНФИГУРАЦИЯ И ПУТИ К ФАЙЛАМ
BASE_DIR = Path(__file__).resolve().parent

# Путь к словарю навыков
SKILLS_DICT_PATH = BASE_DIR / "datasets_and_skills" / "skills" / "skills_dict_v2.json"

# Пути к НОВЫМ русским датасетам (укажите свои названия и расширения .csv или .json)
VACANCIES_PATH = BASE_DIR / "datasets_and_skills" / "datasets" / "vacancies_ru.csv"
RESUMES_PATH = BASE_DIR / "datasets_and_skills" / "datasets" / "resumes_ru.csv"

# 2. УНИВЕРСАЛЬНАЯ ЗАГРУЗКА И НОРМАЛИЗАЦИЯ ДАННЫХ
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
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}")


def normalize_skills_dict(skills_data):
    normalized = {}
    if isinstance(skills_data, dict):
        for k, v in skills_data.items():
            normalized[k] = v if isinstance(v, list) else [str(v)]
    elif isinstance(skills_data, list):
        for item in skills_data:
            if isinstance(item, dict):
                name = item.get("skill") or item.get("name") or item.get("title")
                aliases = item.get("aliases") or item.get("synonyms") or item.get("tags") or []
                if name:
                    normalized[name] = aliases if isinstance(aliases, list) else [str(aliases)]
                else:
                    for k, v in item.items():
                        normalized[k] = v if isinstance(v, list) else [str(v)]
            elif isinstance(item, str):
                normalized[item] = []
    return normalized


def extract_title(entity: dict, default_prefix: str = "Объект") -> str:
    """Извлекает название вакансии/резюме из ключей или парсит из текста."""
    # 1. Попытка взять из явных полей
    title = entity.get("title") or entity.get("position")
    if title and pd.notna(title) and str(title).strip():
        return str(title).strip()
        
    # 2. Если название зашито внутри поля text (например, 'Должность: ...')
    text = entity.get("text", "")
    if text and pd.notna(text):
        match = re.search(r'(?:Должность|Position|Роль|Role):\s*(.+)', str(text), re.IGNORECASE)
        if match:
            extracted = match.group(1).split('\n')[0].strip()
            if extracted:
                return extracted
                
    # 3. Резервный вариант, если название не найдено
    entity_id = entity.get("id")
    if entity_id is not None and pd.notna(entity_id):
        return f"{default_prefix} #{entity_id}"
    return f"{default_prefix} без названия"


def get_entity_text(entity: dict) -> str:
    """Извлечение и слияние текста из всех полей."""
    text_parts = []
    
    if "text" in entity and pd.notna(entity["text"]):
        text_parts.append(str(entity["text"]))
        
    fields = ["title", "position", "skills_raw", "skills", "description", "duties", "requirements"]
    for field in fields:
        val = entity.get(field)
        if val and pd.notna(val):
            if isinstance(val, list):
                text_parts.append(" ".join(map(str, val)))
            else:
                text_parts.append(str(val))
                
    return " ".join(text_parts).strip()


def load_data(sample_size: int = 1000):
    skills_raw = load_dataset(SKILLS_DICT_PATH)
    skills_dict = normalize_skills_dict(skills_raw)
    
    vacancies = load_dataset(VACANCIES_PATH)
    resumes = load_dataset(RESUMES_PATH)
    
    if sample_size and len(resumes) > sample_size:
        resumes = resumes[:sample_size]
        
    return skills_dict, vacancies, resumes


# 3. ИЗВЛЕЧЕНИЕ НАВЫКОВ И РАСЧЕТ TF-IDF
def extract_skills(text: str, skills_dict: dict) -> set:
    if not text:
        return set()
    
    text_clean = str(text).lower()
    found_skills = set()
    
    for canonical_skill, aliases in skills_dict.items():
        all_forms = [canonical_skill] + aliases
        for form in all_forms:
            pattern = r'(?:\b|_)' + re.escape(str(form).lower()) + r'(?:\b|_)'
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
        idf_weights[skill] = math.log((N + 1) / (df + 1)) + 1
        
    return idf_weights


# 4. ОСНОВНОЙ ПРОЦЕСС МЭТЧИНГА
def run_matching(sample_size: int = 1000):
    skills_dict, vacancies, resumes = load_data(sample_size=sample_size)
    
    print(f"Расчет TF-IDF весов по {len(resumes)} резюме...")
    idf_weights = compute_skills_idf(resumes, skills_dict)
    
    vacancy = vacancies[0]
    vac_text = get_entity_text(vacancy)
    vac_skills = extract_skills(vac_text, skills_dict)
    
    # Красивое извлечение названия вакансии
    vac_title = extract_title(vacancy, default_prefix="Вакансия")
    
    print(f"\nВакансия: {vac_title}")
    print(f"Извлеченные навыки вакансии ({len(vac_skills)}): {list(vac_skills)}\n")
    print("-" * 65)
    
    max_vac_weight = sum(idf_weights.get(skill, 1.0) for skill in vac_skills)
    results = []
    
    for idx, resume in enumerate(resumes):
        res_text = get_entity_text(resume)
        res_skills = extract_skills(res_text, skills_dict)
        
        matched_skills = vac_skills.intersection(res_skills)
        missing_skills = vac_skills - res_skills
        
        base_score = (len(matched_skills) / len(vac_skills) * 100) if vac_skills else 0.0
        matched_weight = sum(idf_weights.get(skill, 1.0) for skill in matched_skills)
        tfidf_score = (matched_weight / max_vac_weight * 100) if max_vac_weight > 0 else 0.0
        
        res_id = resume.get("id") if pd.notna(resume.get("id")) else idx + 1
        res_title = extract_title(resume, default_prefix="Резюме")
        
        results.append({
            "resume_id": res_id,
            "title": res_title,
            "base_score": round(base_score, 1),
            "tfidf_score": round(tfidf_score, 1),
            "matched_skills": list(matched_skills),
            "missing_skills": list(missing_skills),
            "total_candidate_skills": list(res_skills)
        })
        
    results.sort(
        key=lambda x: (x["tfidf_score"], len(x["matched_skills"]), len(x["total_candidate_skills"])), 
        reverse=True
    )
    
    print("ТОП-3 КАНДИДАТА ДЛЯ ВАКАНСИИ (С УЧЕТОМ TF-IDF ВЕСОВ):\n")
    for rank, res in enumerate(results[:3], start=1):
        print(f"Место #{rank} | Резюме ID: {res['resume_id']} | Должность: {res['title']}")
        print(f"TF-IDF Score: {res['tfidf_score']}% (Базовый: {res['base_score']}%)")
        print(f"Совпавшие навыки: {res['matched_skills']}")
        print(f"Недостающие навыки: {res['missing_skills']}")
        print(f"Все навыки кандидата: {res['total_candidate_skills']}")
        print("-" * 65)


if __name__ == "__main__":
    run_matching(sample_size=1000)