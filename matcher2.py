import json
import re
import math
from pathlib import Path
from collections import defaultdict

# Определение базовой директории (папка, где лежит этот скрипт)
BASE_DIR = Path(__file__).resolve().parent

# Определение путей к файлам данных
SKILLS_DICT_PATH = BASE_DIR / "dict" / "skills_dict_v2.json"
VACANCIES_PATH = BASE_DIR / "data" / "vacancies_ru_ready.json"
RESUMES_PATH = BASE_DIR / "data" / "resumes_en_ready.json"


# 1. Приведение словаря к единому формату
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


# 2. Безопасная загрузка данных с использованием pathlib и проверок
def load_json_file(file_path: Path):
    """Вспомогательная функция для безопасного чтения JSON-файла."""
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден по пути: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data():
    skills_raw = load_json_file(SKILLS_DICT_PATH)
    vacancies = load_json_file(VACANCIES_PATH)
    resumes = load_json_file(RESUMES_PATH)
        
    skills_dict = normalize_skills_dict(skills_raw)
    return skills_dict, vacancies, resumes


# 3. Извлечение навыков из текста
def extract_skills(text, skills_dict):
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


# 4. Расчет IDF для каждого навыка из словаря по корпусу резюме
def compute_skills_idf(resumes, skills_dict):
    N = len(resumes)
    doc_freq = defaultdict(int)
    
    # Считаем, в скольких резюме встречается каждый навык
    for resume in resumes:
        res_text = f"{resume.get('title', '')} {resume.get('skills_raw', '')} {resume.get('description', '')}"
        skills_in_doc = extract_skills(res_text, skills_dict)
        for skill in skills_in_doc:
            doc_freq[skill] += 1
            
    # Вычисляем IDF: log((N + 1) / (df + 1)) + 1
    idf_weights = {}
    for skill in skills_dict.keys():
        df = doc_freq[skill]
        idf_weights[skill] = math.log((N + 1) / (df + 1)) + 1
        
    return idf_weights


# 5. Основной процесс мэтчинга с TF-IDF слоем
def run_matching():
    skills_dict, vacancies, resumes = load_data()
    
    print("Расчет TF-IDF весов для навыков...")
    idf_weights = compute_skills_idf(resumes, skills_dict)
    
    vacancy = vacancies[0]
    vac_text = f"{vacancy.get('title', '')} {vacancy.get('skills_raw', '')} {vacancy.get('description', '')}"
    vac_skills = extract_skills(vac_text, skills_dict)
    
    print(f"\nВакансия: {vacancy.get('title')}")
    print(f"Извлеченные навыки вакансии ({len(vac_skills)}): {list(vac_skills)}\n")
    print("-" * 60)
    
    # Максимально возможная сумма весов всех навыков вакансии
    max_vac_weight = sum(idf_weights.get(skill, 1.0) for skill in vac_skills)
    
    results = []
    
    for idx, resume in enumerate(resumes):
        res_text = f"{resume.get('title', '')} {resume.get('skills_raw', '')} {resume.get('description', '')}"
        res_skills = extract_skills(res_text, skills_dict)
        
        matched_skills = vac_skills.intersection(res_skills)
        missing_skills = vac_skills - res_skills
        
        # 1. Базовый Score (простой процент совпадения)
        base_score = (len(matched_skills) / len(vac_skills) * 100) if vac_skills else 0.0
        
        # 2. TF-IDF Score (взвешенный процент совпадения)
        matched_weight = sum(idf_weights.get(skill, 1.0) for skill in matched_skills)
        tfidf_score = (matched_weight / max_vac_weight * 100) if max_vac_weight > 0 else 0.0
        
        results.append({
            "resume_id": idx + 1,
            "title": resume.get("title"),
            "base_score": round(base_score, 1),
            "tfidf_score": round(tfidf_score, 1),
            "matched_skills": list(matched_skills),
            "missing_skills": list(missing_skills),
            "total_candidate_skills": list(res_skills)
        })
        
    # Ранжирование по TF-IDF score
    results.sort(
        key=lambda x: (x["tfidf_score"], len(x["matched_skills"]), len(x["total_candidate_skills"])), 
        reverse=True
    )
    
    print("ТОП-3 КАНДИДАТА ДЛЯ ВАКАНСИИ (С УЧЕТОМ TF-IDF ВЕСОВ):\n")
    for rank, res in enumerate(results[:3], start=1):
        print(f"Место #{rank} | Резюме ID: {res['resume_id']} | TF-IDF Score: {res['tfidf_score']}% (Базовый: {res['base_score']}%)")
        print(f"Совпавшие навыки: {res['matched_skills']}")
        print(f"Недостающие навыки: {res['missing_skills']}")
        print(f"Все навыки кандидата: {res['total_candidate_skills']}")
        print("-" * 60)


if __name__ == "__main__":
    run_matching()