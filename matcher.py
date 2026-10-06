import json
import re

# функция приведения словаря навыков к единому формату {навык: [синонимы]}
def normalize_skills_dict(skills_data):
    normalized = {}
    if isinstance(skills_data, dict):
        for k, v in skills_data.items():
            normalized[k] = v if isinstance(v, list) else [str(v)]
    elif isinstance(skills_data, list):
        for item in skills_data:
            if isinstance(item, dict):

                # формат: {"skill": "Python", "aliases": ["py", "python3"]}
                name = item.get("skill") or item.get("name") or item.get("title")
                aliases = item.get("aliases") or item.get("synonyms") or item.get("tags") or []
                
                if name:
                    normalized[name] = aliases if isinstance(aliases, list) else [str(aliases)]
                else:
                    # формат: {"Python": ["py", "python3"]}
                    for k, v in item.items():
                        normalized[k] = v if isinstance(v, list) else [str(v)]
            elif isinstance(item, str):
                normalized[item] = []
    return normalized

# загрузка данных
def load_data():
    with open("./datasets_and_skills/skills/skills_dict.json", "r", encoding="utf-8") as f:
        skills_raw = json.load(f)
    with open("./data_old/vacancies_ru_ready.json", "r", encoding="utf-8") as f:
        vacancies = json.load(f)
    with open("./data_old/resumes_en_ready.json", "r", encoding="utf-8") as f:
        resumes = json.load(f)
        
    skills_dict = normalize_skills_dict(skills_raw)
    return skills_dict, vacancies, resumes

# функция извлечения навыков из текста
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

# основной процесс мэтчинга
def run_matching():
    skills_dict, vacancies, resumes = load_data()
    
    print("ТЕСТИРОВАНИЕ АЛГОРИТМА МЭТЧИНГА\n")
    
    vacancy = vacancies[0]
    vac_text = f"{vacancy.get('title', '')} {vacancy.get('skills_raw', '')} {vacancy.get('description', '')}"
    vac_skills = extract_skills(vac_text, skills_dict)
    
    print(f"Вакансия: {vacancy.get('title')}")
    print(f"Извлеченные навыки вакансии ({len(vac_skills)}): {list(vac_skills)}\n")
    print("-" * 60)
    
    results = []
    
    for idx, resume in enumerate(resumes):
        res_text = f"{resume.get('title', '')} {resume.get('skills_raw', '')} {resume.get('description', '')}"
        res_skills = extract_skills(res_text, skills_dict)
        
        matched_skills = vac_skills.intersection(res_skills)
        missing_skills = vac_skills - res_skills
        
        score = (len(matched_skills) / len(vac_skills) * 100) if vac_skills else 0.0
        
        results.append({
            "resume_id": idx + 1,
            "title": resume.get("title"),
            "score": round(score, 1),
            "matched_skills": list(matched_skills),
            "missing_skills": list(missing_skills),
            "total_candidate_skills": list(res_skills)
        })
        
    # ранжирование:
    # 1. по проценту совпадения (score) — по убыванию
    # 2. по количеству совпавших навыков — по убыванию
    # 3. по общему числу навыков у кандидата — по убыванию
    results.sort(
    key=lambda x: (x["score"], len(x["matched_skills"]), len(x["total_candidate_skills"])), reverse=True)
    
    print("ТОП-3 КАНДИДАТА ДЛЯ ВАКАНСИИ:\n")
    for rank, res in enumerate(results[:3], start=1):
        print(f"Место #{rank} | Резюме ID: {res['resume_id']} | Совпадение: {res['score']}%")
        print(f"Совпавшие навыки: {res['matched_skills']}")
        print(f"Недостающие навыки: {res['missing_skills']}")
        print(f"Все навыки кандидата: {res['total_candidate_skills']}")
        print("-" * 60)

if __name__ == "__main__":
    run_matching()