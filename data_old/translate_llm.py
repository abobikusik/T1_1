import json
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# кол-во резюме для перевода
MAX_RESUMES = 5

def translate_with_llm(text):
    if not text or not str(text).strip():
        return ""
        
    system_prompt = (
        "Ты профессиональный IT-переводчик. Переведи текст резюме с английского на русский.\n"
        "ПРАВИЛА:\n"
        "1. Все названия технологий, языков программирования, фреймворков и компаний "
        "(например: Python, SQL, Docker, Spark, Coca-Cola) ОСТАВЛЯЙ НА АНГЛИЙСКОМ.\n"
        "2. Верни ТОЛЬКО переведенный текст без дополнительных пояснений."
    )

    try:
        response = client.chat.completions.create(
            model="qwen2.5:3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(text)[:1000]}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Ошибка перевода: {e}")
        return text

# 1. Читаем исходный файл
with open("resumes_en_ready.json", "r", encoding="utf-8") as f:
    resumes_en = json.load(f)

# 2. Берем только первые MAX_RESUMES записей
resumes_to_process = resumes_en[:MAX_RESUMES]
resumes_ru = []
total = len(resumes_to_process)

print(f"Начинаем перевод {total} резюме из {len(resumes_en)} через Ollama...\n")

# 3. Запускаем перевод
for idx, item in enumerate(resumes_to_process, start=1):
    print(f"[{idx}/{total}] Перевод резюме ID {idx}...")
    
    translated_desc = translate_with_llm(item.get("description", ""))
    
    resumes_ru.append({
        "id": idx,
        "title": item.get("title"),
        "skills_raw": item.get("skills_raw"),
        "description": translated_desc
    })

# 4. Сохраняем в итоговый JSON
with open("resumes_ru_ready.json", "w", encoding="utf-8") as f:
    json.dump(resumes_ru, f, ensure_ascii=False, indent=2)

print(f"\nГотово! Переведенные {total} резюме сохранены в 'resumes_ru_ready.json'")