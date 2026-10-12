from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/LaBSE')

# векторы
vec_resume = model.encode("Опытный Python разработчик, знаю Django, PostgreSQL, Docker")
vec_job = model.encode("Ищем Backend-программиста на Python (FastAPI/Django)")

# сходство (выдаст число от 0 до 1)
similarity = util.cos_sim(vec_resume, vec_job)
print(f"Сходство: {similarity.item():.2f}")