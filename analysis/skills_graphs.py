import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. ОПРЕДЕЛЕНИЕ ПУТЕЙ К ПРОЕКТУ
# Определяем текущую папку скрипта (analysis) и корень проекта (на уровень выше)
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

# Указываем точный путь к папке со словарями
DICT_DIR = BASE_DIR / "datasets_and_skills" / "skills"


# 2. ФУНКЦИЯ КАТЕГОРИЗАЦИИ НАВЫКОВ
def categorize_skill(s):
    name = s["name"].strip()
    nl = name.lower()
    stype = s.get("type", "")

    if stype == "soft" or ("язык" in nl and nl not in ["язык r"]):
        if any(
            lang in nl
            for lang in [
                "английский",
                "немецкий",
                "китайский",
                "французский",
                "испанский",
                "русский",
                "турецкий",
                "итальянский",
                "польский",
                "арабский",
                "эстонский",
                "казахский",
                "белорусский",
                "украинский",
                "португальский",
            ]
        ):
            return "Иностранные языки"
        return "Soft Skills"

    if any(
        k in nl
        for k in [
            "test",
            "qa",
            "selenium",
            "playwright",
            "cypress",
            "jest",
            "junit",
            "тестирование",
        ]
    ):
        return "Тестирование (QA)"

    if any(
        k in nl
        for k in [
            "pandas",
            "numpy",
            "pytorch",
            "nlp",
            "llm",
            "computer vision",
            "machine learning",
            "deep learning",
            "нейросети",
            "data science",
            "tensorflow",
            "scikit-learn",
            "keras",
            "xgboost",
            "lightgbm",
            "opencv",
            "catboost",
            "rag",
            "embeddings",
            "yolo",
            "huggingface",
            "langchain",
            "ollama",
            "bert",
            "stable diffusion",
            "matplotlib",
            "seaborn",
            "mlflow",
            "kubeflow",
        ]
    ):
        return "Data Science & AI / ML"

    if any(
        k in nl
        for k in [
            "postgresql",
            "mysql",
            "sql",
            "redis",
            "mongodb",
            "clickhouse",
            "elasticsearch",
            "kibana",
            "sqlite",
            "neo4j",
            "cassandra",
            "векторные бд",
            "hadoop",
            "hive",
            "spark",
            "kafka",
            "airflow",
            "dwh",
            "data lake",
            "olap",
            "dbt",
            "snowflake",
            "databricks",
            "teradata",
            "vertica",
            "greenplum",
            "redshift",
            "bigquery",
            "couchdb",
            "couchbase",
            "dynamodb",
            "scylladb",
            "solr",
            "influxdb",
            "memcached",
            "data engineering",
            "etl",
            "big data",
            "impala",
            "hbase",
            "apache flink",
            "apache beam",
            "delta lake",
        ]
    ):
        return "Базы данных & Big Data"

    if any(
        k in nl
        for k in [
            "docker",
            "linux",
            "bash",
            "ci/cd",
            "devops",
            "kubernetes",
            "ansible",
            "terraform",
            "grafana",
            "prometheus",
            "zabbix",
            "sentry",
            "nginx",
            "виртуализация",
            "active directory",
            "aws",
            "azure",
            "google cloud",
            "yandex cloud",
            "cloud computing",
            "sre",
            "vault",
            "consul",
            "istio",
            "helm",
            "argocd",
            "openstack",
            "hyper-v",
            "podman",
            "serverless",
            "jenkins",
            "gitlab",
            "teamcity",
            "circleci",
            "travis",
            "puppet",
            "chef",
            "saltstack",
            "nagios",
            "loki",
            "opentelemetry",
            "jaeger",
            "zipkin",
            "datadog",
            "new relic",
            "dynatrace",
            "pagerduty",
            "graylog",
            "haproxy",
            "traefik",
            "cloudflare",
            "xen",
            "zookeeper",
        ]
    ):
        return "DevOps, Cloud & SysAdmin"

    if any(
        k in nl
        for k in [
            "tcp/ip",
            "dns",
            "dhcp",
            "vpn",
            "ssh",
            "cisco",
            "mikrotik",
            "wireshark",
            "kali linux",
            "burp suite",
            "siem",
            "информационная безопасность",
            "пентестинг",
            "devsecops",
            "zero trust",
            "edr/xdr",
            "soar",
            "dlp",
            "iam",
            "pki",
        ]
    ):
        return "Сети & Кибербезопасность"

    if any(
        k in nl
        for k in [
            "python",
            "django",
            "fastapi",
            "javascript",
            "typescript",
            "react",
            "node.js",
            "html/css",
            "c++",
            "java",
            "c#",
            "go",
            "ооп",
            "rest api",
            "git",
            "github",
            "rabbitmq",
            "celery",
            "flask",
            "spring",
            "vue",
            "angular",
            "next.js",
            "graphql",
            "grpc",
            "websockets",
            "php",
            "laravel",
            "symfony",
            "ruby",
            "rust",
            "kotlin",
            "swift",
            "dart",
            "flutter",
            "asp.net",
            "unity",
            "unreal",
            "express",
            "nestjs",
            "redux",
            "sass",
            "bootstrap",
            "webpack",
            "svelte",
            "nuxt",
            "gatsby",
            "tailwind",
            "scala",
            "groovy",
            "perl",
            "delphi",
            "assembly",
            "fortran",
            "cobol",
            "abap",
            "elixir",
            "haskell",
            "lua",
            "matlab",
            "r",
            "stata",
            "sas",
            "xml",
            "yaml",
            "soap",
            "swagger",
            "powershell",
            "ios / android",
            "микросервисы",
            "парсинг данных",
            "регулярные выражения",
            "pulsar",
            "nats",
            "message broker",
            "event-driven",
            "onnx",
            "postman",
            "json",
        ]
    ):
        return "Разработка ПО"

    if any(
        k in nl
        for k in [
            "анализ данных",
            "power bi",
            "tableau",
            "metabase",
            "продуктовая аналитика",
            "юнит-экономика",
            "веб-аналитика",
            "статистический анализ",
            "a/b тестирование",
            "визуализация данных",
            "бизнес-анализ",
            "кро",
            "data visualization",
        ]
    ):
        return "Аналитика данных & Продукт"

    if any(
        k in nl
        for k in [
            "figma",
            "photoshop",
            "illustrator",
            "indesign",
            "after effects",
            "premiere pro",
            "tilda",
            "wordpress",
            "magento",
            "shopify",
            "ux/ui",
            "дизайн",
            "маркетинг",
            "smm",
            "seo",
            "контекстная реклама",
            "копирайтинг",
            "таргетированная реклама",
            "btl",
            "мерчандайзинг",
            "sem/ppc",
            "реклама в соцсетях",
            "email-маркетинг",
            "бренд-менеджмент",
            "pr",
            "контент-стратегия",
            "видеомонтаж",
            "фотография",
            "графический дизайн",
            "организация мероприятий",
            "исследования рынка",
        ]
    ):
        return "Маркетинг, Дизайн & Контент"

    if any(
        k in nl
        for k in [
            "продажи",
            "b2b продажи",
            "колл-центр",
            "работа с клиентами",
            "fmcg",
            "розничные продажи",
            "pos-системы",
        ]
    ):
        return "Продажи & Работа с клиентами"

    if any(
        k in nl
        for k in [
            "бухгалтерия",
            "бухгалтерский учёт",
            "налоги",
            "финансовый анализ",
            "закупки",
            "госзакупки",
            "вэд",
            "складской учет",
            "кассовые операции",
            "банковское дело",
            "платежные системы",
            "страхование",
            "инвестиции",
            "финансовый мониторинг",
            "1с бухгалтерия",
            "1с управление торговлей",
            "контур",
            "сбис",
            "такском",
            "клиент-банк",
            "дебеторка",
            "финансовая отчётность",
            "бюджетирование",
            "зарплатные системы",
            "quickbooks",
            "netsuite",
            "sage",
            "sox",
            "aml/kyc",
            "кредитный анализ",
            "урегулирование претензий",
            "актуарные",
            "банковские операции",
        ]
    ):
        return "Финансы, Учёт & Закупки"

    if any(
        k in nl
        for k in [
            "hr",
            "рекрутинг",
            "документооборот",
            "управление персоналом",
            "кадровое делопроизводство",
            "1с зуп",
            "компенсации и льготы",
            "управление эффективностью",
            "hris",
            "обучение и развитие",
            "управление проектами",
            "agile",
            "scrum",
            "kanban",
            "lean",
            "методологии разработки",
            "стратегическое планирование",
            "управление рисками",
            "управление изменениями",
            "планирование спроса",
        ]
    ):
        return "HR & Управление (Management)"

    if any(
        k in nl
        for k in [
            "юриспруденция",
            "договорная работа",
            "судебная практика",
            "интеллектуальная собственность",
            "due diligence",
            "иммиграционное право",
            "комплаенс",
        ]
    ):
        return "Юриспруденция & Комплаенс"

    if any(
        k in nl
        for k in [
            "гис",
            "геодезия",
            "bim",
            "autocad",
            "blender",
            "охрана труда",
            "сметное дело",
            "строительный контроль",
            "электромонтаж",
            "сантехнические",
            "сварочные",
            "станки с чпу",
            "грузоперевозки",
            "водитель погрузчика",
            "авторемонт",
            "тяжёлая техника",
            "встраиваемые системы",
            "fpga",
            "проектирование печатных плат",
            "iot",
            "5g/lte",
            "цифровая обработка",
            "робототехника",
            "solidworks",
            "catia",
            "ansys",
        ]
    ):
        return "Инженерия, Производство & Строительство"

    if any(
        k in nl
        for k in [
            "фармацевтика",
            "медицина",
            "bls/cpr",
            "уход за пациентами",
            "психическое здоровье",
            "медсестринское дело",
            "экг",
            "радиология",
            "уход на дому",
            "флеботомия",
            "медицинское кодирование",
            "hipaa",
            "emr/ehr",
            "epic",
            "реабилитация",
        ]
    ):
        return "Медицина, Фармацевтика & Уход"

    if any(
        k in nl
        for k in [
            "педагогика",
            "психология",
            "общепит",
            "гостиничный",
            "клининг",
            "туризм",
            "первая помощь",
            "видеонаблюдение",
            "охранная",
            "водительское",
            "медицинская книжка",
            "ввод данных",
        ]
    ):
        return "Образование, Сервис & Рабочие профессии"

    if any(
        k in nl
        for k in [
            "excel",
            "ms word",
            "powerpoint",
            "outlook",
            "google workspace",
            "ms office",
            "ms access",
            "1с",
            "битрикс24",
            "salesforce",
            "sap",
            "oracle",
            "jira",
            "confluence",
            "trello",
            "notion",
            "slack",
            "microsoft teams",
            "zoom",
            "sharepoint",
            "power platform",
            "презентации",
        ]
    ):
        return "Офисные инструменты & Корпоративное ПО"

    return "Прочие навыки"


# ----------------------------------------------------------------------
# 3. ПОСТРОЕНИЕ И СОХРАНЕНИЕ ГРАФИКОВ
# ----------------------------------------------------------------------
def plot_dictionary_stats(file_path: Path, title: str, palette_name: str, output_filename: Path):
    if not file_path.exists():
        print(f"Пропущен файл (не найден по пути): {file_path}")
        return

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["category"] = df.apply(categorize_skill, axis=1)

    counts = df["category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(
        data=counts,
        x="Count",
        y="Category",
        hue="Category",
        palette=palette_name,
        legend=False,
    )

    plt.title(f"{title} (Всего навыков: {len(df)})", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Количество навыков", fontsize=12)
    plt.ylabel("")

    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(
                f"{int(width)}",
                (width + 0.3, p.get_y() + p.get_height() / 2.0),
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#333333",
            )

    output_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"График успешно сохранён в: {output_filename}")


if __name__ == "__main__":
    # Запуск построения графиков
    plot_dictionary_stats(
        DICT_DIR / "skills_dict.json",
        "Словарь V1",
        "viridis",
        SCRIPT_DIR / "chart_v1.png",
    )
    plot_dictionary_stats(
        DICT_DIR / "skills_dict_v2.json",
        "Словарь V2",
        "mako",
        SCRIPT_DIR / "chart_v2.png",
    )
    plot_dictionary_stats(
        DICT_DIR / "skills_dict_v3.json",
        "Словарь V3",
        "rocket",
        SCRIPT_DIR / "chart_v3.png",
    )