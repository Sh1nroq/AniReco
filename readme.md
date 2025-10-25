# Anime Recommender (AniReco)

![Python](https://img.shields.io/badge/python-3.11-blue) ![PyTorch](https://img.shields.io/badge/pytorch-2.1-orange) ![PostgreSQL](https://img.shields.io/badge/postgresql-15-blue) ![MIT License](https://img.shields.io/badge/license-MIT-green)

**Anime Recommender** — персонализированная система рекомендаций аниме, которая подбирает контент на основе предпочтений пользователя.  
Проект сочетает машинное обучение и внешние API для анализа вкусов и подбора подходящих аниме.

---

## 🔥 Особенности

- Генерация персонализированных рекомендаций на основе истории предпочтений.  
- Использует API [Jikan](https://jikan.moe/) для получения данных о популярных аниме.  
- Хранение информации о пользователях и рейтингах в PostgreSQL.  
- Быстрая сериализация данных через `orjson`.  
- Модель на PyTorch для предсказания интереса пользователя к аниме.  

---

## 🛠 Стек технологий

- **Язык программирования:** Python 3.11  
- **Библиотеки:** PyTorch, requests, orjson  
- **База данных:** PostgreSQL 15  
- **API:** Jikan API  

---

## 🏗 Архитектура модели

1. **Входные данные:** пользовательские оценки и история просмотров.  
2. **Предобработка:** нормализация оценок, кодирование жанров.  
3. **Модель:** PyTorch feedforward neural network / embedding-based recommender (user-item).  
4. **Выход:** прогноз вероятности, что пользователю понравится конкретное аниме.  

> Можно легко расширять до рекуррентных или трансформерных моделей для более сложных рекомендаций.

---

## 🚀 Установка

### Локальная установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/Sh1nroq/AnimeRecomendationNeuronNetwork.git
cd nimeRecomendationNeuronNetwork
```

2. Настройте окружение
```
conda create -n AnimeRecomendation
conda activate AnimeRecomendation
```

3. Установите зависимости:
```
pip install -r requirements.txt
```

## ⚡ Использование
Получение рекомендаций
```python main.py --user_id 123```

## 📂 Структура проекта
```
anime-recommender/
├─ data/                 # Данные и кеш API
├─ models/               # PyTorch модели
├─ scripts/              # Скрипты для обучения и получения рекомендаций
├─ requirements.txt
├─ Dockerfile
├─ README.md
```
## 📜 Лицензия

MIT License © 2025
