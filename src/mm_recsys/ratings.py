"""Простая статистика по оценкам из отзывов."""


def mean_rating(ratings: list[float]) -> float:
    """Возвращает среднюю оценку. Для пустого списка возвращает 0.0."""
    if not ratings:
        return 0.0
    return sum(ratings) / len(ratings)
