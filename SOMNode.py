import numpy as np


class CSOMNode:
    """
    Узел карты самоорганизующихся карт (SOM).

    Атрибуты:
        x, y (float): координаты узла на сетке (для вычисления расстояний между узлами)
        weights (np.ndarray): веса узла, размерность (3,) — например, RGB
    """
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        """
        Инициализация узла SOM.

        Параметры:
            x1, y1, x2, y2 (float): координаты границ ячейки
        """
        self.x = (x1 + x2) / 2.0
        self.y = (y1 + y2) / 2.0
        self.weights = np.random.rand(3) * 255.0  # Размерность: (3,)

    def distance(self, vector: np.ndarray) -> float:
        """
        Евклидово расстояние между весами узла и входным вектором.

        Формула: || w - x ||^2

        Параметры:
            vector (np.ndarray): входной вектор, размерность (3,)

        Возвращает:
            float: квадрат расстояния
        """
        return np.sum((self.weights - vector) ** 2)

    def adjust(self, vector: np.ndarray, lr: float, influence: float):
        """
        Обновление весов узла.

        Формула: w_new = w + lr * influence * (x - w)

        Параметры:
            vector (np.ndarray): входной вектор, размерность (3,)
            lr (float): скорость обучения
            influence (float): влияние соседства
        """
        self.weights += lr * influence * (vector - self.weights)