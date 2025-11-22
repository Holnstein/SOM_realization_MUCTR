import numpy as np
import random
from SOMNode import CSOMNode


class CSOM:
    """
    Карта самоорганизующихся карт (SOM).

    Поддерживает:
        - Обучение
        - Расчёт ошибки квантования
        - Сохранение истории весов и ошибок
    """
    def __init__(self, hexagonal: bool = True, show_borders: bool = True):
        self.hexagonal = hexagonal
        self.show_borders = show_borders
        self.nodes = []             # List[CSOMNode]
        self.training_data = []     # List[np.ndarray], каждый размерности (3,) это все цвета из WEB_COLORS, которые добавили через som.add_data([r, g, b]).
        self.initial_lr = 0.3       # скорость обучения
        self.map_radius = 0.0       # радиус карты
        self.time_constant = 0.0    # константа затухания
        self.bmu_history = []

    def init_map(self, iterations: int, xcells: int, ycells: int, width: float, height: float):
        """
        Инициализация сетки SOM.

        Параметры:
            iterations (int): общее число итераций обучения
            xcells, ycells (int): размер сетки N x M
            width, height (float): физические размеры карты
        """
        self.iterations = iterations
        self.xcells = xcells
        self.ycells = ycells
        self.width = width
        self.height = height

        cell_w = width / xcells # ширина ячейки
        cell_h = height / ycells # высота ячейки
        self.nodes = []
        for i in range(xcells):
            for j in range(ycells):
                x1 = i * cell_w
                y1 = j * cell_h
                x2 = (i + 1) * cell_w
                y2 = (j + 1) * cell_h
                self.nodes.append(CSOMNode(x1, y1, x2, y2)) # передача координат границ ячейки

        self.map_radius = max(width, height) / 2.0 # радиус карты
        self.time_constant = iterations / np.log(self.map_radius) # константа затухания

    def add_data(self, vector: np.ndarray):
        """Добавить обучающий вектор (размерность (3,))."""
        self.training_data.append(np.array(vector, dtype=float))

    def quantization_error(self) -> float:
        """
        Вычисляет среднюю ошибку квантования (Quantization Error).

        Формула: QE = (1/N) * Σ || x_i - w_BMU(x_i) ||^2

        Возвращает:
            float: средняя ошибка квантования по всем обучающим векторам
        """ 
        if not self.training_data:
            return 0.0
        total_error = 0.0
        for x in self.training_data:
            bmu_idx = self._find_bmu(x)
            total_error += self.nodes[bmu_idx].distance(x)
        return total_error / len(self.training_data)

    def _find_bmu(self, vector: np.ndarray) -> int:
        """Найти индекс BMU для вектора."""
        distances = [node.distance(vector) for node in self.nodes]
        return int(np.argmin(distances))

    def train_with_snapshots(self, snapshot_interval: int = 200):
        """
        Обучение SOM с сохранением снимков, ошибки и истории BMU.

        Возвращает:
            snapshots: List[(step, weights)], weights — (N*M, 3)
            q_errors: List[float] — ошибка квантования по шагам
        """
        total = len(self.training_data)
        if total == 0:
            raise ValueError("Нет обучающих данных.")

        weights_history = []
        q_errors = []
        self.bmu_history = []

        for step in range(self.iterations):  # эпоха
            vector = random.choice(self.training_data)  # Размер: (3,) случайно выбирает один из этих цветов на каждой итерации.
            bmu_idx = self._find_bmu(vector) # Индекс лучшей ячейки
            self.bmu_history.append(bmu_idx) # Добавляем индекс лучшей ячейки в историю

            radius = self.map_radius * np.exp(-step / self.time_constant) # радиус у лучшей ячейки
            lr = self.initial_lr * np.exp(-step / self.iterations) # скорость обучения

            for i, node in enumerate(self.nodes): # i - индекс узла, node - сам узел (СSOMNode)
                dist_sq = (node.x - self.nodes[bmu_idx].x)**2 + (node.y - self.nodes[bmu_idx].y)**2 # Считает квадрат евклидова расстояния между текущим узлом node и BMU.
                if dist_sq < radius**2: # проверка находится ли текущий узел внутри радиуса влияния BMU
                    influence = np.exp(-dist_sq / (2 * radius**2)) # Вычисляет влияние соседства на основе гауссовой функции. Чем ближе узел к BMU — тем сильнее влияние.
                    node.adjust(vector, lr, influence) # обновление весов узла

            # если шаг кратен интервалу сохранения, то сохраняем снимок
            if step % snapshot_interval == 0 or step == self.iterations - 1:
                weights = np.array([node.weights for node in self.nodes])  # (N*M, 3)
                weights_history.append((step, weights.copy()))
                q_errors.append(self.quantization_error())

        return weights_history, q_errors