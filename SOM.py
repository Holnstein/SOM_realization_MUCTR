import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import random

# Список web-цветов с именами
WEB_COLORS = [
    (0, 0, 0, "Black"),
    (0, 100, 0, "DarkGreen"),
    (47, 79, 79, "DarkSlateGray"),
    (128, 128, 0, "Olive"),
    (0, 128, 0, "Green"),
    (0, 128, 128, "Teal"),
    (0, 0, 128, "Navy"),
    (128, 0, 128, "Purple"),
    (128, 0, 0, "Maroon"),
    (75, 0, 130, "Indigo"),
    (25, 25, 112, "MidnightBlue"),
    (0, 0, 139, "DarkBlue"),
    (85, 107, 47, "DarkOliveGreen"),
    (139, 69, 19, "SaddleBrown"),
    (34, 139, 34, "ForestGreen"),
    (107, 142, 35, "OliveDrab"),
    (46, 139, 87, "SeaGreen"),
    (184, 134, 11, "DarkGoldenrod"),
    (72, 61, 139, "DarkSlateBlue"),
    (160, 82, 45, "Sienna"),
    (0, 0, 205, "MediumBlue"),
    (165, 42, 42, "Brown"),
    (0, 206, 209, "DarkTurquoise"),
    (105, 105, 105, "DimGray"),
    (32, 178, 170, "LightSeaGreen"),
    (148, 0, 211, "DarkViolet"),
    (178, 34, 34, "FireBrick"),
    (199, 21, 133, "MediumVioletRed"),
    (60, 179, 113, "MediumSeaGreen"),
    (210, 105, 30, "Chocolate"),
    (220, 20, 60, "Crimson"),
    (70, 130, 180, "SteelBlue"),
    (218, 165, 32, "Goldenrod"),
    (0, 250, 154, "MediumSpringGreen"),
    (124, 252, 0, "LawnGreen"),
    (95, 158, 160, "CadetBlue"),
    (153, 50, 204, "DarkOrchid"),
    (154, 205, 50, "YellowGreen"),
    (50, 205, 50, "LimeGreen"),
    (255, 69, 0, "OrangeRed"),
    (255, 140, 0, "DarkOrange"),
    (255, 165, 0, "Orange"),
    (255, 215, 0, "Gold"),
    (255, 255, 0, "Yellow"),
    (127, 255, 0, "Chartreuse"),
    (0, 255, 0, "Lime"),
    (0, 255, 127, "SpringGreen"),
    (127, 255, 212, "Aquamarine"),
    (0, 191, 255, "DeepSkyBlue"),
    (0, 0, 255, "Blue"),
    (255, 0, 255, "Magenta"),
    (255, 0, 0, "Red"),
    (128, 128, 128, "Gray"),
    (112, 128, 144, "SlateGray"),
    (205, 133, 63, "Peru"),
    (138, 43, 226, "BlueViolet"),
    (119, 136, 153, "LightSlateGray"),
    (255, 20, 147, "DeepPink"),
    (72, 209, 204, "MediumTurquoise"),
    (30, 144, 255, "DodgerBlue"),
    (64, 224, 208, "Turquoise"),
    (65, 105, 225, "RoyalBlue"),
    (106, 90, 205, "SlateBlue"),
    (189, 183, 107, "DarkKhaki"),
    (205, 92, 92, "IndianRed"),
    (186, 85, 211, "MediumOrchid"),
    (173, 255, 47, "GreenYellow"),
    (102, 205, 170, "MediumAquamarine"),
    (143, 188, 143, "DarkSeaGreen"),
    (255, 99, 71, "Tomato"),
    (188, 143, 143, "RosyBrown"),
    (218, 112, 214, "Orchid"),
    (176, 48, 96, "PaleVioletRed"),
    (255, 127, 80, "Coral"),
    (100, 149, 237, "CornflowerBlue"),
    (169, 169, 169, "DarkGray"),
    (244, 164, 96, "SandyBrown"),
    (123, 104, 238, "MediumSlateBlue"),
    (210, 180, 140, "Tan"),
    (233, 150, 122, "DarkSalmon"),
    (222, 184, 135, "BurlyWood"),
    (255, 105, 180, "HotPink"),
    (250, 128, 114, "Salmon"),
    (238, 130, 238, "Violet"),
    (240, 128, 128, "LightCoral"),
    (135, 206, 250, "SkyBlue"),
    (255, 160, 122, "LightSalmon"),
    (221, 160, 221, "Plum"),
    (240, 230, 140, "Khaki"),
    (144, 238, 144, "LightGreen"),
    (127, 255, 212, "Aquamarine"),
    (192, 192, 192, "Silver"),
    (135, 206, 250, "LightSkyBlue"),
    (176, 196, 222, "LightSteelBlue"),
    (173, 216, 230, "LightBlue"),
    (152, 251, 152, "PaleGreen"),
    (221, 160, 221, "Thistle"),
    (176, 224, 230, "PowderBlue"),
    (238, 232, 170, "PaleGoldenrod"),
    (175, 238, 238, "PaleTurquoise"),
    (211, 211, 211, "LightGray"),
    (245, 222, 179, "Wheat"),
    (255, 222, 173, "NavajoWhite"),
    (255, 228, 181, "Moccasin"),
    (255, 182, 193, "LightPink"),
    (220, 220, 220, "Gainsboro"),
    (255, 218, 185, "PeachPuff"),
    (255, 192, 203, "Pink"),
    (255, 228, 196, "Bisque"),
    (255, 218, 165, "LightGoldenrod"),
    (255, 235, 205, "BlanchedAlmond"),
    (255, 250, 205, "LemonChiffon"),
    (245, 245, 220, "Beige"),
    (250, 235, 215, "AntiqueWhite"),
    (255, 239, 213, "PapayaWhip"),
    (255, 248, 220, "Cornsilk"),
    (255, 255, 224, "LightYellow"),
    (224, 255, 255, "LightCyan"),
    (250, 240, 230, "Linen"),
    (230, 230, 250, "Lavender"),
    (255, 228, 225, "MistyRose"),
    (253, 245, 230, "OldLace"),
    (245, 245, 245, "WhiteSmoke"),
    (255, 245, 238, "Seashell"),
    (255, 255, 240, "Ivory"),
    (240, 255, 240, "Honeydew"),
    (240, 248, 255, "AliceBlue"),
    (255, 240, 245, "LavenderBlush"),
    (245, 255, 250, "MintCream"),
    (255, 250, 250, "Snow"),
    (255, 255, 255, "White")
]


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


class CSOM:
    """
    Карта самоорганизующихся карт (SOM).

    Поддерживает:
        - Анимацию обучения
        - Расчёт ошибки квантования
        - Отслеживание BMU-траектории
        - Визуализацию в 3D
    """
    def __init__(self, hexagonal: bool = True, show_borders: bool = True):
        self.hexagonal = hexagonal
        self.show_borders = show_borders
        self.nodes = []             # List[CSOMNode]
        self.training_data = []     # List[np.ndarray], каждый размерности (3,)
        self.initial_lr = 0.1
        self.map_radius = 0.0
        self.time_constant = 0.0
        self.bmu_history = []       # Список индексов BMU на каждом шаге (для визуализации траектории)

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

        cell_w = width / xcells
        cell_h = height / ycells
        self.nodes = []
        for i in range(xcells):
            for j in range(ycells):
                x1 = i * cell_w
                y1 = j * cell_h
                x2 = (i + 1) * cell_w
                y2 = (j + 1) * cell_h
                self.nodes.append(CSOMNode(x1, y1, x2, y2))

        self.map_radius = max(width, height) / 2.0
        self.time_constant = iterations / np.log(self.map_radius)

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

        for step in range(self.iterations):
            vector = random.choice(self.training_data)  # Размер: (3,)
            bmu_idx = self._find_bmu(vector)
            self.bmu_history.append(bmu_idx)

            radius = self.map_radius * np.exp(-step / self.time_constant)
            lr = self.initial_lr * np.exp(-step / self.iterations)

            for i, node in enumerate(self.nodes):
                dist_sq = (node.x - self.nodes[bmu_idx].x)**2 + (node.y - self.nodes[bmu_idx].y)**2
                if dist_sq < radius**2:
                    influence = np.exp(-dist_sq / (2 * radius**2))
                    node.adjust(vector, lr, influence)

            if step % snapshot_interval == 0 or step == self.iterations - 1:
                weights = np.array([node.weights for node in self.nodes])  # (N*M, 3)
                weights_history.append((step, weights.copy()))
                q_errors.append(self.quantization_error())

        return weights_history, q_errors


def animate_som(hexagonal=True, save_gif=False):
    random.seed(200)
    np.random.seed(200)

    som = CSOM(hexagonal=hexagonal)
    som.init_map(iterations=3000, xcells=20, ycells=20, width=20, height=20)

    rgb_list = []
    name_list = []
    for r, g, b, name in WEB_COLORS:
        som.add_data([r, g, b])
        rgb_list.append((r, g, b))
        name_list.append(name)

    print("Начинаем обучение...")
    snapshots, q_errors = som.train_with_snapshots(snapshot_interval=100)

    # === 1. Анимация SOM ===
    fig1, (ax_map, ax_error) = plt.subplots(1, 2, figsize=(14, 6))
    ax_map.set_aspect('equal')
    ax_map.axis('off')

    if hexagonal:
        hex_radius = 0.5
        patches = []
        for idx in range(len(som.nodes)):
            i = idx // som.ycells
            j = idx % som.ycells
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            hexagon = RegularPolygon((x, y), 6, radius=hex_radius, orientation=np.radians(30),
                                     facecolor='white', edgecolor='gray')
            patches.append(hexagon)
            ax_map.add_patch(hexagon)
        ax_map.set_xlim(-1, som.xcells + 1)
        ax_map.set_ylim(-1, som.ycells * np.sqrt(3) / 2 + 1)
    else:
        patches = []
        for idx in range(len(som.nodes)):
            i = idx // som.ycells
            j = idx % som.ycells
            rect = plt.Rectangle((i, j), 1, 1, facecolor='white', edgecolor='gray')
            patches.append(rect)
            ax_map.add_patch(rect)
        ax_map.set_xlim(0, som.xcells)
        ax_map.set_ylim(0, som.ycells)

    ax_map.invert_yaxis()

    # Ошибки
    steps = [s[0] for s in snapshots]
    ax_error.plot(steps, q_errors, 'b-')
    ax_error.set_xlabel('Итерация')
    ax_error.set_ylabel('Ошибка квантования')
    ax_error.set_title('Сходимость модели')
    ax_error.grid(True)

    line_error = ax_error.axvline(x=0, color='r', linestyle='--')

    def update(frame):
        step, weights = snapshots[frame]
        for patch, w in zip(patches, weights):
            color = np.clip(w / 255.0, 0, 1)
            patch.set_facecolor(color)
        fig1.suptitle(f'Обучение SOM — Шаг {step}', fontsize=14)
        line_error.set_xdata([step])
        return patches + [line_error]

    anim = animation.FuncAnimation(fig1, update, frames=len(snapshots), blit=False, repeat=True)

    if save_gif:
        print("Сохранение анимации...")
        anim.save('som_training.gif', writer='pillow', fps=5)

    plt.tight_layout()
    plt.show()

    # === 2. Финальная карта с именами ===
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.set_aspect('equal')
    ax2.axis('off')

    if hexagonal:
        for idx in range(len(som.nodes)):
            i = idx // som.ycells
            j = idx % som.ycells
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            color = np.clip(som.nodes[idx].weights / 255.0, 0, 1)
            hexagon = RegularPolygon((x, y), 6, radius=0.5, orientation=np.radians(30),
                                     facecolor=color, edgecolor='black')
            ax2.add_patch(hexagon)
        ax2.set_xlim(-1, som.xcells + 1)
        ax2.set_ylim(-1, som.ycells * np.sqrt(3) / 2 + 1)
    else:
        for idx in range(len(som.nodes)):
            i = idx // som.ycells
            j = idx % som.ycells
            color = np.clip(som.nodes[idx].weights / 255.0, 0, 1)
            rect = plt.Rectangle((i, j), 1, 1, facecolor=color, edgecolor='black')
            ax2.add_patch(rect)
        ax2.set_xlim(0, som.xcells)
        ax2.set_ylim(0, som.ycells)

    for rgb, name in zip(rgb_list, name_list):
        bmu_idx = som._find_bmu(np.array(rgb, dtype=float))
        node = som.nodes[bmu_idx]
        i = bmu_idx // som.ycells
        j = bmu_idx % som.ycells
        if hexagonal:
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
        else:
            x, y = i + 0.5, j + 0.5

        text_color = 'white' if np.mean(node.weights) < 150 else 'black'
        ax2.text(x, y, name, ha='center', va='center', color=text_color,
                 fontsize=5, weight='bold', clip_on=True)

    ax2.invert_yaxis()
    ax2.set_title("Финальная SOM-карта с именами цветов", fontsize=16)
    plt.show()

    # === 3. 3D визуализация весов ===
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111, projection='3d')

    weights_all = np.array([node.weights for node in som.nodes])  # (N*M, 3)
    R, G, B = weights_all[:, 0], weights_all[:, 1], weights_all[:, 2]
    colors_3d = np.clip(weights_all / 255.0, 0, 1)

    ax3.scatter(R, G, B, c=colors_3d, s=50)
    ax3.set_xlabel('Red')
    ax3.set_ylabel('Green')
    ax3.set_zlabel('Blue')
    ax3.set_title('Веса узлов SOM в RGB-пространстве')
    plt.show()

    # === 4. Траектория BMU (на карте) ===
    if som.bmu_history:
        fig4, ax4 = plt.subplots(figsize=(8, 8))
        ax4.set_aspect('equal')
        ax4.axis('off')

        if hexagonal:
            for idx in range(len(som.nodes)):
                i = idx // som.ycells
                j = idx % som.ycells
                x = i + 0.5 * (j % 2)
                y = j * np.sqrt(3) / 2
                color = np.clip(som.nodes[idx].weights / 255.0, 0, 1)
                hexagon = RegularPolygon((x, y), 6, radius=0.5, orientation=np.radians(30),
                                         facecolor=color, edgecolor='lightgray')
                ax4.add_patch(hexagon)
            ax4.set_xlim(-1, som.xcells + 1)
            ax4.set_ylim(-1, som.ycells * np.sqrt(3) / 2 + 1)
        else:
            for idx in range(len(som.nodes)):
                i = idx // som.ycells
                j = idx % som.ycells
                color = np.clip(som.nodes[idx].weights / 255.0, 0, 1)
                rect = plt.Rectangle((i, j), 1, 1, facecolor=color, edgecolor='lightgray')
                ax4.add_patch(rect)
            ax4.set_xlim(0, som.xcells)
            ax4.set_ylim(0, som.ycells)

        # Координаты BMU по истории
        traj_x, traj_y = [], []
        for idx in som.bmu_history[::20]:  # шаг 20 для уменьшения
            i = idx // som.ycells
            j = idx % som.ycells
            if hexagonal:
                x = i + 0.5 * (j % 2)
                y = j * np.sqrt(3) / 2
            else:
                x, y = i + 0.5, j + 0.5
            traj_x.append(x)
            traj_y.append(y)

        ax4.plot(traj_x, traj_y, 'r-', linewidth=1, alpha=0.7, label='Траектория BMU')
        ax4.scatter(traj_x[0], traj_y[0], c='green', s=50, label='Начало')
        ax4.scatter(traj_x[-1], traj_y[-1], c='red', s=50, label='Конец')
        ax4.legend()
        ax4.invert_yaxis()
        ax4.set_title("Траектория активации BMU во время обучения")
        plt.show()

    return anim


if __name__ == "__main__":
    animate_som(hexagonal=True, save_gif=False)