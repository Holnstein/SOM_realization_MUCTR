import numpy as np
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, grid_size, input_dim):
        self.grid_size = grid_size
        self.input_dim = input_dim
        # self.learning_rate = learning_rate
        # self.radius = radius

        self.weights = np.random.rand(grid_size[0], grid_size[1], input_dim)
        # self.weights = np.array([[0.1, 0.1],
                                # [0.9, 0.1]])

        # self.weights = np.array([[[0.1, 0.1], [0.9, 0.1]],
        #                         [[0.1, 0.9], [0.9, 0.9]]])

        print(f"Создана SOM: {grid_size[0]}x{grid_size[1]}, вход: {input_dim}D")
        print(f"Форма weights: {self.weights.shape}")
        print(self.weights)
        self.A = 1
        self.B = 2
        self.learn_rate_coef = 0.5

    def find_bmu(self, input_vector):
        min_dist = np.linalg.norm(input_vector - self.weights[0, 0])
        bmu_coords = (0, 0)
        # print(f"Поиск BMU для {input_vector}:")
        # print(f"  Начальное: нейрон (0,0), dist={min_dist:.4f}")


        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                distance = np.linalg.norm(input_vector - self.weights[i, j])
                # print(f"  Нейрон ({i},{j}): {self.weights[i, j]}, dist={distance:.4f}")
                if min_dist > distance:
                    min_dist = distance
                    bmu_coords = (i, j)
                    # print(f"    -> НОВЫЙ BMU!")

        # print(f"  ФИНАЛЬНЫЙ BMU: {bmu_coords}, dist={min_dist:.4f}")
        return bmu_coords, min_dist

    def neighborhood_function(self, bmu_index, current_radius):
        influence_mtrx = np.zeros(self.grid_size)
        bmu_i, bmu_j = bmu_index

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                distance = np.sqrt((np.square(i - bmu_i) + np.square(j - bmu_j)))

                influence = np.exp(-np.square(distance) / (2 * np.square(current_radius)))

                influence_mtrx[i, j] = influence

        return influence_mtrx

    def update_weights(self, input_vector, bmu_index, radius, learning_rate):

        influence = self.neighborhood_function(bmu_index, radius)
        # print(influence)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                weight_delta = learning_rate * influence[i, j] * (input_vector - self.weights[i, j])

                self.weights[i, j] += weight_delta

    def calc_error(self, data):
        total_error = 0

        for sample in data:
            _, distance = self.find_bmu(sample)
            total_error += distance

        total_error /= len(data)
        return total_error


    def train(self, data, epochs):
        errors = []
        initial_radius = max(self.grid_size[0], self.grid_size[1]) / 2
        initial_learning_rate = 0.03
        time_const = epochs / np.log(initial_radius)

        for epoch in range(epochs):
            # current_lr = self.A/(epoch + self.B)
            current_lr = 0.1 * (1 - epoch/epochs)
            current_lr = initial_learning_rate * np.exp(-epoch / epochs)
            current_radius = initial_radius * (1 - epoch/epochs)
            current_radius = initial_radius * np.exp(-epoch / time_const)
            indices = np.random.permutation(len(data))
            for i in indices:
                sample = data[i]
                bmu, _ = self.find_bmu(sample)

                self.update_weights(sample, bmu, current_radius, current_lr)
            errors.append(self.calc_error(data))
        print(errors[-1])
        self.plot_learning_curve(errors)

    def plot_learning_curve(self, errors):
        plt.plot(errors)
        plt.title('Кривая обучения SOM')
        plt.xlabel('Эпоха')
        plt.ylabel('Error')
        plt.grid(True)
        plt.savefig("graphics/error.png", dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_u_matrix(self):
        """Рассчитывает U-Matrix - расстояния между соседними нейронами"""
        rows, cols = self.grid_size
        u_matrix = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                distances = []
                # Проверяем всех соседей
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            dist = np.linalg.norm(self.weights[i, j] - self.weights[ni, nj])
                            distances.append(dist)

                # Усредняем расстояния до соседей
                u_matrix[i, j] = np.mean(distances) if distances else 0

        return u_matrix

    # def plot_u_matrix(self):
    #     """Визуализирует U-Matrix"""
    #     u_matrix = self.calculate_u_matrix()
    #
    #     plt.figure(figsize=(10, 8))
    #     plt.imshow(u_matrix, cmap='hot', interpolation='nearest')
    #     plt.colorbar(label='Расстояние между нейронами')
    #     plt.title('U-Matrix SOM')
    #     plt.xlabel('X координата нейрона')
    #     plt.ylabel('Y координата нейрона')
    #
    #     # Добавляем сетку для лучшей читаемости
    #     for i in range(self.grid_size[0] + 1):
    #         plt.axhline(i - 0.5, color='white', linewidth=0.5)
    #     for j in range(self.grid_size[1] + 1):
    #         plt.axvline(j - 0.5, color='white', linewidth=0.5)
    #
    #     plt.tight_layout()
    #     plt.show()
    #
    #     return u_matrix

    # def plot_component_planes(self, feature_names):
    #     """Визуализирует карты компонентов для каждого признака"""
    #     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    #     axes = axes.ravel()
    #
    #     for idx, feature_name in enumerate(feature_names):
    #         # Извлекаем веса для конкретного признака
    #         component_plane = self.weights[:, :, idx]
    #
    #         im = axes[idx].imshow(component_plane, cmap='viridis', interpolation='hanning')
    #         axes[idx].set_title(f'Признак: {feature_name}')
    #         axes[idx].set_xlabel('X нейрона')
    #         axes[idx].set_ylabel('Y нейрона')
    #         plt.colorbar(im, ax=axes[idx])
    #
    #     # Скрываем лишние subplots
    #     for idx in range(len(feature_names), len(axes)):
    #         axes[idx].set_visible(False)
    #
    #     plt.tight_layout()
    #     plt.show()