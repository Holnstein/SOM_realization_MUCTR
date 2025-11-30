# config.py
# Параметры для SOM
from sys import prefix

GRID_SIZE = (12, 12)  # Размер сетки нейронов
INPUT_DIM = 5         # Размерность входных признаков (у вас 5 критериев)

# Параметры данных
TRAIN_SIZE = 120      # Количество стран для обучения

EPOCHS = 1500

# Пути к файлам
DATA_FILE_PATH = 'data-1762357600736.csv'
TRAIN_DATA_DIR = 'train_data'

# Названия файлов
WEIGHTS_FILE = 'train_som_weights.npy'
U_MATRIX_FILE = 'train_som_u_matrix.npy'
NEURON_CLUSTERS_FILE = 'train_som_neuron_clusters.npy'
TRAIN_MAPPING_FILE = 'train_som_mapping.csv'
SCALER_FILE = 'scaler_train.pkl'
TEST_MAPPING_FILE = 'som_test_mapping.csv'

# Параметры кластеризации
N_CLUSTERS = 4