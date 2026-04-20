import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_reader
import visualize_test

from SOM import SOM
import os

from config import DATA_FILE_PATH, TRAIN_SIZE, GRID_SIZE, INPUT_DIM


def load_all_results(prefix=""):
    """Загружает все сохраненные результаты"""
    try:
        # som = SOM(grid_size=(7, 7), input_dim=5)
        # som.weights = np.load(f"train_data/{prefix}_weights.npy")

        u_matrix = np.load(f"train_data/{prefix}_u_matrix.npy")
        neuron_clusters = np.load(f"train_data/{prefix}_neuron_clusters.npy")
        train_mapping_df = pd.read_csv(f"train_data/{prefix}_mapping.csv")

        print("Результаты обучения загружены")
        return train_mapping_df, u_matrix, neuron_clusters
    except FileNotFoundError as e:
        print(f"Ошибка загрузки результатов обучения: {e}")
        print("Сначала запустите train_and_save_results.py")
        return None, None, None


def create_custom_visualization(mapping_df, neuron_clusters, cluster_names):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Карта кластеров нейронов
    im = ax1.imshow(neuron_clusters, cmap='tab10', interpolation='nearest')
    ax1.set_title('Карта кластеров нейронов', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()

    # 2. Столбчатая диаграмма
    cluster_counts = mapping_df['neuron_cluster'].value_counts().sort_index()
    labels = [cluster_names[i] for i in cluster_counts.index]

    bars = ax2.bar(labels, cluster_counts.values, color=plt.cm.tab10(range(len(cluster_counts))))
    ax2.set_title('Распределение стран по кластерам', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Количество стран')
    ax2.tick_params(axis='x', rotation=45)

    for bar, count in zip(bars, cluster_counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Выводим детальную информацию
    print("\n" + "=" * 60)
    print("КЛАСТЕРЫ")
    print("=" * 60)
    for cluster_num, name in cluster_names.items():
        countries = mapping_df[mapping_df['neuron_cluster'] == cluster_num]['country'].tolist()
        print(f"\n🏷️  {name} ({len(countries)} стран):")
        print("   " + ", ".join(countries))


def main():
    csv_file_path = DATA_FILE_PATH
    prefix = "train_som"

    # Загружаем сохраненные данные
    train_mapping_df, u_matrix, neuron_clusters = load_all_results(prefix)
    if train_mapping_df is None:
        return

    all_features, all_country_names, feature_columns = data_reader.load_all_data(csv_file_path)
    print(f"Всего стран в датасете: {len(all_country_names)}")

    test_start_idx = TRAIN_SIZE
    test_end_idx = len(all_country_names)

    if test_start_idx >= len(all_country_names):
        print("Ошибка: Обучающая выборка больше или равна всему датасету.")
        return

    # Обучающие данные
    features_train_raw = all_features[:TRAIN_SIZE]
    country_names_train = all_country_names[:TRAIN_SIZE]

    # Тестовые данные
    features_test_raw = all_features[test_start_idx:test_end_idx]
    country_names_test = all_country_names[test_start_idx:test_end_idx]

    print(f"Обучающая выборка: {len(country_names_train)} стран (индексы 0-{TRAIN_SIZE - 1})")
    print(f"Тестовая выборка: {len(country_names_test)} стран (индексы {test_start_idx}-{test_end_idx - 1})")

    scaler_path = "train_data/scaler_train.pkl"
    if not os.path.exists(scaler_path):
        print(f"Ошибка: Scaler не найден по пути {scaler_path}. Обучите модель заново.")
        return
    scaler = data_reader.load_scaler(scaler_path)

    features_test_normalized = data_reader.prepare_test_data(features_test_raw, scaler)

    weights_path = f"train_data/{prefix}_weights.npy"
    if not os.path.exists(weights_path):
        print(f"Ошибка: Веса SOM не найдены по пути {weights_path}. Обучите модель заново.")
        return
    loaded_weights = np.load(weights_path)

    som = SOM(grid_size=GRID_SIZE, input_dim=INPUT_DIM)
    som.weights = loaded_weights  # Загрузка обученных весов
    print("Обученная SOM загружена.")

    print("\nТестирование SOM на новых данных")
    test_bmus = []
    test_distances = []
    for i, test_sample in enumerate(features_test_normalized):
        bmu_coords, min_dist = som.find_bmu(test_sample)
        test_bmus.append(bmu_coords)
        test_distances.append(min_dist)
        # print(f"Тестовый пример {i} ({country_names_test[i]}): BMU={bmu_coords}, dist={min_dist:.4f}")

    avg_test_distance = np.mean(test_distances)
    print(f"Среднее расстояние до BMU на ТЕСТОВОЙ выборке: {avg_test_distance:.4f}")

    test_mapping_df = pd.DataFrame({
        'country': country_names_test,
        'bmu_i': [bmu[0] for bmu in test_bmus],
        'bmu_j': [bmu[1] for bmu in test_bmus],
        'distance_to_bmu': test_distances
    })

    test_mapping_df['neuron_cluster'] = test_mapping_df.apply(
        lambda row: neuron_clusters[row['bmu_i'], row['bmu_j']], axis=1
    )

    test_mapping_df.to_csv(f"train_data/som_test_mapping.csv", index=False)
    print(f"Результаты теста сохранены в train_data/{prefix}_mapping.csv")

    mapping_df = train_mapping_df

    # КАКИЕ КЛАСТЕРЫ БЫЛИ СОХРАНЕНЫ
    print("\nСОХРАНЕННЫЕ КЛАСТЕРЫ:")
    for cluster_num in sorted(mapping_df['neuron_cluster'].unique()):
        countries = mapping_df[mapping_df['neuron_cluster'] == cluster_num]['country'].tolist()
        print(f"Кластер {cluster_num}: {', '.join(countries[:5])}...")

    # 3. ВВОДИМ НАЗВАНИЯ
    print("\n" + "=" * 50)

    my_cluster_names = {
        2: "2",
        1: "1",
        0: "0",
        3: "3"
    }

    print("Названия кластеров:")
    for cluster_num, name in my_cluster_names.items():
        print(f"  Кластер {cluster_num} → '{name}'")

    # 4. Создаем визуализацию с названиями кластеров
    create_custom_visualization(mapping_df, neuron_clusters, my_cluster_names)

    print("\n--- Распределение ТЕСТОВЫХ стран по кластерам ---")
    test_cluster_counts = test_mapping_df['neuron_cluster'].value_counts().sort_index()
    print(test_cluster_counts)

    print("\n--- Сравнение распределений ---")
    print("Обучающие:")
    print(train_mapping_df['neuron_cluster'].value_counts().sort_index())
    print("\nТестовые:")
    print(test_cluster_counts)

    visualize_test.visualize_test_results(u_matrix, neuron_clusters, train_mapping_df, test_mapping_df)

if __name__ == "__main__":
    main()