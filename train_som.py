import os

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from SOM import SOM
from config import GRID_SIZE, INPUT_DIM, EPOCHS


def train_and_analyze_som(data_for_som, country_names, feature_names):
    # инициализация структуры сети som
    som = SOM(grid_size=GRID_SIZE, input_dim=INPUT_DIM)
    # обучение som
    som.train(data=data_for_som, epochs=EPOCHS)

    # Расчет U-Matrix
    u_matrix = som.calculate_u_matrix()

    # сопоставление стран и нейронов
    rows = []
    for idx, country in enumerate(country_names):
        bmu, dist = som.find_bmu(data_for_som[idx])
        rows.append({'country': country, 'bmu_i': bmu[0], 'bmu_j': bmu[1], 'distance': dist})

    mapping_df = pd.DataFrame(rows)

    # Кластеризация нейронов
    weights = som.weights.reshape((-1, som.input_dim))
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42).fit(weights)
    neuron_clusters = kmeans.labels_.reshape(som.grid_size)

    # Добавляем кластеры к данным
    def coord_to_index(i, j):
        return i * som.grid_size[1] + j

    mapping_df['neuron_index'] = mapping_df.apply(lambda r: coord_to_index(r['bmu_i'], r['bmu_j']), axis=1)
    mapping_df['neuron_cluster'] = mapping_df['neuron_index'].map(lambda idx: int(kmeans.labels_[idx]))

    # Анализ заполненности нейронов
    mapping_df_sorted = mapping_df.sort_values(['bmu_i', 'bmu_j', 'distance']).reset_index(drop=True)
    occupied = mapping_df.groupby(['bmu_i', 'bmu_j']).size().reset_index(name='count').sort_values('count',
                                                                                                   ascending=False)
    top10 = occupied.head(10)

    print("Топ-10 нейронов по числу стран (i, j, count):")
    print(top10.to_string(index=False))

    save_all_results(som, mapping_df, u_matrix, neuron_clusters, prefix="train_som")

    # examples in top5 neurons
    # print("\nПримеры стран в 10 самых заполненных нейронах:")
    # examples = []
    # for _, r in top10.iterrows():
    #     i, j = int(r['bmu_i']), int(r['bmu_j'])
    #     subset = mapping_df[(mapping_df.bmu_i == i) & (mapping_df.bmu_j == j)]['country'].tolist()
    #     examples.append({'neuron': (i, j), 'count': len(subset), 'countries_sample': subset[:15]})
    # examples_df = pd.DataFrame(examples)
    # print(examples_df.to_string(index=False))

    return som, mapping_df, u_matrix, neuron_clusters, feature_names, top10

def save_all_results(som, mapping_df, u_matrix, neuron_clusters, prefix="som_run"):
    """Сохраняет все результаты обучения"""
    os.makedirs("train_data", exist_ok=True)

    np.save(f"train_data/{prefix}_weights.npy", som.weights)
    np.save(f"train_data/{prefix}_u_matrix.npy", u_matrix)
    np.save(f"train_data/{prefix}_neuron_clusters.npy", neuron_clusters)
    mapping_df.to_csv(f"train_data/{prefix}_mapping.csv", index=False)
    print("Результаты обучения сохранены")

if __name__ == "__main__":
    som, mapping_df, u_matrix, neuron_clusters, feature_names, top10 = train_and_analyze_som()
