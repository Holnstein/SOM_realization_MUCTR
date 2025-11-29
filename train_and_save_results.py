import os

import train_som
import visualize_som
import pandas as pd
import numpy as np

def main():
    print("ШАГ 1: Обучение SOM...")
    som, mapping_df, u_matrix, neuron_clusters, feature_names, top10 = train_som.train_and_analyze_som()

    print("ШАГ 2: Визуализация...")
    visualize_som.create_visualizations(som, mapping_df, u_matrix, neuron_clusters, feature_names)

    # print("ШАГ 3: Анализ кластеров...")
    # analyze_clusters(mapping_df)

    print("ШАГ 4: Сохранение результатов...")
    # save_results(som, mapping_df, u_matrix, neuron_clusters)
    save_all_results(som, mapping_df, u_matrix, neuron_clusters)

def analyze_clusters(mapping_df):
    print("\n=== РАСПРЕДЕЛЕНИЕ СТРАН ПО КЛАСТЕРАМ ===")
    for cluster_num in sorted(mapping_df['neuron_cluster'].unique()):
        cluster_countries = mapping_df[mapping_df['neuron_cluster'] == cluster_num]['country'].tolist()
        print(f"\nКластер {cluster_num} ({len(cluster_countries)} стран):")
        print(", ".join(cluster_countries[:10]), "..." if len(cluster_countries) > 10 else "")


def save_all_results(som, mapping_df, u_matrix, neuron_clusters, prefix="som_run"):
    """Сохраняет все результаты обучения"""
    os.makedirs("train_data", exist_ok=True)

    np.save(f"train_data/{prefix}_weights.npy", som.weights)
    np.save(f"train_data/{prefix}_u_matrix.npy", u_matrix)
    np.save(f"train_data/{prefix}_neuron_clusters.npy", neuron_clusters)
    mapping_df.to_csv(f"train_data/{prefix}_mapping.csv", index=False)
    print("Все результаты сохранены!")

def load_all_results(som, prefix="som_run"):
    som.weights = np.load(f"train_data/{prefix}_weights.npy")  # загружаем веса обратно в SOM объект
    u_matrix = np.load(f"train_data/{prefix}_u_matrix.npy")
    neuron_clusters = np.load(f"train_data/{prefix}_neuron_clusters.npy")
    mapping_df = pd.read_csv(f"train_data/{prefix}_mapping.csv")
    print("Все результаты загружены!")
    return mapping_df, u_matrix, neuron_clusters

if __name__ == "__main__":
    main()
