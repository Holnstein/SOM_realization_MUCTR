import os

import train_som
import data_reader
import visualize_training

from config import DATA_FILE_PATH, TRAIN_SIZE


def main():
    csv_file_path = DATA_FILE_PATH

    all_features, all_country_names, feature_columns = data_reader.load_all_data(csv_file_path)

    features_train_raw = all_features[:TRAIN_SIZE]
    country_names_train = all_country_names[:TRAIN_SIZE]

    print(f"Используется {len(country_names_train)} стран для обучения.")

    features_train_normalized, scaler = data_reader.prepare_train_data(features_train_raw, country_names_train)

    os.makedirs("train_data", exist_ok=True)
    data_reader.save_scaler(scaler, "train_data/scaler_train.pkl")

    som, mapping_df, u_matrix, neuron_clusters, feature_names, top10 = train_som.train_and_analyze_som(
        features_train_normalized, country_names_train, feature_columns
    )

    print("Обучение завершено. Результаты сохранены.")


    # print("ШАГ 1: Обучение SOM...")
    # som, mapping_df, u_matrix, neuron_clusters, feature_names, top10 = train_som.train_and_analyze_som(
    #                                                             features_train_raw, country_names_train, feature_names)

    # print("ШАГ 2: Визуализация...")
    visualize_training.visualize_training_results(som, mapping_df, u_matrix, neuron_clusters, feature_names, k=4)

    # print("ШАГ 3: Анализ кластеров...")
    # analyze_clusters(mapping_df)

    # print("ШАГ 4: Сохранение результатов...")
    # save_results(som, mapping_df, u_matrix, neuron_clusters)
    # save_all_results(som, mapping_df, u_matrix, neuron_clusters)

def analyze_clusters(mapping_df):
    print("\n=== РАСПРЕДЕЛЕНИЕ СТРАН ПО КЛАСТЕРАМ ===")
    for cluster_num in sorted(mapping_df['neuron_cluster'].unique()):
        cluster_countries = mapping_df[mapping_df['neuron_cluster'] == cluster_num]['country'].tolist()
        print(f"\nКластер {cluster_num} ({len(cluster_countries)} стран):")
        print(", ".join(cluster_countries[:10]), "..." if len(cluster_countries) > 10 else "")


# def save_all_results(som, mapping_df, u_matrix, neuron_clusters, prefix="som_run"):
#     """Сохраняет все результаты обучения"""
#     os.makedirs("train_data", exist_ok=True)
#
#     np.save(f"train_data/{prefix}_weights.npy", som.weights)
#     np.save(f"train_data/{prefix}_u_matrix.npy", u_matrix)
#     np.save(f"train_data/{prefix}_neuron_clusters.npy", neuron_clusters)
#     mapping_df.to_csv(f"train_data/{prefix}_mapping.csv", index=False)
#     print("Результаты обучения сохранены")

# def load_all_results(som, prefix="som_run"):
#     som.weights = np.load(f"train_data/{prefix}_weights.npy")  # загружаем веса обратно в SOM объект
#     u_matrix = np.load(f"train_data/{prefix}_u_matrix.npy")
#     neuron_clusters = np.load(f"train_data/{prefix}_neuron_clusters.npy")
#     mapping_df = pd.read_csv(f"train_data/{prefix}_mapping.csv")
#     print("Все результаты загружены!")
#     return mapping_df, u_matrix, neuron_clusters

if __name__ == "__main__":
    main()
