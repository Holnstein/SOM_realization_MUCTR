import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_reader
from SOM import SOM

def load_all_results(prefix="som_run"):
    """Загружает все сохраненные результаты"""
    try:
        som = SOM(grid_size=(10, 10), input_dim=5)
        som.weights = np.load(f"train_data/{prefix}_weights.npy")

        u_matrix = np.load(f"train_data/{prefix}_u_matrix.npy")
        neuron_clusters = np.load(f"train_data/{prefix}_neuron_clusters.npy")
        mapping_df = pd.read_csv(f"train_data/{prefix}_mapping.csv")

        print("Все результаты загружены!")
        return mapping_df, u_matrix, neuron_clusters
    except FileNotFoundError:
        print("Сначала запустите train_and_save_results.py")
        return None, None, None


def create_custom_visualization(mapping_df, neuron_clusters, cluster_names):
    """Создает кастомную визуализацию с вашими названиями"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Карта кластеров нейронов
    im = ax1.imshow(neuron_clusters, cmap='tab10', interpolation='nearest')
    ax1.set_title('Карта кластеров нейронов', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()

    # 2. Столбчатая диаграмма с вашими названиями
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
    print("КЛАСТЕРЫ С ВАШИМИ НАЗВАНИЯМИ")
    print("=" * 60)
    for cluster_num, name in cluster_names.items():
        countries = mapping_df[mapping_df['neuron_cluster'] == cluster_num]['country'].tolist()
        print(f"\n🏷️  {name} ({len(countries)} стран):")
        print("   " + ", ".join(countries))


def main():
    # 1. Загружаем сохраненные данные
    mapping_df, u_matrix, neuron_clusters = load_all_results("som_run")
    if mapping_df is None:
        return

    # 2. ПОКАЗЫВАЕМ КАКИЕ КЛАСТЕРЫ БЫЛИ СОХРАНЕНЫ
    print("\nСОХРАНЕННЫЕ КЛАСТЕРЫ:")
    for cluster_num in sorted(mapping_df['neuron_cluster'].unique()):
        countries = mapping_df[mapping_df['neuron_cluster'] == cluster_num]['country'].tolist()
        print(f"Кластер {cluster_num}: {', '.join(countries[:5])}...")

    # 3. ВВОДИМ СВОИ НАЗВАНИЯ
    print("\n" + "=" * 50)
    print("ВВЕДИТЕ ВАШИ НАЗВАНИЯ КЛАСТЕРОВ")
    print("=" * 50)

    # МЕНЯЙТЕ ЭТИ НАЗВАНИЯ НА ОСНОВЕ ТОГО, ЧТО ВИДИТЕ ВЫШЕ!
    my_cluster_names = {
        0: "Страны 2 мира",
        1: "Бедные страны",
        2: "Страны 2 мира",
        3: "Развитые страны"
    }

    print("Ваши названия:")
    for cluster_num, name in my_cluster_names.items():
        print(f"  Кластер {cluster_num} → '{name}'")

    # 4. Создаем визуализацию с вашими названиями
    create_custom_visualization(mapping_df, neuron_clusters, my_cluster_names)

if __name__ == "__main__":
    main()