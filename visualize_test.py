from matplotlib import pyplot as plt
import numpy as np

def visualize_test_results(u_matrix, neuron_clusters, train_mapping_df, test_mapping_df):
    overlay_test_on_umatrix(u_matrix, train_mapping_df, test_mapping_df)
    overlay_test_on_clusters(neuron_clusters, train_mapping_df, test_mapping_df)
    compare_train_test_cluster_distributions(train_mapping_df, test_mapping_df)

def overlay_test_on_umatrix(u_matrix, train_mapping_df, test_mapping_df):
    plt.figure(figsize=(8, 6))
    plt.imshow(u_matrix, cmap='hot', interpolation='nearest', alpha=0.7) # alpha - прозрачность
    plt.title('U-Matrix с наложенными тестовыми странами')
    plt.colorbar(label='Distance')

    # Отображение обучающих данных
    occupied_train = train_mapping_df.groupby(['bmu_i', 'bmu_j']).size().reset_index(name='count')
    for _, row in occupied_train.iterrows():
        i, j, count = int(row['bmu_i']), int(row['bmu_j']), int(row['count'])
        plt.text(j, i, str(count), color='white', ha='center', va='center', fontsize=8, fontweight='bold')

    # Отображение тестовых данных
    for _, row in test_mapping_df.iterrows():
        i, j = row['bmu_i'], row['bmu_j']
        plt.plot(j, i, 'bo', markersize=5)

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("graphics/u-matrix_test_overlay.png", dpi=300, bbox_inches='tight')
    plt.close()

def overlay_test_on_clusters(neuron_clusters, train_mapping_df, test_mapping_df):
    plt.figure(figsize=(8, 6))
    plt.imshow(neuron_clusters, cmap='tab10', interpolation='nearest', alpha=0.7)
    plt.title('Карта кластеров с наложенными тестовыми странами')
    plt.colorbar(ticks=range(4)) # Предполагаем k=4

    # Отображение обучающих данных
    occupied_train = train_mapping_df.groupby(['bmu_i', 'bmu_j']).size().reset_index(name='count')
    for _, row in occupied_train.iterrows():
        i, j, count = int(row['bmu_i']), int(row['bmu_j']), int(row['count'])
        plt.text(j, i, str(count), color='black', ha='center', va='center', fontsize=8, fontweight='bold')

    # Отображение тестовых данных
    for _, row in test_mapping_df.iterrows():
        i, j = row['bmu_i'], row['bmu_j']
        plt.plot(j, i, 'ro', markersize=5)

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("graphics/neuron_clusters_test_overlay.png", dpi=300, bbox_inches='tight')
    plt.close()

def compare_train_test_cluster_distributions(train_mapping_df, test_mapping_df):
    train_counts = train_mapping_df['neuron_cluster'].value_counts().sort_index()
    test_counts = test_mapping_df['neuron_cluster'].value_counts().sort_index()

    clusters_all = set(train_counts.index).union(set(test_counts.index))
    clusters_sorted = sorted(list(clusters_all))

    train_vals = [train_counts.get(c, 0) for c in clusters_sorted]
    test_vals = [test_counts.get(c, 0) for c in clusters_sorted]

    x = np.arange(len(clusters_sorted))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, train_vals, width, label='Обучающие', alpha=0.8)
    ax.bar(x + width/2, test_vals, width, label='Тестовые', alpha=0.8)

    ax.set_xlabel('Номер кластера')
    ax.set_ylabel('Количество стран')
    ax.set_title('Сравнение распределения стран по кластерам')
    ax.set_xticks(x)
    ax.legend()

    plt.tight_layout()
    plt.savefig("graphics/cluster_comparison_train_test.png", dpi=300, bbox_inches='tight')
    plt.close()

# ... другие функции для визуализации теста ...