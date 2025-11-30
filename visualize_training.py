from matplotlib import pyplot as plt
import numpy as np

def visualize_training_results(som, train_mapping_df, u_matrix, neuron_clusters, feature_names, k):

    create_u_matrix_visualization(u_matrix, train_mapping_df, title_suffix="(Обучающие)")
    create_neuron_clusters_visualization(neuron_clusters, k, title_suffix="(Обучающие)")
    create_component_planes(som, feature_names)
    create_additional_visualizations(train_mapping_df)
    save_all_results(som, train_mapping_df, u_matrix, neuron_clusters)
    create_train_cluster_distribution(train_mapping_df)

def create_u_matrix_visualization(u_matrix, mapping_df, title_suffix=""):
    plt.figure(figsize=(8, 6))
    plt.imshow(u_matrix, cmap='hot', interpolation='nearest')
    plt.title(f'U-Matrix {title_suffix}')
    plt.colorbar(label='Distance')

    # Добавляем количество стран в каждый нейрон
    occupied = mapping_df.groupby(['bmu_i', 'bmu_j']).size().reset_index(name='count')

    for _, row in occupied.iterrows():
        i, j, count = int(row['bmu_i']), int(row['bmu_j']), int(row['count'])
        plt.text(j, i, str(count), color='white', ha='center', va='center', fontsize=8, fontweight='bold')

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("graphics/u-matrix.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

def create_neuron_clusters_visualization(neuron_clusters, k=4, title_suffix=""):
    plt.figure(figsize=(8, 6))
    plt.imshow(neuron_clusters, cmap='tab10', interpolation='nearest')
    plt.title(f'Кластеры нейронов {title_suffix}')
    plt.colorbar(ticks=range(k))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("graphics/neuron_clusters.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

def create_component_planes(som, feature_names):
    n_features = len(feature_names)
    cols = 3
    rows = (n_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if rows > 1:
        axes = axes.ravel()
    else:
        axes = [axes] if cols == 1 else axes

    # axes = axes.ravel()

    for idx, fname in enumerate(feature_names):
        comp = som.weights[:, :, idx]
        im = axes[idx].imshow(comp, interpolation='nearest', cmap='viridis')
        axes[idx].set_title(fname)
        axes[idx].invert_yaxis()
        fig.colorbar(im, ax=axes[idx])

    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig("graphics/component_planes.png", dpi=300)
    # plt.show()
    plt.close()

    pass

def create_additional_visualizations(mapping_df):
    """Дополнительные визуализации"""
    # Распределение стран по кластерам
    plt.figure(figsize=(10, 6))
    cluster_counts = mapping_df['neuron_cluster'].value_counts().sort_index()
    plt.bar(cluster_counts.index, cluster_counts.values)
    plt.title('Количество стран в каждом кластере')
    plt.xlabel('Номер кластера')
    plt.ylabel('Количество стран')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("graphics/cluster_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def save_all_results(som, mapping_df, u_matrix, neuron_clusters, prefix="som_run"):
    np.save(f"train_data/{prefix}_weights.npy", som.weights)
    np.save(f"train_data/{prefix}_u_matrix.npy", u_matrix)
    np.save(f"train_data/{prefix}_neuron_clusters.npy", neuron_clusters)
    mapping_df.to_csv(f"train_data/{prefix}_mapping.csv", index=False)
    print("Все результаты сохранены!")

def create_train_cluster_distribution(mapping_df):
    plt.figure(figsize=(10, 6))
    cluster_counts = mapping_df['neuron_cluster'].value_counts().sort_index()
    plt.bar(cluster_counts.index, cluster_counts.values)
    plt.title('Количество стран в каждом кластере (Обучающие)')
    plt.xlabel('Номер кластера')
    plt.ylabel('Количество стран')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("graphics/cluster_distribution_train.png", dpi=300, bbox_inches='tight')
    plt.close()