import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import SOM
import data_reader
from sklearn.cluster import KMeans

def main():
    data_for_som, country_names, feature_names = data_reader.prepare_data('data-1762357600736.csv')

    som = SOM.SOM(grid_size=(10, 10), input_dim=data_for_som.shape[1])

    som.train(data=data_for_som, epochs=100)

    u_matrix = som.calculate_u_matrix()

    rows = []
    for idx, country in enumerate(country_names):
        bmu, dist = som.find_bmu(data_for_som[idx])
        rows.append({'country': country, 'bmu_i': bmu[0], 'bmu_j': bmu[1], 'distance': dist})

    mapping_df = pd.DataFrame(rows)

    weights = som.weights.reshape((-1, som.input_dim))
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=0).fit(weights)
    neuron_clusters = kmeans.labels_.reshape(som.grid_size)

    def coord_to_index(i, j):
        return i * som.grid_size[1] + j

    mapping_df['neuron_index'] = mapping_df.apply(lambda r: coord_to_index(r['bmu_i'], r['bmu_j']), axis=1)
    mapping_df['neuron_cluster'] = mapping_df['neuron_index'].map(lambda idx: int(kmeans.labels_[idx]))

    mapping_df_sorted = mapping_df.sort_values(['bmu_i', 'bmu_j', 'distance']).reset_index(drop=True)
    occupied = mapping_df.groupby(['bmu_i', 'bmu_j']).size().reset_index(name='count').sort_values('count',
                                                                                                   ascending=False)
    top10 = occupied.head(10)

    print("Топ-10 нейронов по числу стран (i, j, count):")
    print(top10.to_string(index=False))

    # U-matrix with counts
    plt.figure(figsize=(8, 6))
    plt.imshow(u_matrix, cmap='hot', interpolation='nearest')
    plt.title('U-Matrix (with counts overlay)')
    plt.colorbar(label='Distance')
    for _, row in occupied.iterrows():
        i, j, count = int(row['bmu_i']), int(row['bmu_j']), int(row['count'])
        plt.text(j, i, str(count), color='white', ha='center', va='center', fontsize=9, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.show()

    # neuron clusters map
    plt.figure(figsize=(6, 5))
    plt.imshow(neuron_clusters, cmap='tab10', interpolation='nearest')
    plt.title(f'Кластеры нейронов (kmeans k={k})')
    plt.colorbar(ticks=range(k))
    plt.gca().invert_yaxis()
    plt.show()

    # component planes
    n_features = len(feature_names)
    cols = 3
    rows_pl = (n_features + cols - 1) // cols
    fig, axes = plt.subplots(rows_pl, cols, figsize=(12, 4 * rows_pl))
    axes = axes.ravel()
    for idx, fname in enumerate(feature_names):
        comp = som.weights[:, :, idx]
        im = axes[idx].imshow(comp, interpolation='nearest')
        axes[idx].set_title(fname)
        axes[idx].invert_yaxis()
        fig.colorbar(im, ax=axes[idx])
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()

    # examples in top5 neurons
    print("\nПримеры стран в 10 самых заполненных нейронах:")
    examples = []
    for _, r in top10.iterrows():
        i, j = int(r['bmu_i']), int(r['bmu_j'])
        subset = mapping_df[(mapping_df.bmu_i == i) & (mapping_df.bmu_j == j)]['country'].tolist()
        examples.append({'neuron': (i, j), 'count': len(subset), 'countries_sample': subset[:15]})
    examples_df = pd.DataFrame(examples)
    print(examples_df.to_string(index=False))

    # Покажем первые 30 строк mapping_df_sorted
    display_df = mapping_df_sorted.head(30).copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df[['country', 'bmu_i', 'bmu_j', 'distance', 'neuron_cluster']]

    display_df.to_csv('mapping.csv', index=False)

    for cl in range(4):
        subset = mapping_df[mapping_df['neuron_cluster'] == cl]['country'].tolist()
        print(f"\nКластер {cl} ({len(subset)} стран):")
        for c in subset:
            print("  ", c)

    def save_all_results(som, mapping_df, u_matrix, neuron_clusters, prefix="som_run"):
        np.save(f"{prefix}_weights.npy", som.weights)
        np.save(f"{prefix}_u_matrix.npy", u_matrix)
        np.save(f"{prefix}_neuron_clusters.npy", neuron_clusters)
        mapping_df.to_csv(f"{prefix}_mapping.csv", index=False)
        print("Все результаты сохранены!")

    def load_all_results(som, prefix="som_run"):
        som.weights = np.load(f"{prefix}_weights.npy")  # загружаем веса обратно в SOM объект
        u_matrix = np.load(f"{prefix}_u_matrix.npy")
        neuron_clusters = np.load(f"{prefix}_neuron_clusters.npy")
        mapping_df = pd.read_csv(f"{prefix}_mapping.csv")
        print("Все результаты загружены!")
        return mapping_df, u_matrix, neuron_clusters



if __name__ == "__main__":
    main()

