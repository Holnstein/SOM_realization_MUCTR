import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def prepare_data(csv_file_path):
    df = pd.read_csv(csv_file_path, sep=';')

    feature_columns = ['security_index', 'life_expectancy', 'co2_emissions',
                       'urban_population_in_slums', 'renewable_energy']

    # feature_columns = ['life_expectancy', 'urban_population']

    features = df[feature_columns].values
    country_names = df['country_name'].values

    df.dropna(inplace=True)


    # print(f"Количество строк после очистки: {len(df)}")
    # print(f"Удалено строк: {len(df) - len(df)}")

    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)

    # print(features_normalized)

    print(f"Подготовлено данных: {len(features_normalized)} стран")
    print(f"Размерность: {features_normalized.shape}")
    print(f"Диапазоны признаков после нормализации:")
    # for i, col in enumerate(feature_columns):
    #     print(f"  {col}: [{features_normalized[:, i].min():.2f}, {features_normalized[:, i].max():.2f}]")

    return features_normalized, country_names, feature_columns