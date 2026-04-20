import pickle

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_all_data(csv_file_path):
    df = pd.read_csv(csv_file_path, sep=';')

    feature_columns = ['security_index', 'life_expectancy', 'co2_emissions',
                       'urban_population_in_slums', 'renewable_energy']

    # df.dropna(inplace=True)
    features = df[feature_columns].values
    country_names = df['country_name'].values

    return features, country_names, feature_columns

def prepare_train_data(features_train_raw, country_names_train):
    """Нормализует обучающие данные и возвращает нормализованные данные и scaler."""
    scaler = MinMaxScaler()
    features_train_normalized = scaler.fit_transform(features_train_raw)
    print(f"Подготовлено обучающих данных: {len(features_train_normalized)} стран")
    return features_train_normalized, scaler

def prepare_test_data(features_test_raw, scaler):
    """Применяет scaler к тестовым данным."""
    features_test_normalized = scaler.transform(features_test_raw)
    print(f"Подготовлено тестовых данных: {len(features_test_normalized)} стран")
    return features_test_normalized

def save_scaler(scaler, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler сохранен в {filepath}")

def load_scaler(filepath):
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler загружен из {filepath}")
    return scaler