import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import matplotlib.dates as mdates


CSV_FILE_NAME = 'clima_vix.csv' # ARQUIVO CSV VAI AQUI.
TARGET_COLUMN = 'Potencial_Alergia' # Nome da coluna que vamos criar

# Foco em temperatura, umidade, vento?
# FONTES: https://www.cdc.gov/climate-health/php/effects/allergens-and-pollen.html
# https://www.telfast.com/en-au/understanding-and-managing-allergies/allergy-triggers/climate-change-and-allergies
COLUNAS_PARA_ALERGIA = [
    "Temp. Ins. (C)",
    "Umi. Ins. (%)",
    "Vel. Vento (m/s)",
    "Pressao Ins. (hPa)",
    "Chuva (mm)"
]

# Processamento de dados:
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    new_columns = []
    for col in df.columns:
        clean_col = col.replace('.', '').replace('(', '').replace(')', '').replace('%', '').replace('/', '_').replace('²', '').replace(' ', '_')
        clean_col = '_'.join(filter(None, clean_col.split('_')))
        if clean_col.endswith('_'):
            clean_col = clean_col[:-1]
        new_columns.append(clean_col)
    df.columns = new_columns

    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')

    df['Timestamp'] = pd.to_datetime(df['Data'].dt.strftime('%Y-%m-%d') + ' ' + df['Hora_UTC'].astype(str).apply(lambda x: f"{int(x):04d}"), format='%Y-%m-%d %H%M', errors='coerce')

    for col in df.columns:
        if col in ['Data', 'Hora_UTC', 'Timestamp']:
            continue
        if df[col].dtype == 'object': # Se for string (objeto)
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce') # errors='coerce' transforma erros em NaN


    global COLUNAS_PARA_ALERGIA 
    COLUNAS_PARA_ALERGIA = [
        "Temp_Ins_C",         
        "Umi_Ins",        
        "Vel_Vento_m_s",
        "Pressao_Ins_hPa",
        "Chuva_mm"
    ]

    for col in COLUNAS_PARA_ALERGIA:
        if col in df.columns:
            if df[col].isnull().any():
                print(f"Tratando NaNs na coluna: {col}")
                df[col] = df[col].fillna(df[col].mean())
        else:
            print(f"Aviso: Coluna '{col}' não encontrada no DataFrame após renomeamento.")

    # Ordenar por data e hora
    df = df.sort_values(by=['Data', 'Hora_UTC']).reset_index(drop=True)
    print("Colunas no DataFrame após renomeação:")
    print(df.columns.tolist())

    return df

def create_synthetic_allergy_index(df):

    scaler = MinMaxScaler()
    df_scaled = df[COLUNAS_PARA_ALERGIA].copy()

    df_scaled[COLUNAS_PARA_ALERGIA] = scaler.fit_transform(df_scaled[COLUNAS_PARA_ALERGIA])

    # Chutando os pesos com base nas fontes. Será que está bom assim? 
    df['Indice_Alergia_Sintetico'] = (
        (df_scaled["Vel_Vento_m_s"] * 0.4) +        # Vento: forte espalha pólen
        (df_scaled["Umi_Ins"] * 0.3) +           # Umidade: alta favorece ácaros/mofo
        (df_scaled["Temp_Ins_C"] * 0.1) -         # Temperatura: moderada (perto de 0.5 após escala) pode ser pior que extremos
        (df_scaled["Chuva_mm"] * 0.2) +            # Chuva: tende a diminuir alergia
        (df_scaled["Pressao_Ins_hPa"] * 0.05)    # Pressão: menos impacto, mas mudanças podem afetar
    )

    df['Indice_Alergia_Sintetico'] = MinMaxScaler().fit_transform(df[['Indice_Alergia_Sintetico']])

    # Categorizar o índice em Baixo, Médio, Alto
    df[TARGET_COLUMN] = pd.cut(
        df['Indice_Alergia_Sintetico'],
        bins=[-0.01, 0.33, 0.66, 1.01], # Bins para Low, Medium, High
        labels=['Baixo', 'Medio', 'Alto'],
        right=True
    )
    return df

# função principal:
def run_allergy_prediction(file_path):
    print("--- 1. Carregando e Pré-processando Dados ---")
    df = load_and_preprocess_data(file_path)
    print("DataFrame original (primeiras 5 linhas):")
    print(df.head())
    print("\nInformações do DataFrame:")
    df.info()

    print("\n--- 2. Criando Índice de Alergia Sintético ---")
    df = create_synthetic_allergy_index(df)
    print("\nDataFrame com 'Potencial_Alergia' (primeiras 5 linhas):")
    print(df[['Data', 'Hora_UTC', 'Temp_Ins_C', 'Umi_Ins', 'Vel_Vento_m_s', 'Indice_Alergia_Sintetico', TARGET_COLUMN]].head())
    print("\nContagem de valores da categoria de Alergia:")
    print(df[TARGET_COLUMN].value_counts())


    features = COLUNAS_PARA_ALERGIA 
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df[TARGET_COLUMN]

    y = y.map({'Baixo': 0, 'Medio': 1, 'Alto': 2})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nDados divididos: Treino={len(X_train)} amostras, Teste={len(X_test)} amostras.")

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    print("\nFeatures escalonadas.")

    print("\n--- 4.1. Treinando o Modelo de Machine Learning: Regressão Logística ---")
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_train_scaled, y_train)
    print("Regressão Logística treinada com sucesso!")

    y_pred_lr = model_lr.predict(X_test_scaled)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    class_names = ['Baixo', 'Medio', 'Alto']

    print(f"\n--- 5.1. Avaliação do Modelo: Regressão Logística ---")
    print(f"Acurácia da Regressão Logística: {accuracy_lr:.4f}")
    print("\nRelatório de Classificação (Regressão Logística):")
    print(classification_report(y_test, y_pred_lr, target_names=class_names))

    print("\n--- 4.2. Treinando o Modelo de Machine Learning: Random Forest Classifier ---")

    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train_scaled, y_train)
    print("Random Forest Classifier treinado com sucesso!")

    y_pred_rf = model_rf.predict(X_test_scaled)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    print(f"\n--- 5.2. Avaliação do Modelo: Random Forest Classifier ---")
    print(f"Acurácia do Random Forest Classifier: {accuracy_rf:.4f}")
    print("\nRelatório de Classificação (Random Forest Classifier):")
    print(classification_report(y_test, y_pred_rf, target_names=class_names))

    print("\n--- 6. Comparação Final dos Modelos ---")
    print(f"Acurácia Regressão Logística: {accuracy_lr:.4f}")
    print(f"Acurácia Random Forest Classifier: {accuracy_rf:.4f}")

    if accuracy_rf > accuracy_lr:
        print("\nO Random Forest Classifier apresentou melhor acurácia no conjunto de teste.")
    elif accuracy_lr > accuracy_rf:
        print("\nA Regressão Logística apresentou melhor acurácia no conjunto de teste.")
    else:
        print("\nAmbos os modelos apresentaram a mesma acurácia no conjunto de teste.")

    print("\n--- 7. Gerando Visualizações ---")
    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'], df['Indice_Alergia_Sintetico'], label='Índice de Alergia Sintético')
    plt.axhline(y=0.33, color='r', linestyle='--', label='Limite Médio/Baixo')
    plt.axhline(y=0.66, color='g', linestyle='--', label='Limite Alto/Médio')
    plt.xlabel('Data')
    plt.ylabel('Índice de Alergia')
    plt.title('Variação do Índice de Alergia Sintético ao Longo do Tempo')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(ticker.AutoLocator())
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


    print("Sucesso")

run_allergy_prediction(CSV_FILE_NAME)
