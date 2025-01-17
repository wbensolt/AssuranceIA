#Projet Prime Assurance

# Data_cleaning_

## Overview
This first script performs data cleaning and preprocessing on an insurance dataset. The objective is to prepare the data for further analysis and machine learning modeling, particularly for predicting insurance charges. The script:

1. Loads the dataset.
2. Handles duplicates and outliers.
3. Adds new features through transformations and interactions.
4. Saves the cleaned and enriched dataset for future use.

---

## Input
- **Dataset**: `data/4072eb5e-e963-4a17-a794-3ea028d0a9c4.csv`  
  This file contains raw insurance data with the following columns:
  - `age`
  - `sex`
  - `bmi`
  - `children`
  - `smoker`
  - `region`
  - `charges`

---

## Steps Performed

### 1. Loading the Data
The dataset is loaded into a pandas DataFrame, and its structure is inspected:
```python
df = pd.read_csv("data/4072eb5e-e963-4a17-a794-3ea028d0a9c4.csv")
df.info()
```

### 2. Handling Duplicates
- Identifies duplicate rows:
  ```python
  duplicates_rows = df[df.duplicated()]
  print(duplicates_rows)
  ```
- Removes duplicate rows.

### 3. Removing Outliers
- Specific rows identified as outliers are removed by their indices:
  ```python
  lignes = [806, 917, 957, 1146, 964, 806, 876, 1303, 39, 1080]
  df = df.drop(index=lignes, errors="ignore")
  ```

### 4. Feature Engineering
Several new features are created to enrich the dataset:

#### Log Transformations
- `log_charges`: Logarithm of insurance charges.
- `log_age`: Logarithm of age.
- `log_children`: Logarithm of the number of children.
  ```python
  df['log_charges'] = np.log1p(df['charges'])
  df['log_age'] = np.log(df['age'] + 1)
  df['log_children'] = np.log(df['children'] + 1)
  ```

#### Age Group Categorization
Age is categorized into discrete groups:
- Bins: `[0, 28, 51, 65, np.inf]`
- Labels: `['Jeune', 'Mature', 'Âgé', 'Senior']`
  ```python
  bins_age = [0, 28, 51, 65, np.inf]
  labels_age = ['Jeune', 'Mature', 'Âgé', 'Senior']
  df['age_group'] = pd.cut(df['age'], bins=bins_age, labels=labels_age)
  ```

#### BMI Categorization
BMI is categorized as:
- Bins: `[0, 18, 30, 40, np.inf]`
- Labels: `['Maigre', 'Normal', 'Surpoids', 'Obèse']`
  ```python
  bins_bmi = [0, 18, 30, 40, np.inf]
  labels_bmi = ['Maigre', 'Normal', 'Surpoids', 'Obèse']
  df['bmi_category'] = pd.cut(df['bmi'], bins=bins_bmi, labels=labels_bmi)
  ```

#### Encodings and Interactions
- `smoker_encoded`: Binary encoding of smoking status (`yes` -> 1, `no` -> 0).
- Interaction terms:
  - `bmi_smoker`: Interaction of BMI and smoking status.
  - `age_smoker`: Interaction of age and smoking status.
  - `age_bmi`: Interaction of age and BMI.
  ```python
  df['smoker_encoded'] = df['smoker'].map({'yes': 1, 'no': 0})
  df['bmi_smoker'] = df['bmi'] * df['smoker_encoded']
  df['age_smoker'] = df['age'] * df['smoker_encoded']
  df['age_bmi'] = df['age'] * df['bmi']
  ```

### 5. Selecting Relevant Columns
The dataset is reduced to the most relevant features for modeling:
```python
df = df[["age", "sex", "bmi", "children", "smoker", "region", "age_bmi", "age_group", "bmi_category", "bmi_smoker", "age_smoker", "charges"]]
```

### 6. Saving the Cleaned Data
The cleaned dataset is saved as a CSV file:
```python
df.to_csv("models/df_assurance_clean_with_log_v4_with_new_val_7_remove_OUTLIERS.csv", index=False)
```

---

## Output
- **Cleaned Dataset**: `models/df_assurance_clean_with_log_v4_with_new_val_7_remove_OUTLIERS.csv`  
  This file contains the cleaned and enriched data ready for modeling.

---

## Requirements
- Python libraries:
  - `pandas`
  - `numpy`

Install dependencies with:
```bash
pip install pandas numpy
```

---

## Usage
1. Place the raw dataset in the `data/` directory.
2. Run the script:
   ```bash
   python data_cleaning.py
   ```
3. The cleaned dataset will be saved in the `models/` directory.




# Modélisation avec Scikit-learn

Ce projet implémente un pipeline de modélisation machine learning en utilisant Scikit-learn. L'objectif principal est de construire et d'évaluer des modèles de régression pour estimer les charges d'assurance à partir d'un jeu de données nettoyé.

---

## Structure du Script

### 1. Préparation des Données
La fonction `prepare_data` divise les colonnes du DataFrame en deux catégories :
- **Colonnes numériques** : stockées dans `df_numeric`.
- **Colonnes catégorielles** : stockées dans `df_category`.

### 2. Transformation des Données
La fonction `transform_data` applique un encodage One-Hot aux colonnes catégorielles :
- Les colonnes catégorielles sélectionnées (ex. `region`, `age_group`, `bmi_category`) sont entièrement encodées.
- Les autres colonnes catégorielles utilisent un encodage binaire.

### 3. Normalisation des Données
La fonction `normalize_data` propose trois méthodes de normalisation :
- **Code 0** : Z-score (manuelle).
- **Code 1** : Utilisation de `StandardScaler`.
- **Code 2** : Utilisation de `MinMaxScaler`.

### 4. Modélisation
La fonction `model_data` entraîne et évalue plusieurs modèles :
- Dummy Regressor (baseline).
- Régression linéaire simple.
- Modèles régularisés (Lasso, Ridge, ElasticNet).

Les performances sont évaluées avec le coefficient de détermination \( R^2 \).

### 5. Recherche de Performances
- Génération de combinaisons de colonnes pour identifier les sous-ensembles de données optimaux.
- Évaluation de la performance \( R^2 \) pour chaque combinaison et normalisation.

---

## Fichiers Générés

### Résultats
Les performances des modèles sont sauvegardées dans un fichier CSV nommé dynamiquement en fonction de la date et de l'heure :
- Exemple : `results_RL_Norm1_2025-01-17_14-30-00.csv`.

### Modèle Enregistré
Le modèle de régression linéaire est sauvegardé dans un fichier pickle :
- `linear_regression_model_1.pkl`.

---

## Dépendances

Assurez-vous d'installer les requirements.txt