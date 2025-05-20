import pandas as pd
import matplotlib.pyplot as plt
import os

# Vérifier si le fichier de réponses existe et s'il est vide, ajouter l'en-tête si nécessaire
# 检查答案文件是否存在且为空，如果是则写入表头
csv_path = 'YuefanLIU_MouzhengLI_LianghongLI.csv'
if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('Question,Reponse1,Reponse2,Reponse3\n')

# Lire le fichier data1.csv
# 读取 data1.csv 文件
# Afficher les premières lignes pour vérifier l'importation
# 显示前几行数据，确认导入成功
df_data1 = pd.read_csv('data1.csv')
print(df_data1.head())

# 统计城市总数 (q1a)
n_villes = df_data1['Ville'].nunique()

# 统计含有缺失值的城市数 (q1b)
villes_nan = df_data1[df_data1.isnull().any(axis=1)]['Ville'].tolist()
n_villes_nan = len(villes_nan)

# 删除含缺失值的城市
# Supprimer les villes avec des valeurs manquantes
df_data1_clean = df_data1.dropna()

# 将答案写入csv文件
# Écrire les réponses dans le fichier csv
df_reponses = pd.read_csv('YuefanLIU_MouzhengLI_LianghongLI.csv')
df_reponses.loc[df_reponses['Question'] == 'q1a', 'Reponse1'] = n_villes
df_reponses.loc[df_reponses['Question'] == 'q1b', 'Reponse1'] = n_villes_nan

# Trouver la ville et la valeur associée à chaque extrême demandé (q2a-h)
# Pour chaque variable, on cherche la ville avec la valeur minimale et maximale

def get_ville_val(df, col, mode='min'):
    if mode == 'min':
        idx = df[col].idxmin()
    else:
        idx = df[col].idxmax()
    return df.loc[idx, 'Ville'], df.loc[idx, col]

# Température minimale minimale (q2a)
ville_2a, val_2a = get_ville_val(df_data1_clean, 'Temperature_minimale', 'min')
# Température minimale maximale (q2b)
ville_2b, val_2b = get_ville_val(df_data1_clean, 'Temperature_minimale', 'max')
# Température maximale minimale (q2c)
ville_2c, val_2c = get_ville_val(df_data1_clean, 'Temperature_maximale', 'min')
# Température maximale maximale (q2d)
ville_2d, val_2d = get_ville_val(df_data1_clean, 'Temperature_maximale', 'max')
# Précipitations minimales (q2e)
ville_2e, val_2e = get_ville_val(df_data1_clean, 'Hauteur_precipitations', 'min')
# Précipitations maximales (q2f)
ville_2f, val_2f = get_ville_val(df_data1_clean, 'Hauteur_precipitations', 'max')
# Ensoleillement minimal (q2g)
ville_2g, val_2g = get_ville_val(df_data1_clean, 'Duree_ensoleillement', 'min')
# Ensoleillement maximal (q2h)
ville_2h, val_2h = get_ville_val(df_data1_clean, 'Duree_ensoleillement', 'max')

# Écrire les résultats dans le fichier de réponses
# Les colonnes Reponse1 pour la ville, Reponse2 pour la valeur
reps = [
    ('q2a', ville_2a, val_2a),
    ('q2b', ville_2b, val_2b),
    ('q2c', ville_2c, val_2c),
    ('q2d', ville_2d, val_2d),
    ('q2e', ville_2e, val_2e),
    ('q2f', ville_2f, val_2f),
    ('q2g', ville_2g, val_2g),
    ('q2h', ville_2h, val_2h),
]
for q, ville, val in reps:
    df_reponses.loc[df_reponses['Question'] == q, 'Reponse1'] = ville
    df_reponses.loc[df_reponses['Question'] == q, 'Reponse2'] = val

# Calculer la variance pour chaque variable demandée (q3a-d)
# Utiliser la méthode .var() de pandas (par défaut, ddof=1 pour l'échantillon)
var_temp_min = df_data1_clean['Temperature_minimale'].var()
var_temp_max = df_data1_clean['Temperature_maximale'].var()
var_precip = df_data1_clean['Hauteur_precipitations'].var()
var_ensoleil = df_data1_clean['Duree_ensoleillement'].var()

# Écrire les résultats dans le fichier de réponses
# Les colonnes Reponse1 pour la variance
var_reps = [
    ('q3a', var_temp_min),
    ('q3b', var_temp_max),
    ('q3c', var_precip),
    ('q3d', var_ensoleil),
]
for q, var in var_reps:
    df_reponses.loc[df_reponses['Question'] == q, 'Reponse1'] = var

# Calculer la moyenne, la médiane et l'écart-type de la température minimale (q4a, q4b, q4c)
# Moyenne
moyenne_temp_min = df_data1_clean['Temperature_minimale'].mean()
# Médiane
mediane_temp_min = df_data1_clean['Temperature_minimale'].median()
# Écart-type
std_temp_min = df_data1_clean['Temperature_minimale'].std()

# Écrire les résultats dans le fichier de réponses
stat_reps = [
    ('q4a', moyenne_temp_min),
    ('q4b', mediane_temp_min),
    ('q4c', std_temp_min),
]
for q, val in stat_reps:
    df_reponses.loc[df_reponses['Question'] == q, 'Reponse1'] = val

# Question 5
# Calculer la moyenne, la médiane et l'écart-type de la variable de variance maximale (Duree_ensoleillement, q3d)
# Moyenne
moyenne_ensoleil = df_data1_clean['Duree_ensoleillement'].mean()
# Médiane
mediane_ensoleil = df_data1_clean['Duree_ensoleillement'].median()
# Écart-type
std_ensoleil = df_data1_clean['Duree_ensoleillement'].std()

# Écrire les résultats dans le fichier de réponses
stat_reps_5 = [
    ('q5a', moyenne_ensoleil),
    ('q5b', mediane_ensoleil),
    ('q5c', std_ensoleil),
]
for q, val in stat_reps_5:
    df_reponses.loc[df_reponses['Question'] == q, 'Reponse1'] = val

df_reponses.to_csv('YuefanLIU_MouzhengLI_LianghongLI.csv', index=False)

# Explication :
# La variance mesure la dispersion des valeurs autour de la moyenne pour chaque variable.
# Une variance élevée indique que les valeurs sont très dispersées, tandis qu'une faible variance indique que les valeurs sont proches de la moyenne.
# Cela permet de comparer la variabilité des températures, des précipitations et de l'ensoleillement entre les villes.

# Question 4 image
# Afficher l'histogramme de la température minimale
plt.figure(figsize=(8,5))
plt.hist(df_data1_clean['Temperature_minimale'], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogramme de la température minimale')
plt.xlabel('Température minimale (°C)')
plt.ylabel('Nombre de villes')
plt.grid(axis='y', alpha=0.5)
plt.show()

# Explication :
# On calcule la moyenne, la médiane et l'écart-type pour résumer la distribution de la température minimale.
# L'histogramme permet de visualiser la répartition des températures minimales parmi les villes.
# Si la moyenne et la médiane sont proches, la distribution est plutôt symétrique. Un écart-type faible indique peu de dispersion.

# Question 5 image
# Afficher l'histogramme de la durée d'ensoleillement
plt.figure(figsize=(8,5))
plt.hist(df_data1_clean['Duree_ensoleillement'], bins=10, color='orange', edgecolor='black')
plt.title("Histogramme de la durée d'ensoleillement")
plt.xlabel("Durée d'ensoleillement (heures)")
plt.ylabel('Nombre de villes')
plt.grid(axis='y', alpha=0.5)
plt.show()

# Explication :
# On calcule la moyenne, la médiane et l'écart-type pour la durée d'ensoleillement, qui est la variable la plus dispersée selon la variance.
# L'histogramme permet de visualiser la répartition de l'ensoleillement parmi les villes.
# Si la moyenne et la médiane sont proches, la distribution est plutôt symétrique. Un écart-type élevé indique une grande dispersion.
