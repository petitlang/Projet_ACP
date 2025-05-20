import pandas as pd

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

df_reponses.to_csv('YuefanLIU_MouzhengLI_LianghongLI.csv', index=False)

# Explication :
# La variance mesure la dispersion des valeurs autour de la moyenne pour chaque variable.
# Une variance élevée indique que les valeurs sont très dispersées, tandis qu'une faible variance indique que les valeurs sont proches de la moyenne.
# Cela permet de comparer la variabilité des températures, des précipitations et de l'ensoleillement entre les villes.
