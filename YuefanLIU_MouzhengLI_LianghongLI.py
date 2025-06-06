import os
import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy import stats
from scipy.special import comb  # Importer la fonction comb pour les combinaisons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Question 0 : Initialisation du fichier de réponses
# 初始化答案文件
csv_path = 'YuefanLIU_MouzhengLI_LianghongLI.csv'
if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
    # Créer le DataFrame avec tous les numéros de questions si le fichier n'existe pas ou est vide
    # 如果文件不存在或为空，创建包含所有题号的DataFrame
    data = {
        'Question': [
            'q1a', 'q1b',
            'q2a', 'q2b', 'q2c', 'q2d', 'q2e', 'q2f', 'q2g', 'q2h',
            'q3a', 'q3b', 'q3c', 'q3d',
            'q4a', 'q4b', 'q4c',
            'q5a', 'q5b', 'q5c',
            'q6a', 'q6b', 'q6c',
            'q8a', 'q8b',
            'q12a', 'q12b', 'q12c', 'q12d', 'q12e',
            'q13a', 'q13b',
            'q14a', 'q14b',
            'q16a', 'q16b',
            'q17'
        ],
        'Reponse1': [''] * 37,
        'Reponse2': [''] * 37,
        'Reponse3': [''] * 37
    }
    df_reponses = pd.DataFrame(data)
    df_reponses.to_csv(csv_path, index=False)
else:
    # Si le fichier existe déjà, le lire simplement
    # 如果文件已存在，直接读取
    df_reponses = pd.read_csv(csv_path)

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
df_data1_clean = df_data1.dropna().copy()  # 添加.copy()创建副本

# 将答案写入csv文件
# Écrire les réponses dans le fichier csv
df_reponses = pd.read_csv('YuefanLIU_MouzhengLI_LianghongLI.csv')
df_reponses.loc[df_reponses['Question'] == 'q1a', 'Reponse1'] = n_villes
df_reponses.loc[df_reponses['Question'] == 'q1b', 'Reponse1'] = n_villes_nan

# 打印回答
print("\n[q1a] Nombre total de villes dans le dataset :", n_villes)
print("[q1b] Nombre de villes avec des valeurs manquantes :", n_villes_nan)
if n_villes_nan > 0:
    print("   Villes concernées :", villes_nan)

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

# 打印回答
print("\n[q2a] Ville avec la température minimale la plus basse :", ville_2a, f"({val_2a}°C)")
print("[q2b] Ville avec la température minimale la plus haute :", ville_2b, f"({val_2b}°C)")
print("[q2c] Ville avec la température maximale la plus basse :", ville_2c, f"({val_2c}°C)")
print("[q2d] Ville avec la température maximale la plus haute :", ville_2d, f"({val_2d}°C)")
print("[q2e] Ville avec les précipitations les plus faibles :", ville_2e, f"({val_2e}mm)")
print("[q2f] Ville avec les précipitations les plus élevées :", ville_2f, f"({val_2f}mm)")
print("[q2g] Ville avec l'ensoleillement le plus faible :", ville_2g, f"({val_2g}heures)")
print("[q2h] Ville avec l'ensoleillement le plus élevé :", ville_2h, f"({val_2h}heures)")

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

# 打印回答
print("\n[q3a] Variance de la température minimale :", f"{var_temp_min:.4f}")
print("[q3b] Variance de la température maximale :", f"{var_temp_max:.4f}")
print("[q3c] Variance des précipitations :", f"{var_precip:.4f}")
print("[q3d] Variance de l'ensoleillement :", f"{var_ensoleil:.4f}")

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

# 打印回答
print("\n[q4a] Moyenne de la température minimale :", f"{moyenne_temp_min:.2f}°C")
print("[q4b] Médiane de la température minimale :", f"{mediane_temp_min:.2f}°C")
print("[q4c] Écart-type de la température minimale :", f"{std_temp_min:.2f}°C")

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

# 打印回答
print("\n[q5a] Moyenne de la durée d'ensoleillement :", f"{moyenne_ensoleil:.2f} heures")
print("[q5b] Médiane de la durée d'ensoleillement :", f"{mediane_ensoleil:.2f} heures")
print("[q5c] Écart-type de la durée d'ensoleillement :", f"{std_ensoleil:.2f} heures")

# Question 6
# Calculer la matrice de corrélation pour les variables météorologiques
cols = ['Temperature_minimale', 'Temperature_maximale', 'Hauteur_precipitations', 'Duree_ensoleillement']
corr_matrix = df_data1_clean[cols].corr()

# Chercher les deux variables les plus positivement corrélées (hors diagonale)
corr_pairs = corr_matrix.unstack()
corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
corr_pairs = corr_pairs.drop_duplicates()

# Plus positivement corrélées (q6a)
max_corr = corr_pairs.idxmax()
max_corr_val = corr_pairs.max()
# Plus négativement corrélées (q6b)
min_corr = corr_pairs.idxmin()
min_corr_val = corr_pairs.min()
# Les moins corrélées (q6c, valeur la plus proche de 0)
min_abs_corr = corr_pairs.abs().idxmin()
min_abs_corr_val = corr_pairs[min_abs_corr]

# Écrire les résultats dans le fichier de réponses
# q6a
q6a_vars = f"{max_corr[0]} & {max_corr[1]}"
df_reponses.loc[df_reponses['Question'] == 'q6a', 'Reponse1'] = q6a_vars
df_reponses.loc[df_reponses['Question'] == 'q6a', 'Reponse2'] = max_corr_val
# q6b
q6b_vars = f"{min_corr[0]} & {min_corr[1]}"
df_reponses.loc[df_reponses['Question'] == 'q6b', 'Reponse1'] = q6b_vars
df_reponses.loc[df_reponses['Question'] == 'q6b', 'Reponse2'] = min_corr_val
# q6c
q6c_vars = f"{min_abs_corr[0]} & {min_abs_corr[1]}"
df_reponses.loc[df_reponses['Question'] == 'q6c', 'Reponse1'] = q6c_vars
df_reponses.loc[df_reponses['Question'] == 'q6c', 'Reponse2'] = min_abs_corr_val

# 打印回答
print("\n[q6a] Variables les plus corrélées positivement :")
print(f"   {q6a_vars} (r = {max_corr_val:.4f})")
print("[q6b] Variables les plus corrélées négativement :")
print(f"   {q6b_vars} (r = {min_corr_val:.4f})")
print("[q6c] Variables les moins corrélées :")
print(f"   {q6c_vars} (r = {min_abs_corr_val:.4f})")

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

# Question 6 image
# Afficher les nuages de points pour chaque paire trouvée, avec le nom des villes
def scatter_with_labels(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(8,5))
    plt.scatter(df_data1_clean[x], df_data1_clean[y], color='green')
    for i, row in df_data1_clean.iterrows():
        plt.text(row[x], row[y], row['Ville'], fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()

# Plus positivement corrélées
scatter_with_labels(max_corr[0], max_corr[1], max_corr[0], max_corr[1], f"Corrélation positive maximale : {q6a_vars} (r={max_corr_val:.2f})")
# Plus négativement corrélées
scatter_with_labels(min_corr[0], min_corr[1], min_corr[0], min_corr[1], f"Corrélation négative maximale : {q6b_vars} (r={min_corr_val:.2f})")
# Les moins corrélées
scatter_with_labels(min_abs_corr[0], min_abs_corr[1], min_abs_corr[0], min_abs_corr[1], f"Corrélation la plus faible : {q6c_vars} (r={min_abs_corr_val:.2f})")

# Explication :
# On calcule la matrice de corrélation pour toutes les variables météorologiques.
# On identifie les deux variables les plus corrélées positivement, négativement et les moins corrélées.
# Les nuages de points permettent de visualiser la relation linéaire entre ces variables, avec le nom de chaque ville affiché.

# Question 7
# Calculer la matrice de corrélation entre les villes (corrélation sur les variables météorologiques)
# On transpose le DataFrame pour avoir les villes en colonnes
villes_corr = df_data1_clean[cols].T.corr()

# Afficher la matrice de corrélation sous forme de heatmap
plt.figure(figsize=(12,10))
sns.heatmap(villes_corr, 
            cmap='coolwarm', 
            annot=True,  # 显示数值
            fmt='.2f',   # 数值格式为两位小数
            xticklabels=df_data1_clean['Ville'], 
            yticklabels=df_data1_clean['Ville'],
            annot_kws={'size': 8})  # 设置标签字体大小
plt.title('Matrice de corrélation entre les villes')
plt.xlabel('Ville')
plt.ylabel('Ville')
plt.tight_layout()
plt.show()

# Explication :
# Cette matrice montre à quel point les profils météorologiques des villes sont similaires.
# Une valeur proche de 1 indique que deux villes ont des conditions météorologiques très similaires sur l'année, tandis qu'une valeur proche de 0 ou négative indique des profils très différents.
# On peut repérer des groupes de villes très corrélées (par exemple, des villes géographiquement proches ou ayant un climat similaire).

# Question 8
# Standardiser les données (centrer et réduire)
X = df_data1_clean[cols].values
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Appliquer l'ACP
pca = PCA()
X_pca = pca.fit_transform(X_std)

# Afficher les deux premières composantes principales
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], color='purple')
for i, ville in enumerate(df_data1_clean['Ville']):
    plt.text(X_pca[i,0], X_pca[i,1], ville, fontsize=8)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('Projection des villes sur les deux premières composantes principales (ACP)')
plt.grid(alpha=0.3)
plt.show()

# Écrire le pourcentage de variance expliquée par PC1 et PC2 dans le fichier de réponses
var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100
df_reponses.loc[df_reponses['Question'] == 'q8a', 'Reponse1'] = var_pc1
df_reponses.loc[df_reponses['Question'] == 'q8b', 'Reponse1'] = var_pc2
df_reponses.to_csv(csv_path, index=False)

# 打印回答
print("\n[q8a] Pourcentage de variance expliquée par PC1 :", f"{var_pc1:.2f}%")
print("[q8b] Pourcentage de variance expliquée par PC2 :", f"{var_pc2:.2f}%")

# Explication :
# On centre et réduit les données, puis on applique une ACP.
# On affiche les villes dans le plan des deux premières composantes principales, qui résument la majorité de la variance des données.
# Les pourcentages de variance expliquée par PC1 et PC2 sont indiqués sur les axes et enregistrés dans le fichier de réponses.

# Question 9
# Afficher le cercle de corrélation de l'ACP
plt.figure(figsize=(7,7))
# Tracer le cercle
circle = plt.Circle((0,0), 1, color='gray', fill=False, linestyle='--')
plt.gca().add_artist(circle)

# Les coordonnées des variables sur les deux premiers axes principaux
for i, var in enumerate(cols):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i],
              color='red', alpha=0.7, head_width=0.05)
    plt.text(pca.components_[0, i]*1.1, pca.components_[1, i]*1.1, var, color='blue', fontsize=12)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Cercle de corrélation (ACP)')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.grid(alpha=0.3)
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.show()

# Explication :
# Le cercle de corrélation permet de visualiser la contribution et la corrélation des variables initiales avec les deux premières composantes principales.
# Plus une flèche est longue et proche du cercle, plus la variable est bien représentée par les deux axes.
# Les variables proches l'une de l'autre sont corrélées positivement, celles en opposition sont corrélées négativement.

# Question 10
# Superposer les résultats de la question 8 (projection des villes) et de la question 9 (cercle de corrélation)
plt.figure(figsize=(8,8))
# Cercle de corrélation
circle = plt.Circle((0,0), 1, color='gray', fill=False, linestyle='--')
plt.gca().add_artist(circle)
# Variables (flèches)
for i, var in enumerate(cols):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i],
              color='red', alpha=0.7, head_width=0.05)
    plt.text(pca.components_[0, i]*1.1, pca.components_[1, i]*1.1, var, color='blue', fontsize=12)
# Villes (points)
plt.scatter(X_pca[:,0], X_pca[:,1], color='purple')
for i, ville in enumerate(df_data1_clean['Ville']):
    plt.text(X_pca[i,0], X_pca[i,1], ville, fontsize=8, color='black')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Projection des villes et cercle de corrélation (ACP)')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.grid(alpha=0.3)
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.show()

# Explication :
# Cette figure combine la projection des villes sur les deux premières composantes principales et le cercle de corrélation.
# On peut visualiser à la fois la position des villes et l'influence des variables météorologiques.
# Les villes extrêmes sur un axe sont généralement celles qui présentent les valeurs extrêmes pour la variable correspondante (voir question 2).
# Ainsi, on peut retrouver les villes associées aux valeurs minimales ou maximales de chaque variable en observant leur position par rapport aux flèches.

## Partie 2 : Régression linéaire simple 

# Question 11 : Afficher l'évolution de la température en 2024 à Paris en fonction du mois
# 读取 data2.csv 文件
# Lecture du fichier data2.csv

df_data2 = pd.read_csv('data2.csv')

# 只保留2024年的数据
# Garder uniquement les données de l'année 2024
paris_2024 = df_data2[df_data2['Annee'] == 2024].copy()  # 创建副本

# 为了保证月份顺序，定义法语月份顺序
# Définir l'ordre des mois en français
mois_fr = ['janvier', 'fevrier', 'mars', 'avril', 'mai', 'juin', 'juillet', 'aout', 'septembre', 'octobre', 'novembre', 'decembre']
paris_2024['Mois'] = pd.Categorical(paris_2024['Mois'], categories=mois_fr, ordered=True)
paris_2024 = paris_2024.sort_values('Mois')

# 绘制温度变化曲线
# Tracer l'évolution de la température maximale
plt.figure(figsize=(10,6))
plt.plot(paris_2024['Mois'], paris_2024['Temperature_maximale'], marker='o', color='blue')
plt.title("Évolution de la température maximale à Paris en 2024")
plt.xlabel("Mois")
plt.ylabel("Température maximale (°C)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 评论：
# On observe que la température maximale à Paris en 2024 augmente progressivement du janvier jusqu'à l'été (juillet-août),
# puis diminue vers la fin de l'année. Ce schéma est typique du climat tempéré, avec des étés chauds et des hivers doux.
# Les pics de température sont atteints en juillet et août, tandis que les températures les plus basses sont en hiver (janvier, décembre). 

# Question 12 : Régression linéaire pour prédire la température maximale en janvier 2025
from sklearn.linear_model import LinearRegression
import numpy as np

# 构造"numéro mois"变量，0表示1月，11表示12月
# Créer la variable 'numéro mois' (0 pour janvier, 11 pour décembre)
paris_2024 = paris_2024.copy()
paris_2024['numero_mois'] = np.arange(0, 12)

# 用于存储每个n的回归结果
# Stocker les résultats pour chaque n
results = []

for n in range(1, 13): # 从1月到12月  
    # 只取最近n个月的数据
    # Prendre les n derniers mois
    data_n = paris_2024.tail(n)
    if len(data_n) < 2:  # 确保至少有2个样本
        continue
    X = data_n['numero_mois'].values.reshape(-1, 1)
    y = data_n['Temperature_maximale'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    # 计算R2
    # Calculer R2
    r2 = model.score(X, y)
    # 计算R2 ajusté
    # Calculer R2 ajusté
    if len(y) > X.shape[1] + 1:  # 确保样本数大于变量数+1
        r2_adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    else:
        r2_adj = np.nan
    results.append({
        'n': n,
        'r2': r2,
        'r2_adj': r2_adj,
        'beta0': model.intercept_,
        'beta1': model.coef_[0],
        'model': model
    })

# 找到R2 ajusté最大的n
# Trouver la valeur optimale de n (R2 ajusté maximal)
results_df = pd.DataFrame(results)
if not results_df.empty and not results_df['r2_adj'].isna().all():
    best_row = results_df.loc[results_df['r2_adj'].idxmax()]
    n_opt = int(best_row['n'])
    r2_opt = best_row['r2']
    r2_adj_opt = best_row['r2_adj']
    beta0_opt = best_row['beta0']
    beta1_opt = best_row['beta1']
    model_opt = best_row['model']
else:
    print("Aucun modèle valide n'a été trouvé")
    n_opt = np.nan
    r2_opt = np.nan
    r2_adj_opt = np.nan
    beta0_opt = np.nan
    beta1_opt = np.nan
    model_opt = None

if model_opt is not None:
    print("\n=== Détails du modèle de régression linéaire simple ===")
    print(f"Nombre de mois utilisés (n) : {n_opt}")
    print(f"R² : {r2_opt:.4f}")
    print(f"R² ajusté : {r2_adj_opt:.4f}")
    print(f"\nCoefficients du modèle :")
    print(f"Constante (β₀) : {beta0_opt:.4f}")
    print(f"Pente (β₁) : {beta1_opt:.4f}")
    
    print("\nÉquation du modèle :")
    print(f"Température = {beta0_opt:.4f} + {beta1_opt:.4f} × Numéro_mois")
    
    # 预测2025年1月（numéro mois=12）的温度
    # Prédire la température maximale pour janvier 2025 (numéro mois=12)
    pred_2025_jan = model_opt.predict(np.array([[12]]))[0]
    print(f"\nPrédiction pour janvier 2025 : {pred_2025_jan:.2f}°C")

    # 可视化：最优n下的拟合曲线
    # Visualisation : courbe de régression pour n optimal
    plt.figure(figsize=(10,6))
    # 画真实点
    plt.scatter(paris_2024['numero_mois'], paris_2024['Temperature_maximale'], color='blue', label='Données réelles')
    # 画拟合点
    X_fit = paris_2024['numero_mois'].values.reshape(-1, 1)
    y_fit = model_opt.predict(X_fit)
    plt.plot(paris_2024['numero_mois'], y_fit, color='red', label='Régression linéaire (n optimal)')
    plt.xlabel('Numéro du mois (0=janvier, 11=décembre)')
    plt.ylabel('Température maximale (°C)')
    plt.title(f"Régression linéaire (n optimal={n_opt}) sur les {n_opt} derniers mois")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 写入csv文件
# Écrire les résultats dans le fichier de réponses pour la question 12
df_reponses.loc[df_reponses['Question'] == 'q12a', 'Reponse1'] = n_opt
df_reponses.loc[df_reponses['Question'] == 'q12b', 'Reponse1'] = r2_adj_opt
df_reponses.loc[df_reponses['Question'] == 'q12c', 'Reponse1'] = r2_opt
df_reponses.loc[df_reponses['Question'] == 'q12d', 'Reponse1'] = beta0_opt
df_reponses.loc[df_reponses['Question'] == 'q12e', 'Reponse1'] = beta1_opt
df_reponses.to_csv('YuefanLIU_MouzhengLI_LianghongLI.csv', index=False)

# 打印回答
print("\n[q12a] Valeur optimale de n :", n_opt)
print("[q12b] R² ajusté optimal :", f"{r2_adj_opt:.4f}")
print("[q12c] R² optimal :", f"{r2_opt:.4f}")
print("[q12d] Coefficient β0 :", f"{beta0_opt:.4f}")
print("[q12e] Coefficient β1 :", f"{beta1_opt:.4f}")

# Question 13 : Comparaison entre la température réelle et prédite pour janvier 2025
# La température réelle en janvier 2025 était de 7,5°C
# La température prédite a déjà été calculée : pred_2025_jan

temp_reelle_2025_jan = 7.5
# 计算差值
# Calculer l'écart
écart = pred_2025_jan - temp_reelle_2025_jan
print(f"Température réelle (janvier 2025) : {temp_reelle_2025_jan} °C")
print(f"Écart (prédite - réelle) : {écart:.2f} °C")

# 写入csv文件
# Écrire les résultats dans le fichier de réponses
# q13a : température prédite, q13b : écart
if 'df_reponses' not in locals():
    df_reponses = pd.read_csv('YuefanLIU_MouzhengLI_LianghongLI.csv')
df_reponses.loc[df_reponses['Question'] == 'q13a', 'Reponse1'] = pred_2025_jan
df_reponses.loc[df_reponses['Question'] == 'q13b', 'Reponse1'] = écart
df_reponses.to_csv('YuefanLIU_MouzhengLI_LianghongLI.csv', index=False)

# 打印回答
print("\n[q13a] Température maximale prédite pour janvier 2025 :", f"{pred_2025_jan:.2f}°C")
print("[q13b] Écart entre la température prédite et réelle :", f"{écart:.2f}°C")

# Question 14 : Test d'hypothèse pour le coefficient β1 du modèle optimal

# 取最优n下的数据
# Prendre les données utilisées pour le modèle optimal
X_opt = paris_2024['numero_mois'].values[-n_opt:].reshape(-1, 1)
y_opt = paris_2024['Temperature_maximale'].values[-n_opt:]

# 预测值与残差
# Valeurs prédites et résidus
y_pred_opt = model_opt.predict(X_opt)
residuals = y_opt - y_pred_opt
n = n_opt
p = 1  # 只有一个自变量

# 计算标准误差
# Calcul de l'erreur standard de β1
SSE = np.sum(residuals**2)
mean_x = np.mean(X_opt)
Sxx = np.sum((X_opt.flatten() - mean_x)**2)
se_beta1 = np.sqrt(SSE / (n - 2)) / np.sqrt(Sxx)

# t统计量
# Statistique t
beta1 = beta1_opt
t_stat = beta1 / se_beta1
# p值（双尾）
# p-value (bilatérale)
p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-2))

print(f"p-value pour le test de β1 : {p_value:.4g}")

# α=5%下的结论
# Conclusion pour α=5%
alpha = 0.05
if p_value < alpha:
    conclusion = "Oui, la pente est significative : il existe une relation linéaire."
else:
    conclusion = "Non, la pente n'est pas significative : pas de relation linéaire."
print(f"Conclusion (α=5%) : {conclusion}")

# 写入csv文件
# Écrire les résultats dans le fichier de réponses
# q14a : p-value, q14b : conclusion
df_reponses.loc[df_reponses['Question'] == 'q14a', 'Reponse1'] = p_value
df_reponses.loc[df_reponses['Question'] == 'q14b', 'Reponse1'] = conclusion
df_reponses.to_csv('YuefanLIU_MouzhengLI_LianghongLI.csv', index=False)

# 打印回答
print("\n[q14a] p-value pour le test de β1 :", f"{p_value:.4g}")
print("[q14b] Conclusion (α=5%) :", conclusion)

# Question 15 : Superposer l'évolution de température en 2023 et 2024 à Paris
# 在同一张图上画出2023年和2024年每月温度曲线
# Tracer sur la même courbe l'évolution de la température maximale à Paris en 2023 et 2024

data_2023 = df_data2[df_data2['Annee'] == 2023].copy()
data_2024 = df_data2[df_data2['Annee'] == 2024].copy()

# 保证月份顺序一致
# S'assurer que l'ordre des mois est correct
mois_fr = ['janvier', 'fevrier', 'mars', 'avril', 'mai', 'juin', 'juillet', 'aout', 'septembre', 'octobre', 'novembre', 'decembre']
data_2023['Mois'] = pd.Categorical(data_2023['Mois'], categories=mois_fr, ordered=True)
data_2024['Mois'] = pd.Categorical(data_2024['Mois'], categories=mois_fr, ordered=True)
data_2023 = data_2023.sort_values('Mois')
data_2024 = data_2024.sort_values('Mois')

plt.figure(figsize=(10,6))
plt.plot(data_2023['Mois'], data_2023['Temperature_maximale'], marker='o', color='orange', label='2023')
plt.plot(data_2024['Mois'], data_2024['Temperature_maximale'], marker='o', color='blue', label='2024')
plt.title("Évolution de la température maximale à Paris en 2023 et 2024")
plt.xlabel("Mois")
plt.ylabel("Température maximale (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 评论：
# On observe que les deux années présentent une évolution saisonnière similaire :
# la température augmente du début de l'année jusqu'à l'été (juillet-août), puis diminue en automne et hiver.
# Cependant, on peut remarquer des différences de niveau ou de pics entre les deux années, ce qui peut être dû à la variabilité climatique interannuelle.

# Question 16 : Sélection de variables pour la régression multivariée
from itertools import combinations

# 构造滞后特征矩阵
# Construction des variables de température des mois précédents (lags)
max_lag = 12
# 合并2023和2024年的数据
T = np.concatenate([data_2023['Temperature_maximale'].values, data_2024['Temperature_maximale'].values])
X_lags = []
y = []
# 从2024年1月开始建立模型
start_idx = 12  # 2024年的1月
for i in range(start_idx, len(T)):
    X_lags.append([T[i-j-1] for j in range(max_lag)])
    y.append(T[i])
X_lags = np.array(X_lags).reshape(-1, max_lag)  # S'assurer que X_lags est bien une matrice 2D
y = np.array(y)

best_vars = None  # Initialiser best_vars pour éviter NameError
# 检查数据量是否足够
# Vérifier si on a assez de données
if X_lags.shape[0] == 0:
    print("Pas assez de données pour la régression multivariée avec les lags choisis.")
else:
    # 所有组合数（不含空集）
    # Nombre de combinaisons possibles (hors ensemble vide)
    comb_count = sum([int(comb(max_lag, k)) for k in range(1, max_lag+1)])
    print(f"Nombre de combinaisons possibles : {comb_count}")

    # 记录最优结果
    # Stocker le meilleur résultat
    best_r2_adj = -np.inf
    best_vars = None
    best_model = None
    best_pvalues = None
    best_coefs = None

    for k in range(1, max_lag+1):
        for idxs in combinations(range(max_lag), k):
            X_sel = X_lags[:, idxs]
            if X_sel.shape[0] <= X_sel.shape[1] + 1:  # 确保样本量大于变量数+1
                continue
            model = LinearRegression().fit(X_sel, y)
            y_pred = model.predict(X_sel)
            r2 = model.score(X_sel, y)
            n = len(y)
            p = X_sel.shape[1]
            
            # 安全计算调整后的R²
            if n > p + 1:
                r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            else:
                continue
                
            # 安全计算标准误差
            residuals = y - y_pred
            SSE = np.sum(residuals**2)
            if n > p + 1 and SSE > 0:
                se = np.sqrt(SSE / (n - p - 1))
                X_design = np.hstack([np.ones((n,1)), X_sel])
                try:
                    cov = np.linalg.inv(X_design.T @ X_design)
                    se_betas = np.sqrt(np.diag(cov)) * se
                    t_stats = np.hstack([model.intercept_, model.coef_]) / se_betas
                    pvals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-p-1))
                    
                    # 只考虑所有变量p值<0.05的情况
                    if r2_adj > best_r2_adj and np.all(pvals[1:] < 0.05):
                        best_r2_adj = r2_adj
                        best_vars = idxs
                        best_model = model
                        best_pvalues = pvals
                        best_coefs = np.hstack([model.intercept_, model.coef_])
                except np.linalg.LinAlgError:
                    continue
            else:
                continue

    # 输出最优结果
    if best_vars is not None:
        print("\n=== Détails du meilleur modèle ===")
        print(f"R² ajusté : {best_r2_adj:.4f}")
        print(f"Nombre de variables sélectionnées : {len(best_vars)}")
        print("\nDétails des variables :")
        print("Nom de variable\t\tCoefficient\t\tp-valeur")
        print("-" * 60)
        print(f"Constante\t\t{best_coefs[0]:.4f}\t\t{best_pvalues[0]:.4f}")
        for i, (var_idx, coef, pval) in enumerate(zip(best_vars, best_coefs[1:], best_pvalues[1:])):
            print(f"Temp_{var_idx+1}\t\t{coef:.4f}\t\t{pval:.4f}")
        
        print("\nÉquation du modèle :")
        equation = f"Température = {best_coefs[0]:.4f}"
        for i, (var_idx, coef) in enumerate(zip(best_vars, best_coefs[1:])):
            equation += f" + {coef:.4f} × Temp_{var_idx+1}"
        print(equation)
        
        if np.all(best_pvalues[1:] < 0.05):
            conclusion = "Oui, il existe une relation linéaire significative (α=5%)."
        else:
            conclusion = "Non, pas de relation linéaire significative (α=5%)."
        print(f"\nConclusion : {conclusion}")
    else:
        print("Aucune combinaison n'a toutes les p-values < 0.05.")
        conclusion = "Non, pas de relation linéaire significative (α=5%)."

    # 写入csv
    # Écrire les résultats dans le fichier de réponses
    df_reponses.loc[df_reponses['Question'] == 'q16a', 'Reponse1'] = comb_count if X_lags.shape[0] > 0 else 0
    df_reponses.loc[df_reponses['Question'] == 'q16b', 'Reponse1'] = len(best_vars) if best_vars is not None else 0
    df_reponses.loc[df_reponses['Question'] == 'q16b', 'Reponse2'] = best_r2_adj if best_vars is not None else ''
    df_reponses.loc[df_reponses['Question'] == 'q16b', 'Reponse3'] = str([f'Temp_{i+1}' for i in best_vars]) if best_vars is not None else ''
    df_reponses.to_csv('YuefanLIU_MouzhengLI_LianghongLI.csv', index=False)

    # 打印回答
    print("\n[q16a] Nombre total de combinaisons possibles :", comb_count if X_lags.shape[0] > 0 else 0)
    if best_vars is not None:
        print("[q16b] Modèle optimal :")
        print(f"   Nombre de variables : {len(best_vars)}")
        print(f"   R² ajusté : {best_r2_adj:.4f}")
        print(f"   Variables sélectionnées : {[f'Temp_{i+1}' for i in best_vars]}")
        print(f"   Conclusion : {conclusion}")
    else:
        print("[q16b] Aucun modèle avec toutes les variables significatives n'a été trouvé")
        print(f"   Conclusion : {conclusion}")

# Question 17 : Prédiction pour janvier-avril 2025 avec le modèle multivarié optimal
# 真实温度
# Températures réelles
T_reelles = [8.0, 10.0, 14.0, 18.0]
mois_pred = ['janvier', 'février', 'mars', 'avril']

# 用2024年最后max_lag个月温度递推预测2025年1-4月
# Prédire les températures maximales pour janvier à avril 2025
T_2024 = list(data_2024['Temperature_maximale'].values)
T_predites = []
T_input = T_2024.copy()
for i in range(4):
    if best_vars is not None:
        X_pred = np.array([T_input[-(j+1)] for j in best_vars]).reshape(1, -1)
        T_pred = best_model.predict(X_pred)[0]
        T_predites.append(T_pred)
        T_input.append(T_pred)
    else:
        T_predites.append(np.nan)

# 计算与真实值的差值
# Calculer les écarts
ecarts = [T_predites[i] - T_reelles[i] if not np.isnan(T_predites[i]) else None for i in range(4)]

for i in range(4):
    pred = T_predites[i]
    ecart = ecarts[i]
    # 检查是否为None或nan，避免格式化报错
    # Vérifier si la valeur est None ou nan pour éviter les erreurs de formatage
    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
        pred_str = "NA"
    else:
        pred_str = f"{pred:.2f}"
    if ecart is None or (isinstance(ecart, float) and np.isnan(ecart)):
        ecart_str = "NA"
    else:
        ecart_str = f"{ecart:.2f}"
    print(f"{mois_pred[i].capitalize()} 2025 : prédite = {pred_str} °C, réelle = {T_reelles[i]} °C, écart = {ecart_str} °C")

# 写入csv
# Écrire les résultats dans le fichier de réponses
df_reponses.loc[df_reponses['Question'] == 'q17', 'Reponse1'] = str(T_predites)
df_reponses.loc[df_reponses['Question'] == 'q17', 'Reponse2'] = str(ecarts)
df_reponses.to_csv('YuefanLIU_MouzhengLI_LianghongLI.csv', index=False)

# 打印回答
print("\n[q17] Prédictions pour janvier-avril 2025 :")
for i in range(4):
    pred = T_predites[i]
    ecart = ecarts[i]
    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
        pred_str = "NA"
    else:
        pred_str = f"{pred:.2f}"
    if ecart is None or (isinstance(ecart, float) and np.isnan(ecart)):
        ecart_str = "NA"
    else:
        ecart_str = f"{ecart:.2f}"
    print(f"  {mois_pred[i].capitalize()} 2025 :")
    print(f"    Température prédite : {pred_str}°C")
    print(f"    Température réelle : {T_reelles[i]}°C")
    print(f"    Écart : {ecart_str}°C")

# 评论：
# On constate que l'écart entre les températures prédites et réelles peut être important, surtout pour les mois éloignés de la période d'apprentissage.
# Cela montre que le modèle multivarié, bien qu'il puisse s'ajuster aux données passées, n'est pas toujours fiable pour la prévision à long terme (effet d'accumulation d'erreur, extrapolation).
# Il est donc préférable de rester prudent dans l'utilisation de ce modèle pour la prévision de plusieurs mois à l'avance.


