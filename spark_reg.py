# Import de SparkSession et SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkContext

# Définition d'un SparkContext en local
sc = SparkContext.getOrCreate()

# Construction d'une session Spark
spark = SparkSession \
    .builder \
    .appName("Introduction à Spark ML") \
    .getOrCreate()
    
spark
# L'objectif est d'estimer l'année de sortie d'une chanson en 
# fonction de ses caractéristiques audio. 
# Pour cela nous allons implémenter une régression linéaire simple sur les informations du timbre pour prédire l'année de sortie.
# Chargement du fichier " YearPredictionMSD.txt" dans un DataFrame
df_raw = spark.read.csv('YearPredictionMSD.txt')

# Première méthode d'affichage 
print(df_raw.show(2, truncate = 4))
# Modifier les valeurs de 'truncate' ne permet pas de bien visualiser les données
# à cause du nombre de variables

# Deuxième méthode d'affichage
print(df_raw.sample(False, .00001, seed = 222).toPandas())
# Utiliser toPandas permet de mieux visualiser les données
#affichage des types de variables 
df_raw.printSchema()

# Importation de col du sous-module pyspark.sql.functions
from pyspark.sql.functions import col

# Convertir des colonnes relatives au timbre en double et l'année en int
#créer une variable qui sert comme argument de la fonction select
#changer le type de la premère variable en int et le reste en double
exprs = [col(c).cast("double") for c in df_raw.columns[1:91]]
df = df_raw.select(df_raw._c0.cast('int'), *exprs)

# Affichage du schéma des variables "df"
df.printSchema()

## Affichage d'un résumé descriptif des données
print(df.describe().toPandas())

#mise en forme svmlib
"""Pour pouvoir être utilisée par les algorithmes de Machine Learning de Spark ML, la base de données doit être un DataFrame contenant 2 colonnes :

La colonne label contenant la variable à prédire (label en anglais).
La colonne features contenant les variables explicatives (features en anglais).
La fonction DenseVector() issue du package pyspark.ml.linalg permet de regrouper plusieurs variables en une seule variable.

   Pour pouvoir utiliser la fonction DenseVector(), il faut utiliser la méthode map après avoir transformé le DataFrame en rdd."""
# Import de DenseVector du package pyspark.ml.linalg
from pyspark.ml.linalg import DenseVector

# Création d'un rdd en séparant la variable à expliquer des features
rdd_ml = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

# Création d'un DataFrame composé de deux variables : label et features
df_ml = spark.createDataFrame(rdd_ml, ['label', 'features'])

# Affichage des 10 premières lignes du DataFrame
df_ml.show(10)

# Créer deux DataFrames appelés train et test contenant respectivement 80% et 20% des données
# Décomposition des données en deux ensembles d'entraînement et de test
# Par défaut l'échantillon est aléatoirement réparti
train, test = df_ml.randomSplit([.8, .2], seed= 1234)
# Import de LinearRegression du package pyspark.ml.regression
from pyspark.ml.regression import LinearRegression

# Création d'une fonction de régression linéaire
lr = LinearRegression(labelCol='label', featuresCol= 'features')

# Apprentissage sur les données d'entraînement 
linearModel = lr.fit(train)
# Calcul des prédictions sur les données test
predicted = linearModel.transform(test)

# Affichage des prédictions
predicted.show()
# Calcul et affichage du RMSE
print("RMSE:", linearModel.summary.rootMeanSquaredError)

# Calcul et affichage du R2
print("R2:  ", linearModel.summary.r2)

from pprint import pprint

# Affichage des Coefficients du modèle linéaire
#affichage plus élegant
pprint(linearModel.coefficients)

# Fermeture de la session Spark 
spark.stop()
