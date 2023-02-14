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