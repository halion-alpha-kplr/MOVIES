# Utilisez une image de base contenant Python et Spark
FROM apache/spark:3.4.0

# Définissez le répertoire de travail de l'application
WORKDIR /app

# Copiez les fichiers de l'application dans le conteneur
COPY ./requirements.txt  /app/requirements.txt


# Installez les dépendances requises
RUN pip install -r requirements.txt

# Copie tous les fichiers et répertoires du répertoire de construction vers le répertoire de travail du conteneur.
COPY  . .

# Exposez le port sur lequel l'application Flask sera exécutée
EXPOSE 5432
 
# Démarrez l'application Flask
CMD ["spark-submit", "app/server.py", "ml-latest/movies.csv", "ml-latest/ratings.csv"]