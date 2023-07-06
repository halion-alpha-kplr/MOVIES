

/SPARK-MOVIE
│   Dockerfile (à faire)
│   requirements.txt
│
└───app
    │   app.py (à faire)
    │   engine.py (à faire)
    │   server.py (à faire)
    │
    ├───ml-latest
    │       movies.csv
    │       new-ratings.csv
    │       ratings.csv
    │
    ├───static
    │       index.js
    │       omr.css
    │
    └───templates
            index.html

Workshop 0:

1. Utiliser Docker pour créer une image conteneurisée de l'application Flask avec Spark et les dépendances requises.
2. Configurer les paramètres Docker pour exposer le conteneur Flask sur un port spécifique.
3. Construire et exécuter le conteneur Docker pour déployer l'application de recommandation de films.
4. Tester l'application en accédant aux différentes fonctionnalités de recommandation via les routes HTTP.

Workshop 1 :

1. Comprendre les bases des systèmes de recommandation et l'algorithme ALS (Alternating Least Squares).
2. Installer Python et Docker sur votre machine.
3. Configurer un environnement de développement Python avec Flask pour la création de l'application.
4. Créer une application Flask simple avec des routes de base pour répondre aux requêtes HTTP.
5. Mettre en place une architecture de base pour le système de recommandation en utilisant Flask, sans implémenter encore l'algorithme ALS.

Workshop 2 :

1. Installer Spark et configurer l'environnement Spark pour travailler avec Python.
2. Comprendre les concepts fondamentaux de Spark, notamment SparkContext et SparkConf.
3. Implémenter l'algorithme ALS pour la recommandation de films en utilisant Spark.
4. Charger les jeux de données des films et des évaluations dans Spark.
5. Appliquer l'algorithme ALS aux données pour générer un modèle de recommandation initial.

Workshop 3 :

1. Créer une application Flask qui intègre l'algorithme ALS pour la recommandation de films.
2. Mettre en place les routes Flask pour permettre l'accès aux fonctionnalités du système de recommandation.
3. Implémenter la logique nécessaire pour récupérer les recommandations de films pour un utilisateur spécifié.
4. Gérer les requêtes de notation de films par les utilisateurs et mettre à jour le modèle de recommandation en conséquence.

