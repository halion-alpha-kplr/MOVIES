from flask import Flask, Blueprint, jsonify, render_template, request
import json
import findspark
from inflect import engine
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from engine import RecommendationEngine
from engine import engine


# Création du Blueprint Flask
main = Blueprint('main', __name__)

# Initialisation de Spark
findspark.init()

@main.route("/", methods=["GET", "POST", "PUT"])
def home():
    return render_template("index.html")

# Définition de la route pour récupérer les détails d'un film
@main.route("/movies/<int:movie_id>", methods=["GET"])
def get_movie(movie_id):
    movie = engine.get_movie(movie_id)
    return jsonify(movie)

# Définition de la route pour ajouter de nouvelles évaluations pour les films
@main.route("/newratings/<int:user_id>", methods=["POST"])
def new_ratings(user_id):
    if not engine.is_user_known(user_id):
        engine.create_user(user_id)
    ratings = json.loads(request.data)
    engine.add_ratings(user_id, ratings)
    return jsonify(user_id if not engine.is_user_known(user_id) else "")

# Définition de la route pour ajouter des évaluations à partir d'un fichier
@main.route("/<int:user_id>/ratings", methods=["POST"])
def add_ratings(user_id):
    file = request.files["file"]
    ratings = parse_ratings_file(file)
    engine.add_ratings(user_id, ratings)
    return "Le modèle de prédiction a été recalculé"

# Définition de la route pour obtenir la note prédite d'un utilisateur pour un film
@main.route("/<int:user_id>/ratings/<int:movie_id>", methods=["GET"])
def movie_ratings(user_id, movie_id):
    prediction = engine.predict_rating(user_id, movie_id)
    return str(prediction)

# Définition de la route pour obtenir les meilleures évaluations recommandées pour un utilisateur
@main.route("/<int:user_id>/recommendations", methods=["GET"])
def user_recommendations(user_id):
    recommendations = engine.recommend_for_user(user_id, 5)
    return jsonify(recommendations)

# Définition de la route pour obtenir les évaluations d'un utilisateur
@main.route("/ratings/<int:user_id>", methods=["GET"])
def get_ratings_for_user(user_id):
    ratings = engine.get_ratings_for_user(user_id)
    return jsonify(ratings)

# Fonction pour créer l'application Flask
def create_app(spark_context, movies_set_path, ratings_set_path):
    # Création de l'application Flask
    app = Flask(__name__)

    # Enregistrement du Blueprint "main" dans l'application
    app.register_blueprint(main)

    # Configuration des options de l'application Flask

    return app