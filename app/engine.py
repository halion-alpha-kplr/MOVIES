from pyspark.sql.types import *
from pyspark.sql.functions import explode, col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext
import pandas as pd
import random
import pyspark.sql.functions as F
from pyspark import SparkContext, SparkConf
from server import sc



class RecommendationEngine:
    def __init__(self, sc, movies_set_path, ratings_set_path):
        self.sql_context = SQLContext(sc)
        self.movies_df = self.__load_movies_data(movies_set_path)
        self.ratings_df = self.__load_ratings_data(ratings_set_path)
        self.max_user_identifier = self.ratings_df.select(F.max("userId")).first()[0]
        self.model = None
        self.training = None
        self.test = None
        self.rmse = None
        self.__train_model()

    def create_user(self, user_id=None):
        if user_id is None:
            user_id = self.max_user_identifier + 1
            self.max_user_identifier = user_id
        elif user_id > self.max_user_identifier:
            self.max_user_identifier = user_id
        return user_id

    def is_user_known(self, user_id):
        if user_id is not None and user_id <= max_user_identifier:
            return True
        else:
            return False

    def get_movie(self, movie_id=None):
        movies_df = pd.read_csv('/workspaces/MOVIES/app/ml-latest/movies.csv')
        if movie_id is None:
            movie = movies_df.sample(n=1)  # Échantillon aléatoire d'un film
        else:
            movie = movies_df[movies_df['movieId'] == movie_id]
        movie = movie[['movieId', 'title']]  # Sélectionner les colonnes "movieId" et "title" du film
        return movie

    def get_ratings_for_user(self, user_id):
        return self.ratings_df.filter(F.col("userId") == user_id).select("movieId", "userId", "rating")

    def add_ratings(self, user_id, ratings):
        new_ratings_df = self.sql_context.createDataFrame(ratings, ["userId", "movieId", "rating"])
        self.ratings_df = self.ratings_df.union(new_ratings_df)
        self.__train_model()

    def predict_rating(self, user_id, movie_id):
        rating_df = self.sql_context.createDataFrame([(user_id, movie_id)], ["userId", "movieId"])
        prediction = self.model.transform(rating_df).select("prediction").collect()
        if prediction:
            return prediction[0]["prediction"]
        else:
            return -1

    def recommend_for_user(self, user_id, nb_movies):
        user_df = self.sql_context.createDataFrame([(user_id,)], ["userId"])
        recommendations = self.model.recommendForUserSubset(user_df, nb_movies)
        rec_movies_df = recommendations.select("userId", F.explode("recommendations").alias("rec")) \
            .select("userId", "rec.movieId")
        rec_movies_df = rec_movies_df.join(self.movies_df, rec_movies_df["movieId"] == self.movies_df["movieId"]) \
            .select("title")
        return rec_movies_df

    def __load_movies_data(self, movies_set_path):
        movies_schema = "movieId INT, title STRING"
        return self.sql_context.read.csv(movies_set_path, header=True, schema=movies_schema)

    def __load_ratings_data(self, ratings_set_path):
        ratings_schema = "userId INT, movieId INT, rating FLOAT"
        return self.sql_context.read.csv(ratings_set_path, header=True, schema=ratings_schema)

    def __train_model(self):
        self.training, self.test = self.ratings_df.randomSplit([0.8, 0.2])
        als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
        self.model = als.fit(self.training)


# Création d'une instance de la classe RecommendationEngine
engine = RecommendationEngine(sc, "app/ml-latest/movies.csv", "app/ml-latest/movies.csv")

# Exemple d'utilisation des méthodes de la classe RecommendationEngine
user_id = engine.create_user(None)
if engine.is_user_known(user_id):
    movie = engine.get_movie(None)
    ratings = engine.get_ratings_for_user(user_id)
    engine.add_ratings(user_id, ratings)
    prediction = engine.predict_rating(user_id, movie.movieId)
    recommendations = engine.recommend_for_user(user_id, 10)