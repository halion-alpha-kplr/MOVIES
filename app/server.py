
# Import des bibliothèques nécessaires
import os
import cherrypy
from cheroot.wsgi import Server as WSGIServer
from cheroot.wsgi import PathInfoDispatcher
from app import create_app


# Création de l'objet SparkConf
conf = SparkConf().setAppName("movie_recommendation-server")

# Initialisation du contexte Spark
sc = SparkContext(conf=conf, pyFiles=['engine.py', 'app.py'])

# Obtention des chemins des jeux de données des films et des évaluations à partir des arguments de la ligne de commande
movies_set_path = sys.argv[1] if len(sys.argv) > 1 else ""
ratings_set_path = sys.argv[2] if len(sys.argv) > 2 else ""

# Création de l'application Flask
app = create_app(sc, movies_set_path, ratings_set_path)

# Configuration et démarrage du serveur CherryPy
cherrypy.tree.graft(app.wsgi_app, '/')
cherrypy.config.update({
    'server.socket_host': '0.0.0.0',
    'server.socket_port': 5432,
    'engine.autoreload.on': False
})
cherrypy.engine.start()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5432)