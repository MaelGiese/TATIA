# Sarcasme Détection

## Description des tâches

* Trouver un dataset utilisable (il est assez difficile de trouver des datasets de phrases sarcastiques)
* Nettoyer les données
* Transformer les données en vecteurs (un model ne peut pas lire du texte brut) 
* Entrainer un model sur les données
* Analyser les résultats du model

##  Jeux de données
Le jeux de données utilisés est un dataset contenant des titres d'articles sarcastique. Nous avons choisi ce dataset car c'est le seul que nous ayons trouvé qui contiennent des phrases grammaticalement correcte et avec peu de bruit.

Chaque ligne du fichier contient 3 attributs :

* ```is_sarcastic```: 1 si le titre est sarcastique, 0 sinon

* ```headline```: Le titre de l'article

* ```article_link```: Lien vers l'article

### Nettoyage des données
Les données du dataset sont assez "propre" peu de nettoyage est nécéssaire.

* Suppréssion de la colonne ```article_link```, on ne l'utilisera pas
* On retire toute les ```headline``` null
* Séparation du label ```is_sarcastic``` et de la feature ```headline```



Source : https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection

## Détails de la conception et implémentation de l'algorithme/système
