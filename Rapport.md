# Sarcasme Détection

## Description des tâches

* Trouver un dataset utilisable (il est assez difficile de trouver des datasets de phrases sarcastiques)
* Nettoyer les données
* Transformation des données en vecteurs (un model ne peut pas lire du texte brut) 
* Entrainer un model sur les données
* Analyser les résultats du model

##  Jeux de données
Le jeux de données utilisés est un dataset contenant des titres d'articles sarcastique. Nous avons choisi ce dataset car c'est le seul que nous ayons trouvé qui contiennent des phrases grammaticalement correcte et avec peu de bruit.

Chaque ligne du fichier contient 3 attributs :

* ```is_sarcastic```: 1 si le titre est sarcastique, 0 sinon

* ```headline```: Le titre de l'article

* ```article_link```: Lien vers l'article

 Les données sont bien réparti
 
 ![alt text](https://github.com/MaelGiese/TATIA/blob/master/image/sarcastic%20vs%20non-sarcastic.png "sarcasme vs non-sarcastique")
 
### Nettoyage des données
Les données du dataset sont assez "propre" peu de nettoyage est nécéssaire.

* Suppréssion de la colonne ```article_link```, on ne l'utilisera pas
* On retire toute les ```headline``` null
* Séparation du label ```is_sarcastic``` et de la seule feature ```headline```

Toutes les fonctions utilisées pour nettoyer les données se trouve dans le fichier `Data_pre_treatment.py`

### Transformation des données en vecteurs
#### Tokenizer
Utilisation d'un tokenizer deja implémenté par keras, création d'un bag of words a partir du dataset.
C'est l'implémentation la plus simple cependant 
#### Word2Vec
#### Pretrained word embedding

Source : https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection

## Détails de la conception et implémentation de l'algorithme/système
