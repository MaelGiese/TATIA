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
C'est l'implémentation la plus simple cependant pas la plus intéréssante a utiliser, on a donc essayé d'autre technique.

#### Word2Vec
L'idée d'utiliser des vecteurs pour représenter chaque mots d'une phrase puis d'entrainer un model sur ce "bag of vectors" semble etre une bonne idée pour détecter du sarcasme en théorie, on pourait supposer que dans une phrase sarcastique on a au moins deux mots très peu similaire.  
On a donc entrainé un model Word2Vec a partir de notre jeu données (toute les fonctions utilisées pour entrainer notre model Word2Vec sont dans le fichier `Word_embedding.py`) puis nous avons utilisé ce model pour créer un tableau de vecteur représentant chaque phrases du dataset.  
Chaque phrase est donc maintenant constituée d'un tableau vecteur, chaque vecteur représentant un de ses mots, seulement deux problèmes se pose :
1. Nos données sont maintenant représenté en 3 dimensions non plus en 2 (avec un tokenizer par exemple on a chaque phrase représenté par un unique vecteur ici on a un tableau de vecteur pour chaque phrase).
2. Chaque phrases ayant un nombre de mots différent la taille de chaque tableau de vecteur est différente (chaque vecteur doit avoir la meme taille si l'ont veut entrainer un model).  
  
Pour régler le 1. problème nous avons du fusionner tous les vecteurs d'un tableau de vecteur (une phrase) en un seul vecteur contenant toute les valeurs, cette solution n'est pas optimal mais nous n'avons pas trouvé d'autre solution.  
Pour régler le 2. problème nous avons fixé une taille maximale pour nos phrases et ensuite remplis les valeurs inexistantes par des 0.


#### Pretrained word embedding
L'idée ici est la meme que pour notre implémentation avec Word2Vec sauf que nous avons utilisé un model de word embedding deja entrainer (tout les fonctions utilisées sont dans le fichier `pretrained_word_embedding.py`).

#### Résultats

Ces solutions ne sont pas optimal et nuisent aux résultats obtenus par le model néanmois il était intéressant d'essayer d'implémenter cet algorithme nous meme.

Source : https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection

## Détails de la conception et implémentation de l'algorithme/système
