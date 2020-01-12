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

 Les données sont bien réparties entre les titres sarcastique et non-sarcastique
 ![alt text](https://github.com/MaelGiese/TATIA/blob/master/image/sarcastic%20vs%20non-sarcastic.png "sarcasme vs non-sarcastique")
 
 Source : https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection
### Nettoyage des données
Les données du dataset sont assez "propre" peu de nettoyage est nécéssaire.

* Suppréssion de la colonne ```article_link```, on ne l'utilisera pas
* On retire toute les ```headline``` null
* Séparation du label ```is_sarcastic``` et de la seule feature ```headline```
* Suppréssion des stop words en fonction de quelle algorithme on va utiliser

Toutes les fonctions utilisées pour nettoyer les données se trouve dans le fichier `Data_pre_treatment.py`

## Détails de la conception et implémentation de l'algorithme/système

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
L'idée ici est la meme que pour notre implémentation avec Word2Vec sauf que nous avons utilisés un model de word embedding deja entrainer (tout les fonctions utilisées sont dans le fichier `pretrained_word_embedding.py`).
Nous avons utilisés le meme algorithme et les meme solutions que pour notre implémentation avec Word2Vec

#### Résultats

Les résultats obtenus par la première solution sont assez bon 80% de précision environ mais les implémentation utilisant les word embedding ont de mauvais résultat (plus proche de 60% de précision).  
  
Les solutions que nous avons utilisées pour les implémentations avec Word2Vec et avec un pretrained model de word embedding ne sont pas optimal et nuisent aux résultats obtenus par le model néanmois il était intéressant d'essayer d'implémenter ces solutions nous meme.
  
Pour la suite nous ne parlerons que de l'implémentation utilisant les bag of words étant données que c'est celle qui donne les meilleurs résultats.

## Model 
Le model utilisé est un model séquentiel de keras.  
  
Code :  
* Setup du model  
![alt text](https://github.com/MaelGiese/TATIA/blob/master/image/Model%20setup.JPG)
  
* Entrainement du model  
![alt text](https://github.com/MaelGiese/TATIA/blob/master/image/Model%20training.JPG)

## Evaluation
Suite a l'entrainement du model on obtient le score suivant :

![alt text](https://github.com/MaelGiese/TATIA/blob/master/image/accuracy.JPG "accuracy")
  
 Le model a une précision de 81% sur les données de tests




