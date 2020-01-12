# Projet TATIA - Sarcasme détection

### Groupe : 
* Giese Mael
* Simon Fabrice

### Sujet :
Le but du projet est de développer un programme capable de reconnaître une phrase sarcastique.

## Dataset

Chaque ligne du fichier contient 3 attributs :

* ```is_sarcastic```: 1 si le titre est sarcastique, 0 sinon

* ```headline```: Le titre de l'article

* ```article_link```: Lien vers l'article

Source : https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection

## Programme
* `main.py` 
* `Data_pre_treatment.py` contient les fonctions permettant de nettoyer les données mais également de transformer les phrases en vecteurs.
* `model.py` contient les fonctions permettant d'entrainer un model sur les données prétraitées puis de sauvegarder ce model et de le recharger.
* `Word_embedding.py` permet d'entrainer un model Word2Vec sur le dataset.
* `pretrained_word_embedding.py` permet de charger un dictionnaire de vecteurs associé 

## Utilisation
Lancer le main puis utiliser la fonction mo.is_sarcasm(model, tokenizer, sentence).
#### ex :
> mo.is_sarcasm(model, tokenizer, 'I like trains')
>I like trains
>Non-sarcastic
