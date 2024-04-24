# P8_cloud

OpenClassrooms Projet 8 : Déployer un modèle dans le cloud

## Description

[Project briefing from OpenClassrooms](https://openclassrooms.com/fr/paths/164/projects/633/assignment)
Une startup de l'AgriTech souhaite développer une application mobile de classification de fruits par reconnaissance d'image, avant de l'implémenter dans un robot cueilleur.

**Mission** : mettre en place une **architecture Big Data sur le cloud** pour traiter les données de l'application mobile

- calcul distribué avec Spark
- cloud AWS dans le respect des normes RGPD
- diffusion des poids du modèle TensorFlow
- réduction de dimension PCA
- sans entrainer modèle

## Usage

## Data

- Kaggle dataset : <https://www.kaggle.com/datasets/moltean/fruits>
- 131 fruits, 90380 images

```raw
├── Apple Braeburn
│   ├── 3_100.jpg
│   ├── r_3_100.jpg
│   └── ...
├── Banana
│   ├── 12_100.jpg
│   ├── r_105_100.jpg
│   └── ...
├── Strawberry
│   ├── 100_100.jpg
│   ├── r_64_100.jpg
│   └── ...
└── ...
    └── ...
```

## Install

## Makefile

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
