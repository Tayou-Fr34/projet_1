# 🧪 ChemLite — Application Web de Visualisation et d’Analyse Moléculaire

***Projet Python – Introduction à la Programmation***

##### **_Binôme : Cédric MANELLI & [Nom de ton binôme]_**

## 🚀 **Description du projet**

ChemLite est une application web interactive développée avec Streamlit et RDKit.
Elle permet de :

- Visualiser des structures moléculaires à partir de chaînes SMILES 🧬

- Calculer automatiquement des propriétés physico-chimiques (poids moléculaire, LogP, HBA/HBD, tPSA, etc.) ⚗️

- Vérifier la conformité avec les règles de Lipinski 💊

- Rechercher des molécules similaires dans une base de données importée (similarité de Tanimoto) 🔍

- Identifier des sous-structures spécifiques à partir d’un motif SMARTS 🧩

Ce projet combine des notions de chimie organique et de programmation Python, dans le cadre d’une introduction à la science des données appliquée à la pharma.

## 🧱 Technologies utilisées

🐍 Python 3.11

💻 Streamlit – interface web interactive

🧬 RDKit – manipulation et analyse de structures chimiques

📊 Pandas – gestion de données tabulaires

🖼️ Pillow – rendu des images moléculaires

⚙️ Conda – gestion d’environnement

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## ⚙️ Installation et exécution

1️⃣ Cloner ou télécharger le projet

```
git clone https://github.com/<votre-repo>/chemlite.git
cd chemlite
```

2️⃣ Créer l’environnement Conda

```
conda env create -f environment.yml
conda activate chemapp
```
3️⃣ Lancer l’application Streamlit
```
streamlit run app.py
```
