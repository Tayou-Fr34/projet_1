

# 🧬 Criblage Ligand-Based — *App integrated_plus*

Cette application **Streamlit** permet de réaliser un criblage **ligand-based complet** :
importation de bibliothèques, fusion, filtrage *drug-likeness*, comparaison de similarité, et visualisation **3D / pharmacophore**. C'est une application web interactive développée avec Streamlit et RDKit.
##### **_Binôme : Guillaume Patient & Cédric MANELLI_**

Cette application permet de :

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

🖼️ Py3DMol – rendu des images moléculaires

⚙️ Conda – gestion d’environnement
---


## ⚙️ 1. Installation et exécution

1️⃣ Cloner ou télécharger le projet

```
git clone https://github.com/Tayou-Fr34/projet_1.git
cd chemlite
```

2️⃣ Créer l’environnement Conda

```
conda env create -f environment.yml
conda activate chemapp-v2-env
```
3️⃣ Lancer l’application Streamlit
```
streamlit run app_criblage_up6-2.py
```

## 🧱 2. Génération et gestion des chimiothèques intégrées



Quatre bibliothèques **intégrées** sont générées à partir de motifs combinatoires :

| Nom                            | Taille approx. | Contenu                            |
| ------------------------------ | -------------- | ---------------------------------- |
| **ChEMBL-subset (~10k)**       | 10 000         | Dérivés aromatiques bio-inspirés   |
| **ZINC-fragments (~8k)**       | 8 000          | Fragments simples issus de ZINC    |
| **DrugBank-core (~2k)**        | 2 000          | Structures médicamenteuses de base |
| **NaturalProducts-core (~5k)** | 5 000          | Dérivés naturels                   |

### Étapes :

1. Référencer un chemin d'accès au dossier librairies ou pour créer les biliothèques à partir du chemin
2. Ouvrir la section **⚙️ Gestion des chimiothèques intégrées**
3. Cliquer sur :

   * **🧱 Créer/mettre à jour** (ajoute si absentes)
   * ou **♻️ Régénérer** (reconstruit depuis zéro)
   * ou ** 🔄 Re-scanner le dossier (dans le cas d'erreur d'importation)
3. Les fichiers `.sdf` sont enregistrés dans `LIB_PATH`.

---

## 🗂️ 3. Fusion et visualisation des bibliothèques

### a) Sélection

* Dans la section **A. Sélection & fusion**, coche une ou plusieurs bibliothèques.
* Tu peux aussi **ajouter ta propre bibliothèque** :

  * Formats acceptés : `.sdf`, `.smi`, `.csv`, `.tsv`.

### b) Fusion et déduplication

L’app charge chaque lib, supprime les doublons (via **InChIKey/SMILES**), calcule la **diversité interne**, puis fusionne :

* **Déduplication globale** → suppression des redondances entre bibliothèques
* **Diversité chimique** → distance Tanimoto moyenne (`1 - similarité moyenne`)

💡 Une diversité proche de **1.0** indique une bibliothèque variée, proche de **0.0** → structures similaires.

---

## 🧪 4. Filtres *drug-likeness*

Section **B. Filtres drug-likeness** → optionnels mais utiles pour la qualité chimique :

| Filtre       | Condition                                | Objectif                          |
| :----------- | :--------------------------------------- | :-------------------------------- |
| **Lipinski** | MW ≤ 500 ; LogP ≤ 5 ; HBD ≤ 5 ; HBA ≤ 10 | Compatibilité orale (règle des 5) |
| **Veber**    | TPSA ≤ 140 ; RotBonds ≤ 10               | Bonne biodisponibilité            |
| **QED**      | QED ≥ seuil (0.0–1.0)                    | Indice global “drug-like”         |

➡️ Active les cases souhaitées et ajuste le **seuil QED** (par défaut 0.4).
L’app indique combien de molécules restent après filtrage.

---

## 🧬 5. Ligands témoins (références)

Les **témoins** servent de modèles pour la comparaison et le pharmacophore.

1. Section **C. Ligands de référence (témoins)**
2. Importer un ou plusieurs fichiers `.sdf` ou `.smi` (2 molécules sont présentes sur github "Structure2D_COMPOUND_CID_2244.sdf" et "Conformer3D_COMPOUND_CID_2519.sdf" afin de les tester. Télécharger et dezipper pour test).
  
3. L’application :

   * supprime les doublons
   * affiche la **structure 3D** (générée si absente)
   * calcule les **propriétés physico-chimiques** :

     * Poids moléculaire, LogP, TPSA, HBA, HBD, RotBonds, QED
   * fournit aussi le **SMILES** et l’**InChIKey**

---

## 🎯 6. Criblage *ligand-based*

1. Section **D. Criblage ligand-based**
2. Importer les **ligands à cribler** (`.sdf` ou `.smi`) (2 molécules sont présentes sur github "Structure2D_COMPOUND_CID_2244.sdf" et "Conformer3D_COMPOUND_CID_2519.sdf" afin de les tester. Télécharger et dezipper pour test).

3. Régler :

   * **Seuil Tanimoto** (souvent ≥ 0.7)
   * **Top-K hits** (0 = tous)
4. Cliquer sur **🚀 Lancer le criblage**

### Fonctionnement

* Calcul de similarité **Tanimoto (empreinte Morgan 2048 bits)**
* Comparaison de chaque ligand avec la bibliothèque filtrée
* Résultats triés par similarité décroissante

### Résultats

| Colonne             | Description                           |
| :------------------ | :------------------------------------ |
| Molécule requête    | Nom du ligand à cribler               |
| Molécule similaire  | Molécule trouvée dans la bibliothèque |
| Similarité Tanimoto | Score entre 0 et 1                    |
| Formule chimique    | Formule brute                         |
| SMILES / InChIKey   | Identifiants structuraux              |

📥 **Téléchargement CSV** disponible (`resultats_criblage_detaille.csv`)

---

## 🧩 7. Pharmacophore 3D

1. Section **E. Pharmacophore 3D des témoins**
2. Choisir le **mode d’affichage** :

   * 💡 *Pharmacophore seul*
   * 🔬 *Superposé sur les molécules de référence*
3. Cliquer **🎥 Générer l’affichage pharmacophore**

### Légende des points

| Couleur | Type         | Signification          |
| :------ | :----------- | :--------------------- |
| 🔵      | Donor        | Donneur de liaison H   |
| 🔴      | Acceptor     | Accepteur de liaison H |
| 🟢      | Aromatic     | Cycle aromatique       |
| 🟡      | Hydrophobe   | Zone apolaire          |
| 🟣      | PosIonizable | Groupe basique         |
| 🟦      | NegIonizable | Groupe acide           |

📦 **Téléchargement du pharmacophore** : `pharmacophore.json`

---

## 💾 8. Résultats et export

* **Bibliothèque finale (fusionnée)** → sauvegardée en `.sdf` dans `LIB_PATH`
* **Résultats du criblage** → `.csv`
* **Pharmacophore 3D** → `.json`

---

## 💡 Notes pratiques

* Sur de **grandes bibliothèques** (>10 000 molécules), le calcul peut être long → réduire *Top-K* ou *seuil Tanimoto*.
* Les **librairies intégrées** sont générées une seule fois puis **mises en cache**.
* La **diversité interne** est un indicateur clé de redondance :

  * > 0.6 = bonne variété
  * <0.3 = trop de similarité entre composés.




