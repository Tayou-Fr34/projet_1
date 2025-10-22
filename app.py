import io
import pandas as pd
import streamlit as st

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Draw, Lipinski, rdMolDescriptors as rdmd, AllChem, DataStructs

st.set_page_config(page_title="ChemLite-RDKit & Streamlit used", layout="wide")

# ---------- Utilitaires RDKit ----------
def mol_from_smiles(smiles: str):
    if not smiles: return None
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        Chem.SanitizeMol(mol)
    return mol

def mol_to_png_bytes(mol, size=(350, 300)):
    if mol is None: return None
    img = Draw.MolToImage(mol, size=size)
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return buff.getvalue()

def mol_properties(mol):
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "tPSA": rdmd.CalcTPSA(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "Rings": rdmd.CalcNumRings(mol),
    }

def lipinski_flags(props):
    ok = (
        props["MolWt"] <= 500 and
        props["LogP"] <= 5 and
        props["HBA"] <= 10 and
        props["HBD"] <= 5
    )
    return "Conforme (Lipinski)" if ok else "Hors borne(s) (Lipinski)"

def morgan_fp(mol, radius=2, nbits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return fp

def tanimoto(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def substructure_match(mol, smarts):
    patt = Chem.MolFromSmarts(smarts)
    return mol.HasSubstructMatch(patt) if (mol and patt) else False

# ---------- UI ----------
st.title("ChemLite — RDKit x Streamlit")
st.caption("MVP : visualisation, proprietes, regles de Lipinski, similarite et sous-structure.")

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.header("🎯 Entree")
    smiles = st.text_input("SMILES", value="CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirine par defaut
    uploaded = st.file_uploader("CSV avec une colonne 'smiles' (optionnel)", type=["csv"])
    smarts = st.text_input("Sous-structure (SMARTS, optionnel)", value="c1ccccc1")  # motif anneau benzenique
    st.markdown("—")
    st.subheader("⚙️ Paramètres")
    radius = st.slider("Rayon Morgan FP", 1, 3, 2)
    nbits = st.selectbox("Bits FP", [512, 1024, 2048], index=2)

with col_right:
    st.header("📷 Molecule & Proprietes")
    mol = mol_from_smiles(smiles)
    if mol:
        st.image(mol_to_png_bytes(mol), caption="Mol 2D", use_column_width=False)
        props = mol_properties(mol)
        st.write(pd.DataFrame([props]))
        st.info(lipinski_flags(props))
    else:
        st.error("SMILES invalide")

st.divider()

# ---------- Similarite / dataset ----------
st.header("🔎 Dataset & Similarite (optionnel)")
if uploaded is not None:
    df = pd.read_csv(uploaded)
    if "smiles" not in df.columns:
        st.error("Le CSV doit contenir une colonne 'smiles'")
    else:
        # Pre-calcul mols + FP
        df["mol"] = df["smiles"].apply(mol_from_smiles)
        df = df[df["mol"].notna()].copy()
        df["fp"] = df["mol"].apply(lambda m: morgan_fp(m, radius=radius, nbits=nbits))

        if mol:
            query_fp = morgan_fp(mol, radius=radius, nbits=nbits)
            df["tanimoto"] = df["fp"].apply(lambda fp: tanimoto(query_fp, fp))
            st.subheader("🔬 Top similaires (Tanimoto)")
            topk = st.slider("Nombre à afficher", 5, 50, 10)
            show = df.sort_values("tanimoto", ascending=False).head(topk).copy()
            show["preview"] = show["mol"].apply(lambda m: mol_to_png_bytes(m))
            # petit rendu : image + smiles + score
            for _, row in show.iterrows():
                c1, c2, c3 = st.columns([1, 3, 1])
                with c1: st.image(row["preview"], width=120)
                with c2: st.code(row["smiles"])
                with c3: st.metric("Tanimoto", f"{row['tanimoto']:.3f}")

        # Sous-structure
        if smarts.strip():
            st.subheader("🧩 Match sous-structure")
            df["has_pattern"] = df["mol"].apply(lambda m: substructure_match(m, smarts))
            st.write(df.loc[df["has_pattern"], ["smiles"]].head(50))

        # Export
        st.download_button("⬇️ Exporter resultats (CSV)",
                           data=df.drop(columns=["mol","fp"]).to_csv(index=False).encode("utf-8"),
                           file_name="results.csv",
                           mime="text/csv")
else:
    st.caption("Astuce : uploadez un petit CSV (colonne `smiles`) pour tester similarite et sous-structure.")

st.divider()
st.markdown("📝 *Projet ‘Intro Python’ — Streamlit × RDKit. equipe : Chimie + Data.*")