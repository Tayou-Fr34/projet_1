# =====================================================
# app_criblage_integrated_plus.py
# Criblage ligand-based complet
# - 4 chimiothèques intégrées + ajout de libs perso
# - Fusion multi-libs, déduplication, métriques & diversité
# - Filtres Lipinski / Veber / QED
# - Témoins 3D + pharmacophore superposé
# - Résultats détaillés (SMILES, Formule, InChIKey, Similarité)
# Auteur : Guillaume Patient
# =====================================================

import os, json, time, random, itertools
import streamlit as st
import pandas as pd
from io import StringIO
from rdkit import Chem
from rdkit.Chem import (
    AllChem, Crippen, Descriptors, rdMolDescriptors,
    DataStructs, ChemicalFeatures, QED
)
from rdkit import RDConfig
from pathlib import Path

# =========================
# 📂 Sélection du dossier LIB_PATH (saisi utilisateur) + détection des libs intégrées
# =========================


st.sidebar.header("📂 Dossier des bibliothèques")

# 👉 mets une valeur par défaut qui te convient
_default_lib_path = r""
folder_path = st.sidebar.text_input(
    "Chemin du dossier contenant les bibliothèques (.sdf) : entre le chemin d'accès pour récupérer les dossiers ou pour les créer.",
    value=_default_lib_path,
    placeholder=r"Ex : C:\Users\...\mylibs"
)

# Vérifie/crée le dossier
if not folder_path:
    st.sidebar.warning("❌ Merci de renseigner un chemin d'accès dossier valide pour lancer l'application.")
    st.stop()

LIB_PATH = Path(folder_path)
try:
    LIB_PATH.mkdir(parents=True, exist_ok=True)
except Exception as e:
    st.sidebar.error(f"❌ Impossible d'accéder/créer le dossier : {e}")
    st.stop()

st.sidebar.success(f"Dossier OK ✅ : {LIB_PATH}")

def _integrated_targets(base: Path) -> dict[str, str]:
    """Noms attendus des bibliothèques intégrées → chemins cibles dans base."""
    return {
        "ChEMBL-subset (~10k)":          str(base / "chembl_subset_auto.sdf"),
        "ZINC-fragments (~8k)":          str(base / "zinc_frag_auto.sdf"),
        "DrugBank-core (~2k)":           str(base / "drugbank_core_auto.sdf"),
        "NaturalProducts-core (~5k)":    str(base / "np_core_auto.sdf"),
    }

def detect_integrated_libs(base: Path) -> dict[str, str]:
    """Retourne seulement celles qui existent vraiment sur disque."""
    out = {}
    for label, p in _integrated_targets(base).items():
        if Path(p).exists():
            out[label] = p
    return out

# Détection initiale
INTEGRATED = detect_integrated_libs(LIB_PATH)
if INTEGRATED:
    st.sidebar.info(f"📦 {len(INTEGRATED)} bibliothèques intégrées détectées :")
    for k in INTEGRATED:
        st.sidebar.write(f"- {k}")
else:
    st.sidebar.warning("⚠️ Aucune bibliothèque intégrée détectée (tu peux les générer à cette emplacement).")

# =========================
# Utilitaires moléculaires
# =========================
def to_inchikey(mol):
    try:
        from rdkit.Chem.inchi import MolToInchiKey
        return MolToInchiKey(mol)
    except Exception:
        return None

def canonical_smiles(mol):
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def deduplicate_mols(mols):
    """Déduplication par InChIKey sinon SMILES canonique."""
    seen = set()
    uniq, dups = [], 0
    for m in mols:
        if m is None:
            continue
        key = to_inchikey(m) or canonical_smiles(m)
        if key in seen:
            dups += 1
            continue
        seen.add(key)
        if not m.HasProp("_Name"):
            m.SetProp("_Name", f"Mol_{len(uniq)+1}")
        uniq.append(m)
    return uniq, dups

def compute_props(mol):
    m = Chem.RemoveHs(mol)
    return {
        "Nom": mol.GetProp("_Name") if mol.HasProp("_Name") else "Molécule",
        "Formule": rdMolDescriptors.CalcMolFormula(m),
        "Masse moléculaire (g/mol)": round(Descriptors.MolWt(m), 2),
        "LogP": round(Crippen.MolLogP(m), 2),
        "TPSA": round(rdMolDescriptors.CalcTPSA(m), 2),
        "HBA": rdMolDescriptors.CalcNumHBA(m),
        "HBD": rdMolDescriptors.CalcNumHBD(m),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(m),
        "QED": round(QED.qed(m), 3),
    }

def molblock3d(mol):
    if mol.GetNumConformers() == 0:
        m = Chem.AddHs(Chem.Mol(mol))
        AllChem.EmbedMolecule(m, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(m)
        return Chem.MolToMolBlock(m)
    return Chem.MolToMolBlock(mol)

def show3d(mol, width=420, height=420):
    import py3Dmol
    v = py3Dmol.view(width=width, height=height)
    v.addModel(molblock3d(mol), "mol")
    v.setStyle({"stick": {"radius": 0.15}})
    v.zoomTo()
    st.components.v1.html(v._make_html(), height=height+20)

def morgan_fp(m, radius=2, nBits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits)

def tanimoto(m1, fp2):
    fp1 = morgan_fp(m1)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def internal_diversity(mols, sample_pairs=1000):
    """Diversité interne = distance Tanimoto moyenne = 1 - sim moyenne (échantillonnée)."""
    if len(mols) < 2:
        return 0.0
    # Pré-calc FP pour échantillon
    sample = mols if len(mols) <= 1500 else random.sample(mols, 1500)
    fps = [morgan_fp(m) for m in sample]
    n = min(sample_pairs, len(sample)*(len(sample)-1)//2)
    if n <= 0:
        return 0.0
    pairs = set()
    while len(pairs) < n:
        i = random.randrange(len(sample))
        j = random.randrange(len(sample))
        if i>=j: 
            continue
        pairs.add((i,j))
    sims = [DataStructs.TanimotoSimilarity(fps[i], fps[j]) for (i,j) in pairs]
    if not sims:
        return 0.0
    return round(1 - sum(sims)/len(sims), 3)

# =========================
# Chargement fichiers
# =========================
def load_sdf_file(path):
    suppl = Chem.SDMolSupplier(path)
    return [m for m in suppl if m]

def load_smi_text(text: str):
    mols = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m:
            m.SetProp("_Name", s)
            mols.append(m)
    return mols

def load_csv_tsv(file_bytes: bytes, sep: str):
    df = pd.read_csv(StringIO(file_bytes.decode("utf-8", errors="ignore")), sep=sep)
    cols = {c.lower(): c for c in df.columns}
    smi_col = cols.get("smiles") or cols.get("smile")
    name_col = cols.get("name")
    if not smi_col:
        raise ValueError("Le fichier ne contient pas de colonne 'smiles'.")
    mols = []
    for i, row in df.iterrows():
        smi = str(row[smi_col]).strip()
        if not smi or smi.lower() == "nan":
            continue
        m = Chem.MolFromSmiles(smi)
        if m:
            nm = str(row[name_col]) if name_col else f"Mol_{i+1}"
            m.SetProp("_Name", nm)
            mols.append(m)
    return mols

def save_library_as_sdf(mols, name_hint: str):
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name_hint).strip("_")
    if not safe:
        safe = f"lib_{int(time.time())}"
    path = os.path.join(LIB_PATH, f"{safe}.sdf")
    w = Chem.SDWriter(path)
    for m in mols:
        w.write(m)
    w.close()
    return path, safe

# =========================
# Génération libs intégrées
# =========================
def combinatorial_smiles(cores, subs_A, subs_B, target_n, name_prefix):
    # (garde ton implémentation telle quelle)
    out = []
    random.shuffle(cores); random.shuffle(subs_A); random.shuffle(subs_B)
    iA = iB = 0
    for c in cores:
        for _ in range(len(subs_A)):
            a = subs_A[iA % len(subs_A)]; iA += 1
            base = c.replace("*", a) if "*" in c else c + a
            if "&&" in base:
                b = subs_B[iB % len(subs_B)]; iB += 1
                smi = base.replace("&&", b)
            else:
                smi = base
            out.append(smi)
            if len(out) >= target_n * 3:
                break
        if len(out) >= target_n * 3:
            break
    mols = []
    seen = set()
    for s in out:
        m = Chem.MolFromSmiles(s)
        if not m: 
            continue
        key = Chem.MolToSmiles(m, isomericSmiles=True)
        if key in seen: 
            continue
        seen.add(key)
        mols.append(m)
        if len(mols) >= target_n:
            break
    for i, m in enumerate(mols, 1):
        m.SetProp("_Name", f"{name_prefix}_{i:06d}")
    return mols

def ensure_integrated_libraries(base: Path, regenerate: bool = False):
    """(Re)génère les 4 bibliothèques intégrées dans `base` si absentes ou si `regenerate`."""
    targets = _integrated_targets(base)

    chembl_cores = [
        "c1ccccc1*", "c1ncccc1*", "c1ccncc1*", "c1ccoc1*", "c1ccsc1*",
        "c1ccc(-c2ccccc2)cc1*", "c1ccc(NC(=O)*)cc1", "c1ccc(OC*)cc1",
        "O=C(N*)c1ccccc1", "O=C(O*)c1ccccc1", "c1ccc(C(=O)N*)cc1", "c1ccc(C(=O)O*)cc1",
    ]
    chembl_subs_A = ["N", "N(C)C", "O", "OC", "OCC", "CC", "CCO", "CCN", "CN", "C(=O)N", "C(=O)O", "CO", "COC", "CCl", "CF", "CBr"]
    chembl_subs_B = ["C", "CC", "CCC", "CO", "CN", "OC", "OCC", "N", "NC", "NCC"]

    zinc_cores = ["c1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1", "C1CCCCC1", "C1=CC=CN=C1", "C1=COC=C1", "C1=CSCC1", "C1CCNCC1", "C1COCCN1"]
    zinc_subs_A = ["", "F", "Cl", "Br", "C", "CC", "OC", "CN", "CF3", "N"]
    zinc_subs_B = ["", "C", "CC", "CO", "CN"]

    db_cores = ["O=C(N*)c1ccccc1", "O=C(O*)c1ccccc1", "c1ccc(O*)cc1", "c1ccc(N*)cc1", "c1ccc(CCN*)cc1", "c1ccc(CCO*)cc1"]
    db_subs_A = ["C", "CC", "CCC", "CO", "CN", "N", "N(C)C", "O", "OC", "C(=O)N", "C(=O)O"]
    db_subs_B = ["C", "CC", "CO", "CN"]

    np_cores = ["C=Cc1ccc(*)cc1", "CC(C)=Cc1ccc(*)cc1", "Oc1ccc(*)cc1", "COc1ccc(*)cc1", "C1CCC(CC1)*", "CC(C)C1=CC(*)=CC=C1"]
    np_subs_A = ["O", "OC", "OCC", "C", "CC", "CCC", "CO", "CN"]
    np_subs_B = ["C", "CC", "CCC", "CO", "OC"]

    jobs = [
        (targets["ChEMBL-subset (~10k)"],      chembl_cores, chembl_subs_A, chembl_subs_B, 10000, "CHEMBL"),
        (targets["ZINC-fragments (~8k)"],      zinc_cores,   zinc_subs_A,   zinc_subs_B,    8000, "ZINCFRAG"),
        (targets["DrugBank-core (~2k)"],       db_cores,     db_subs_A,     db_subs_B,      2000, "DBCORE"),
        (targets["NaturalProducts-core (~5k)"],np_cores,     np_subs_A,     np_subs_B,      5000, "NP"),
    ]

    for out_path, cores, sA, sB, n, prefix in jobs:
        if (not regenerate) and Path(out_path).exists():
            continue
        st.info(f"⏳ Génération de {os.path.basename(out_path)} ({n} molécules)...")
        mols = combinatorial_smiles(cores, sA, sB, n, prefix)
        mols, dups = deduplicate_mols(mols)
        w = Chem.SDWriter(out_path)
        for m in mols:
            w.write(m)
        w.close()
        st.success(f"✅ {os.path.basename(out_path)} : {len(mols)} uniques (doublons retirés: {dups}).")

# =========================
# Règles filtres drug-likeness
# =========================
def passes_lipinski(m):
    p = compute_props(m)
    return (p["Masse moléculaire (g/mol)"] <= 500 and p["LogP"] <= 5 and p["HBD"] <= 5 and p["HBA"] <= 10)

def passes_veber(m):
    p = compute_props(m)
    return (p["TPSA"] <= 140 and p["RotBonds"] <= 10)

def passes_qed(m, thresh):
    try:
        return QED.qed(Chem.RemoveHs(m)) >= thresh
    except Exception:
        return False

def apply_filters(mols, use_lipinski, use_veber, use_qed, qed_thresh):
    out = mols
    if use_lipinski:
        out = [m for m in out if passes_lipinski(m)]
    if use_veber:
        out = [m for m in out if passes_veber(m)]
    if use_qed:
        out = [m for m in out if passes_qed(m, qed_thresh)]
    return out

# =========================
# Pharmacophore
# =========================
def pharmacophore_points(mols):
    ff = ChemicalFeatures.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    )
    pts = []
    for mol in mols:
        for f in ff.GetFeaturesForMol(mol):
            p = f.GetPos()
            pts.append({
                "mol": mol.GetProp("_Name") if mol.HasProp("_Name") else "Molécule",
                "type": f.GetFamily(), "x": p.x, "y": p.y, "z": p.z
            })
    return pts

def show_pharmacophore(points, ref_mols=None, superpose=False):
    import py3Dmol
    color_map = {
        "Donor": "blue", "Acceptor": "red", "Aromatic": "green",
        "Hydrophobe": "yellow", "PosIonizable": "magenta", "NegIonizable": "cyan"
    }
    v = py3Dmol.view(width=600, height=500)
    if superpose and ref_mols:
        for m in ref_mols:
            v.addModel(molblock3d(m), "mol")
        v.setStyle({"stick": {"radius": 0.15}})
    for p in points:
        v.addSphere({
            "center": {"x": p["x"], "y": p["y"], "z": p["z"]},
            "radius": 0.6, "color": color_map.get(p["type"], "white"), "alpha": 0.6
        })
    v.zoomTo()
    st.components.v1.html(v._make_html(), height=520)
    st.markdown("""
### 🧭 Légende du pharmacophore
| Couleur | Type | Description |
|---|---|---|
| 🔵 Bleu | Donor | Donneur de liaison H |
| 🔴 Rouge | Acceptor | Accepteur de liaison H |
| 🟢 Vert | Aromatic | Cycle aromatique |
| 🟡 Jaune | Hydrophobe | Région apolaire |
| 🟣 Magenta | PosIonizable | Groupe basique |
| 🟦 Cyan | NegIonizable | Groupe acide |
""")

# =========================
# UI
# =========================
st.set_page_config(page_title="Criblage LB — libs intégrées + filtres", layout="wide")
st.title("🧬 Criblage Ligand-Based — Libs intégrées, fusion, filtres & pharmacophore")

# -------- Gestion des chimiothèques intégrées --------
with st.expander("⚙️ Gestion des chimiothèques intégrées", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🧱 Créer/mettre à jour (si absentes)"):
            ensure_integrated_libraries(LIB_PATH, regenerate=False)
            INTEGRATED = detect_integrated_libs(LIB_PATH)
            st.success(f"OK. Détectées : {len(INTEGRATED)}")
    with c2:
        if st.button("♻️ Régénérer (remplacer)"):
            ensure_integrated_libraries(LIB_PATH, regenerate=True)
            INTEGRATED = detect_integrated_libs(LIB_PATH)
            st.success(f"Recréées. Détectées : {len(INTEGRATED)}")
    with c3:
        if st.button("🔄 Re-scanner le dossier"):
            INTEGRATED = detect_integrated_libs(LIB_PATH)
            st.info(f"Re-scan terminé : {len(INTEGRATED)} détectées.")

# État session pour libs perso
if "user_libs" not in st.session_state:
    st.session_state.user_libs = {}

# --------- Bloc A : Sélection & fusion des bibliothèques ----------
st.header("🗂️ A. Sélection & fusion de chimiothèques")
available_libs = {**INTEGRATED, **st.session_state.user_libs}
selected = st.multiselect(
    "Sélectionne une ou plusieurs bibliothèques pour la fusion :",
    list(available_libs.keys()),
    default=list(INTEGRATED.keys())[:1]  # par défaut la première intégrée
)

# Ajout de bibliothèque personnalisée
with st.expander("➕ Ajouter ma propre chimiothèque (.sdf / .smi / .csv / .tsv)", expanded=False):
    new_name = st.text_input("Nom de la bibliothèque :", value="MaBibliotheque")
    up = st.file_uploader("Importer le fichier", type=["sdf","smi","csv","tsv"])
    if up:
        try:
            ext = os.path.splitext(up.name)[1].lower()
            mols = []
            if ext == ".sdf":
                tmp = os.path.join(LIB_PATH, "__tmp_upload.sdf")
                with open(tmp, "wb") as f: f.write(up.read())
                mols = load_sdf_file(tmp)
                os.remove(tmp)
            elif ext in [".smi"]:
                mols = load_smi_text(up.read().decode("utf-8", errors="ignore"))
            elif ext in [".csv", ".tsv"]:
                sep = "," if ext == ".csv" else "\t"
                mols = load_csv_tsv(up.read(), sep)
            mols, dups = deduplicate_mols(mols)
            if mols:
                path, safe = save_library_as_sdf(mols, new_name)
                st.session_state.user_libs[f"📁 {safe} ({len(mols)} uniques)"] = path
                st.success(f"✅ Ajoutée : {safe} — {len(mols)} uniques (doublons retirés : {dups}).")
            else:
                st.warning("Aucune molécule valide trouvée.")
        except Exception as e:
            st.error(f"Échec d'import : {e}")

# Charger & dédupliquer chaque lib sélectionnée + montrer métriques
components_per_lib = {}
merged = []
total_dups_removed = 0

if selected:
    cols = st.columns(min(3, len(selected)))
    for idx, name in enumerate(selected):
        path = available_libs[name]
        mols_raw = load_sdf_file(path) if os.path.exists(path) else []
        count_raw = len(mols_raw)
        mols, dups = deduplicate_mols(mols_raw)
        div = internal_diversity(mols, sample_pairs=600)
        components_per_lib[name] = {"count": len(mols), "dups": dups, "div": div}
        merged.extend(mols)
        total_dups_removed += dups
        with cols[idx % len(cols)]:
            st.metric(
                label=f"📚 {os.path.basename(path)}",
                value=f"{len(mols)} uniques",
                delta=f"dup: -{dups}, div: {div}"
            )

    # Fusion globale + dédup globale
    before_merge = len(merged)
    merged, fusion_dups = deduplicate_mols(merged)
    # diversité globale
    global_div = internal_diversity(merged, sample_pairs=1000)

    st.success(f"🔗 Fusion : {before_merge} → {len(merged)} uniques (doublons retirés à la fusion: {fusion_dups}).")
    st.caption(f"📏 Diversité interne de la bibliothèque fusionnée (distance Tanimoto moyenne) : {global_div}")
else:
    st.info("👉 Sélectionne au moins une bibliothèque (tu peux aussi en ajouter une personnelle).")

# --------- Bloc B : Filtres drug-likeness ----------
st.header("🧪 B. Filtres drug-likeness (optionnels)")
c1, c2, c3 = st.columns(3)
with c1:
    use_lipinski = st.checkbox("Appliquer Lipinski", value=False,
        help="MW≤500, LogP≤5, HBD≤5, HBA≤10")
with c2:
    use_veber = st.checkbox("Appliquer Veber", value=False,
        help="TPSA≤140, RotBonds≤10")
with c3:
    use_qed = st.checkbox("Filtrer par QED", value=False)
qed_thresh = st.slider("Seuil QED", 0.0, 1.0, 0.4, 0.05, disabled=not use_qed)

filtered_lib = apply_filters(merged, use_lipinski, use_veber, use_qed, qed_thresh) if selected else []
if selected:
    st.info(f"🎛️ Bibliothèque finale après filtres : {len(filtered_lib)} molécules uniques.")
    st.caption("Astuce : si la bibliothèque devient trop petite, desserre les critères.")

# --------- Bloc C : Ligands témoins ----------
st.header("🧬 C. Ligands de référence (témoins)")
ref_files = st.file_uploader("Importer témoins (.sdf ou .smi)", type=["sdf","smi"], accept_multiple_files=True)
ref_mols = []
print(ref_files)

if ref_files:
    for f in ref_files:
        if f.name.lower().endswith(".sdf"):
            tmp = os.path.join(LIB_PATH, "__tmp_ref.sdf")
            with open(tmp, "wb") as out: out.write(f.read())
            ref_mols += load_sdf_file(tmp)
            os.remove(tmp)
        else:
            text = f.read().decode("utf-8", errors="ignore")
            ref_mols += load_smi_text(text)
    ref_mols, ref_dups = deduplicate_mols(ref_mols)
    st.success(f"{len(ref_mols)} témoins uniques (doublons retirés : {ref_dups}).")

    names = [m.GetProp("_Name") for m in ref_mols]
    sel = st.selectbox("Sélectionner un témoin :", names)
    tmol = ref_mols[names.index(sel)]

    c1, c2 = st.columns([1,2])
    with c1:
        st.subheader("🔬 Structure 3D")
        show3d(tmol)
    with c2:
        st.subheader("📊 Propriétés physico-chimiques")
        df = pd.DataFrame([compute_props(tmol)])
        df["SMILES"] = Chem.MolToSmiles(tmol, isomericSmiles=True)
        df["InChIKey"] = to_inchikey(tmol) or "—"
        st.table(df)
else:
    st.info("👉 Importez des témoins pour activer la visualisation et le pharmacophore.")

# --------- Bloc D : Criblage ----------
st.header("🎯 D. Criblage ligand-based")
qry_files = st.file_uploader("Importer ligands à cribler (.sdf ou .smi)", type=["sdf","smi"], accept_multiple_files=True)
qry_mols = []
print(qry_files)
if qry_files:
    for f in qry_files:
        if f.name.lower().endswith(".sdf"):
            tmp = os.path.join(LIB_PATH, "__tmp_qry.sdf")
            with open(tmp, "wb") as out: out.write(f.read())
            qry_mols += load_sdf_file(tmp)
            os.remove(tmp)
        else:
            text = f.read().decode("utf-8", errors="ignore")
            qry_mols += load_smi_text(text)
    qry_mols, q_dups = deduplicate_mols(qry_mols)
    st.success(f"{len(qry_mols)} ligands à cribler (uniques, doublons retirés : {q_dups}).")

sim_threshold = st.slider("Seuil Tanimoto", 0.0, 1.0, 0.7, 0.05)
topk = st.number_input("Top-K hits (0 = tous)", 0, 5000, 200, 10)

if st.button("🚀 Lancer le criblage"):
    if not qry_mols or not filtered_lib:
        st.warning("Importe des ligands ET construis une bibliothèque finale (sélection + filtres).")
    else:
        # Pré-calcul FP lib
        lib_fps = [morgan_fp(m) for m in filtered_lib]
        results = []
        total = len(qry_mols) * len(filtered_lib)
        prog = st.progress(0); n = 0

        for q in qry_mols:
            q_name = q.GetProp("_Name")
            q_smi = Chem.MolToSmiles(q, isomericSmiles=True)
            q_fp = morgan_fp(q)
            hits = []
            for m, fp in zip(filtered_lib, lib_fps):
                sim = DataStructs.TanimotoSimilarity(q_fp, fp)
                if sim >= sim_threshold:
                    hits.append({
                        "Molécule requête": q_name,
                        "SMILES requête": q_smi,
                        "Molécule similaire": m.GetProp("_Name"),
                        "SMILES similaire": Chem.MolToSmiles(m, isomericSmiles=True),
                        "Formule chimique": rdMolDescriptors.CalcMolFormula(m),
                        "Similarité Tanimoto": round(sim, 3),
                        "InChIKey": to_inchikey(m) or "—"
                    })
                n += 1
                if n % max(1, total // 100) == 0:
                    prog.progress(n/total)

            if topk:
                hits.sort(key=lambda x: x["Similarité Tanimoto"], reverse=True)
                hits = hits[:topk]
            results.extend(hits)

        prog.empty()
        if results:
            df = pd.DataFrame(results).sort_values(
                ["Molécule requête", "Similarité Tanimoto"], ascending=[True, False]
            )
            st.subheader("📈 Résultats détaillés du criblage")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "💾 Télécharger (.csv)",
                df.to_csv(index=False).encode("utf-8"),
                "resultats_criblage_detaille.csv",
                "text/csv"
            )
            st.caption(f"🎯 {len(df)} lignes — {df['SMILES similaire'].nunique()} hits uniques (par SMILES).")
        else:
            st.info("Aucun hit au-dessus du seuil défini.")

# --------- Bloc E : Pharmacophore ----------
st.header("🧩 E. Pharmacophore 3D des témoins")
if ref_mols:
    mode = st.radio(
        "Mode d’affichage",
        ["💡 Pharmacophore seul", "🔬 Superposé sur les molécules de référence"],
        horizontal=True
    )
    if st.button("🎥 Générer l'affichage pharmacophore"):
        pts = pharmacophore_points(ref_mols)
        if pts:
            show_pharmacophore(
                pts,
                ref_mols=ref_mols,
                superpose=(mode == "🔬 Superposé sur les molécules de référence")
            )
            st.download_button(
                "💾 Télécharger le pharmacophore (.json)",
                json.dumps(pts, indent=2),
                "pharmacophore.json",
                "application/json"
            )
else:
    st.info("👉 Importez des témoins pour générer le pharmacophore.")
