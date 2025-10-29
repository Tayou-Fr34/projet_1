# -*- coding: utf-8 -*-
import os
import zipfile
import tempfile
from io import StringIO
from pathlib import Path
import subprocess
import shutil

import numpy as np
import pandas as pd
import streamlit as st
import py3Dmol

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors as rdmd
from rdkit.Chem import AllChem

from Bio.PDB import PDBParser
from scipy.spatial import cKDTree



def _find_vina() -> list[str]:
    """Retourne la commande à exécuter pour Vina, multi-plateforme."""
    import os, shutil
    # 1) variable d’environnement VINA_BIN
    vina_env = os.environ.get("VINA_BIN")
    if vina_env and os.path.exists(vina_env):
        return [vina_env]
    # 2) via PATH
    for name in ("vina", "vina.exe"):
        if shutil.which(name):
            return [name]
    # 3) chemins Windows courants
    for c in (r"C:\vina\vina.exe",
              r"C:\vina\vina_1.2.3_windows_x86_64.exe",
              r"C:\Program Files\Vina\vina.exe"):
        if os.path.exists(c):
            return [c]
    raise RuntimeError(
        "AutoDock Vina introuvable.\n"
        "Définis VINA_BIN (ex: C:\\vina\\vina.exe) ou ajoute C:\\vina au PATH."
    )



# =========================
# ====== UTILITAIRES ======
# =========================
def get_ligand_coords(mol: Chem.Mol) -> np.ndarray:
    """Retourne les coordonnées (N,3) du premier conformère du ligand RDKit."""
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError("Le ligand n'a pas de conformère 3D.")
    conf = mol.GetConformer(0)
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)


def get_protein_coords_and_text(uploaded_protein) -> tuple[np.ndarray, str]:
    """
    Lit un fichier PDB uploadé via Streamlit et renvoie (coords (M,3), pdb_text).
    """
    pdb_bytes = uploaded_protein.getvalue()
    pdb_text = pdb_bytes.decode("utf-8")
    pdb_stream = StringIO(pdb_text)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_stream)

    coords = [atom.coord for atom in structure.get_atoms()]
    if not coords:
        raise ValueError("Aucun atome détecté dans le PDB.")
    return np.array(coords, dtype=float), pdb_text


def distance_stats(prot_xyz: np.ndarray, lig_xyz: np.ndarray,
                   contact_cutoff: float = 4.0, clash_cutoff: float = 2.5) -> tuple[float, float, int, int]:
    """
    Calcule:
      - distance minimale protéine-ligand
      - distance moyenne (nearest-neighbor des atomes ligand -> protéine)
      - nombre de contacts (< contact_cutoff Å)
      - nombre de clashs   (< clash_cutoff Å)
    Implémentation rapide via KDTree.
    """
    if prot_xyz.size == 0 or lig_xyz.size == 0:
        raise ValueError("Coordonnées vides pour la protéine ou le ligand.")

    tree = cKDTree(prot_xyz)

    d_nn, _ = tree.query(lig_xyz, k=1)
    min_d = float(d_nn.min())
    mean_d = float(d_nn.mean())

    n_contacts = int(sum(tree.query_ball_point(lig_xyz, r=contact_cutoff, return_length=True)))
    n_clashs   = int(sum(tree.query_ball_point(lig_xyz, r=clash_cutoff,   return_length=True)))

    return min_d, mean_d, n_contacts, n_clashs


def interpret_affinity(min_dist: float) -> str:
    if min_dist < 3.0:
        return "🟢 Forte affinité potentielle (proximité élevée)"
    elif min_dist < 6.0:
        return "🟡 Affinité moyenne (proximité modérée)"
    else:
        return "🔴 Faible affinité (proximité faible)"


def visualize(protein_pdb: str, ligand_mol: Chem.Mol | None) -> str:
    """Renvoie un fragment HTML avec py3Dmol (protéine cartoon + ligand en sticks)."""
    view = py3Dmol.view(width=600, height=500)
    view.addModel(protein_pdb, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addSurface(py3Dmol.VDW, {"opacity": 0.15, "color": "white"})

    if ligand_mol is not None:
        mb = Chem.MolToMolBlock(ligand_mol)
        view.addModel(mb, "mol")
        view.setStyle({"model": 1}, {"stick": {"colorscheme": "redCarbon"}})

    view.zoomTo()
    return view._make_html()


def mol_properties(mol: Chem.Mol) -> dict:
    """Propriétés classiques RDKit."""
    return {
        "MolWt": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Crippen.MolLogP(mol), 2),
        "tPSA": round(rdmd.CalcTPSA(mol), 2),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "RotBonds": int(Lipinski.NumRotatableBonds(mol)),
        "Rings": int(rdmd.CalcNumRings(mol)),
    }


def lipinski_flags(props: dict) -> str:
    ok = (
        props["MolWt"] <= 500 and
        props["LogP"] <= 5 and
        props["HBA"]  <= 10 and
        props["HBD"]  <= 5
    )
    return "✅ Conforme aux règles de Lipinski" if ok else "⚠️ Hors borne(s) (Lipinski)"


def center_molecules(protein_coords: np.ndarray, ligand_mol: Chem.Mol) -> Chem.Mol:
    """Translate le ligand pour le centrer approximativement sur le centre géométrique de la protéine."""
    protein_center = protein_coords.mean(axis=0)
    conf = ligand_mol.GetConformer(0)
    ligand_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(ligand_mol.GetNumAtoms())], dtype=float)
    ligand_center = ligand_coords.mean(axis=0)

    translation = protein_center - ligand_center
    for i in range(ligand_mol.GetNumAtoms()):
        x, y, z = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (x + translation[0], y + translation[1], z + translation[2]))
    return ligand_mol


def load_mol2_paths_from_zip(uploaded_zip) -> list[str]:
    """Extrait un ZIP dans un dossier temporaire et retourne la liste des chemins .mol2 (récursif)."""
    paths = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "ligands.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getvalue())
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        for root, _, files in os.walk(tmp_dir):
            for fn in files:
                if fn.lower().endswith(".mol2") and not fn.startswith("._") and "__MACOSX" not in root:
                    paths.append(os.path.join(root, fn))
    return paths


def smiles_to_3d(smiles: str) -> Chem.Mol | None:
    """Crée un ligand 3D à partir d'un SMILES (ETKDG + optimisation UFF/MMFF)."""
    if not smiles:
        return None
    m2d = Chem.MolFromSmiles(smiles)
    if m2d is None:
        return None
    m3d = Chem.AddHs(m2d)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xf00d
    if AllChem.EmbedMolecule(m3d, params) != 0:
        if AllChem.EmbedMolecule(m3d) != 0:
            return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(m3d):
            AllChem.MMFFOptimizeMolecule(m3d, maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(m3d, maxIters=200)
    except Exception:
        pass
    return m3d


# =========================
# === Intégration Vina ====
# =========================
def _which(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def _run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)

def receptor_to_pdbqt(pdb_text: str, tmpdir: str) -> str:
    """
    Convertit un PDB texte -> PDBQT (récepteur) via OpenBabel.
    - Ajoute H, charges de Gasteiger, enlève solvant.
    """
    pdb_path = Path(tmpdir) / "receptor.pdb"
    pdbqt_path = Path(tmpdir) / "receptor.pdbqt"
    pdb_path.write_text(pdb_text)

    if not _which("obabel"):
        raise RuntimeError("OpenBabel (obabel) n'est pas disponible dans l'environnement.")

    cmd = ["obabel", "-ipdb", str(pdb_path), "-opdbqt", "-O", str(pdbqt_path),
           "-h", "--partialcharge", "gasteiger", "-xr"]
    _run(cmd)
    if not pdbqt_path.exists():
        raise RuntimeError("Échec conversion récepteur en PDBQT.")
    return str(pdbqt_path)

def ligand_to_pdbqt_with_meeko(mol: Chem.Mol, tmpdir: str) -> str:
    """
    Prépare un ligand RDKit -> PDBQT via Meeko (types, charges, etc.).
    """
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except Exception as e:
        raise RuntimeError("Meeko n'est pas installé correctement.") from e

    prep = MoleculePreparation()
    mk_mol = prep.prepare(mol)
    writer = PDBQTWriterLegacy()
    pdbqt_str = writer.write_string(mk_mol)

    out_path = Path(tmpdir) / "ligand.pdbqt"
    Path(out_path).write_text(pdbqt_str)
    return str(out_path)

def pdbqt_to_sdf(pdbqt_path: str, out_sdf: str):
    """Convertit PDBQT (poses Vina) -> SDF via OpenBabel."""
    if not _which("obabel"):
        raise RuntimeError("OpenBabel (obabel) n'est pas disponible.")
    cmd = ["obabel", "-ipdbqt", pdbqt_path, "-osdf", "-O", out_sdf, "-d"]
    _run(cmd)
    if not Path(out_sdf).exists():
        raise RuntimeError("Échec conversion PDBQT -> SDF.")

def read_first_sdf_mol(sdf_path: str) -> Chem.Mol | None:
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    for m in suppl:
        if m is not None:
            return m
    return None

def run_vina(receptor_pdbqt, ligand_pdbqt, center, size, exhaustiveness, tmpdir):
    cmd = _find_vina()  # <-- au lieu de tester _which("vina")
    out_pdbqt = Path(tmpdir) / "out_vina.pdbqt"
    log_path  = Path(tmpdir) / "vina.log"
    cx, cy, cz = center; sx, sy, sz = size
    cmd += [
        "--receptor", receptor_pdbqt, "--ligand", ligand_pdbqt,
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
        "--exhaustiveness", str(exhaustiveness),
        "--out", str(out_pdbqt), "--log", str(log_path)
    ]
    _run(cmd)

    scores = []
    if out_pdbqt.exists():
        for line in Path(out_pdbqt).read_text().splitlines():
            if "REMARK VINA RESULT:" in line:
                parts = line.split()
                # format typique: REMARK VINA RESULT:    -7.5      0.0      0.0
                for tok in parts:
                    try:
                        scores.append(float(tok))
                        break
                    except ValueError:
                        continue
    if not scores and log_path.exists():
        for line in Path(log_path).read_text().splitlines():
            if "REMARK VINA RESULT:" in line:
                parts = line.split()
                for tok in parts:
                    try:
                        scores.append(float(tok))
                        break
                    except ValueError:
                        continue

    if not out_pdbqt.exists():
        raise RuntimeError("Vina n'a pas produit de pose (out_vina.pdbqt manquant).")
    return str(out_pdbqt), scores


# =========================
# ======   STREAMLIT  =====
# =========================
st.set_page_config(page_title="Docking simplifié", page_icon="🧬", layout="wide")
st.title("🧬 Simulation de docking **simplifiée** (visualisation 3D)")
st.caption("ℹ️ Heuristiques de proximité + option AutoDock Vina. (Ce n’est pas un outil de validation expérimentale.)")

# ---- Entrées (sidebar) ----
with st.sidebar:
    st.header("Paramètres")
    uploaded_protein = st.file_uploader("📂 Protéine (.pdb)", type=["pdb"])
    uploaded_zip     = st.file_uploader("💊 ZIP de ligands (.mol2)", type=["zip"])
    smiles           = st.text_input("Ou SMILES (fallback)", "CCO")
    contact_cutoff   = st.slider("Seuil contacts (Å)", 3.0, 6.0, 4.0, 0.1)
    clash_cutoff     = st.slider("Seuil clashs (Å)",   1.8, 3.0, 2.5, 0.1)
    vina_bin = os.environ.get("VINA_BIN")
    st.sidebar.write("VINA_BIN:", vina_bin or "(non défini)")
    
    st.markdown("---")
    st.subheader("Docking (AutoDock Vina)")
    use_vina = st.checkbox("Activer Vina (docke le ligand)", value=False)
    default_size = (20.0, 20.0, 20.0)
    exhaust = st.number_input("Exhaustiveness", min_value=4, max_value=64, value=8, step=1)

    cx = st.number_input("Center X", value=0.0, step=0.5, format="%.2f")
    cy = st.number_input("Center Y", value=0.0, step=0.5, format="%.2f")
    cz = st.number_input("Center Z", value=0.0, step=0.5, format="%.2f")
    sx = st.number_input("Size X (Å)", min_value=8.0, max_value=60.0, value=default_size[0], step=1.0)
    sy = st.number_input("Size Y (Å)", min_value=8.0, max_value=60.0, value=default_size[1], step=1.0)
    sz = st.number_input("Size Z (Å)", min_value=8.0, max_value=60.0, value=default_size[2], step=1.0)

# Toujours définir la variable pour éviter NameError
ligand_mol = None
ligand_name_for_display = None

# ---- Chargement ligands depuis ZIP (si fourni) ----
if uploaded_zip is not None:
    try:
        ligand_paths = load_mol2_paths_from_zip(uploaded_zip)
        if ligand_paths:
            st.sidebar.success(f"{len(ligand_paths)} fichiers .mol2 détectés ✅")
            display_names = [os.path.basename(p) for p in ligand_paths]
            idx = st.sidebar.selectbox("Choisissez un ligand (.mol2) :", range(len(ligand_paths)),
                                       format_func=lambda i: display_names[i])
            chosen_path = ligand_paths[idx]
            ligand_name_for_display = os.path.basename(chosen_path)

            # Ignorer poliment les .mol2 illisibles
            try:
                ligand_mol = Chem.MolFromMol2File(chosen_path, sanitize=True)
                if ligand_mol is None:
                    raise ValueError("Lecture échouée")
                st.sidebar.info(f"Ligand **{ligand_name_for_display}** chargé.")
            except Exception:
                st.sidebar.warning(f"⚠️ Ligand illisible : {ligand_name_for_display}. Essayez-en un autre.")
                ligand_mol = None
        else:
            st.sidebar.error("❌ Aucun fichier `.mol2` utile détecté dans le ZIP.")
    except Exception as e:
        st.sidebar.error(f"Erreur de lecture du ZIP : {e}")

# ---- Fallback SMILES → 3D si pas de .mol2 utilisable ----
if ligand_mol is None and smiles:
    sm_mol = smiles_to_3d(smiles)
    if sm_mol is not None:
        ligand_mol = sm_mol
        ligand_name_for_display = f"SMILES: {smiles}"
        st.sidebar.info("Ligand généré à partir du SMILES (3D).")
    else:
        st.sidebar.warning("SMILES invalide ou génération 3D impossible.")

# ---- Logique principale ----
if uploaded_protein:
    try:
        protein_coords, pdb_text = get_protein_coords_and_text(uploaded_protein)
    except Exception as e:
        st.error(f"❌ Erreur lors du parsing PDB : {e}")
        st.stop()

    if ligand_mol:
        try:
            # Centrer ligand ~ protéine
            ligand_mol = center_molecules(protein_coords, ligand_mol)
            ligand_coords = get_ligand_coords(ligand_mol)

            # Stats de distances
            min_d, mean_d, n_contacts, n_clashs = distance_stats(
                protein_coords, ligand_coords,
                contact_cutoff=float(contact_cutoff),
                clash_cutoff=float(clash_cutoff)
            )
            score = interpret_affinity(min_d)
            props = mol_properties(ligand_mol)

            # ===== Docking Vina (optionnel) =====
            vina_scores = None
            docked_mol = None

            if use_vina:
                with st.spinner("Docking avec Vina en cours..."):
                    try:
                        with tempfile.TemporaryDirectory() as tdir:
                            # 1) Récepteur -> PDBQT
                            receptor_pdbqt = receptor_to_pdbqt(pdb_text, tdir)
                            # 2) Ligand -> PDBQT (RDKit -> Meeko)
                            ligand_pdbqt = ligand_to_pdbqt_with_meeko(ligand_mol, tdir)

                            # 3) Centre auto si (0,0,0) : centroïde protéine
                            center = (cx, cy, cz)
                            if center == (0.0, 0.0, 0.0):
                                c = protein_coords.mean(axis=0)
                                center = (float(c[0]), float(c[1]), float(c[2]))

                            size = (sx, sy, sz)

                            # 4) Lancer Vina
                            out_pdbqt, vina_scores = run_vina(
                                receptor_pdbqt, ligand_pdbqt, center, size, int(exhaust), tdir
                            )

                            # 5) Convertir meilleur pose en SDF puis RDKit
                            out_sdf = Path(tdir) / "top_pose.sdf"
                            pdbqt_to_sdf(out_pdbqt, str(out_sdf))
                            docked_mol = read_first_sdf_mol(str(out_sdf))

                            if docked_mol is None:
                                st.warning("Docking terminé, mais impossible de lire la pose SDF.")
                            else:
                                st.success("Docking Vina terminé ✅")
                    except Exception as e:
                        st.error(f"❌ Échec du docking Vina : {e}")

            # ---- Affichage ----
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🔬 Visualisation 3D")
                mol_to_show = docked_mol if (use_vina and docked_mol is not None) else ligand_mol
                st.components.v1.html(visualize(pdb_text, mol_to_show), height=520)

            with col2:
                st.subheader("📊 Résultats")
                if ligand_name_for_display:
                    st.write(f"**Ligand :** {ligand_name_for_display}")
                st.write(f"**Distance minimale :** {min_d:.2f} Å")
                st.write(f"**Distance moyenne (NN) :** {mean_d:.2f} Å")
                st.write(f"**Contacts (< {contact_cutoff:.1f} Å) :** {n_contacts}")
                st.write(f"**Clashs (< {clash_cutoff:.1f} Å) :** {n_clashs}")
                st.info(f"**Interprétation (heuristique) :** {score}")

                st.write("**Propriétés RDKit**")
                st.write(pd.DataFrame([props]))
                st.info(lipinski_flags(props))

                if use_vina and vina_scores:
                    best = min(vina_scores)
                    st.write(f"**Vina score (meilleur, kcal/mol) :** {best:.2f}")
                    st.caption("Plus le score est négatif, meilleure est l'affinité prévue (selon Vina).")

            st.caption("ℹ️ Cette application illustre des heuristiques de proximité et permet un docking Vina rapide. "
                       "Pour des études robustes : préparer soigneusement le site, la protonation, et contrôler les paramètres.")

        except Exception as e:
            st.error(f"❌ Erreur durant l'analyse du ligand : {e}")
    else:
        st.error("❌ SMILES ou Ligand invalide, génération 3D impossible.")
else:
    st.warning("📂 Chargez une **protéine (.pdb)** pour commencer. "
               "Puis chargez un **ZIP de ligands (.mol2)** ou saisissez un **SMILES**.")
