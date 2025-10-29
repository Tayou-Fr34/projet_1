# -*- coding: utf-8 -*-
import os, io, zipfile, tempfile, subprocess, shutil
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import py3Dmol

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors as rdmd
from rdkit.Chem import AllChem

from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

from pathlib import Path
import shutil, os



from pathlib import Path
import os, shutil

# --- VINA LOCATOR (unique) ---
import os, shutil
from pathlib import Path



# --- helper: teste si une option est supportée par ce vina ---
_vina_help_cache = None
def _vina_supports(flag: str, vina_cmd_quoted: str) -> bool:
    global _vina_help_cache
    if _vina_help_cache is None:
        # on lit l'aide une seule fois
        code, out, err = _run(f'{vina_cmd_quoted} --help')
        _vina_help_cache = (out or "") + "\n" + (err or "")
    return (flag in _vina_help_cache)

def vina_run_and_extract_best_pose(receptor_pdbqt: str, ligand_pdbqt: str,
                                   center: tuple[float,float,float], size: tuple[float,float,float],
                                   exhaustiveness: int, num_modes: int, tmpdir: str) -> tuple[str | None, list[float]]:
    """Lance Vina et tente de récupérer un SDF de la meilleure pose (compat vieux Vina sans --log)."""
    vina_cmd = _find_vina()  # <- renvoie un chemin déjà entre guillemets
    out_pdbqt = str(Path(tmpdir) / "out_vina.pdbqt")
    log_path  = str(Path(tmpdir) / "vina.log")

    cx, cy, cz = center; sx, sy, sz = size

    # Construit la commande, en ajoutant --log seulement si supporté
    cmd = (
        f'{vina_cmd} '
        f'--receptor "{receptor_pdbqt}" --ligand "{ligand_pdbqt}" '
        f'--center_x {cx:.3f} --center_y {cy:.3f} --center_z {cz:.3f} '
        f'--size_x {sx:.3f} --size_y {sy:.3f} --size_z {sz:.3f} '
        f'--exhaustiveness {exhaustiveness} --num_modes {num_modes} '
        f'--out "{out_pdbqt}"'
    )
    if _vina_supports("--log", vina_cmd):
        cmd += f' --log "{log_path}"'

    code, vout, verr = _run(cmd)
    if code != 0 or (not os.path.exists(out_pdbqt)):
        # certaines vieilles versions ne mettent rien dans stderr → montre stdout si utile
        raise RuntimeError(f"Vina a échoué.\n{verr or vout or '(pas de sortie)'}")

    # --- Parse des scores : d’abord dans le PDBQT (standard), sinon dans stdout/--log si présent ---
    scores = []
    try:
        with open(out_pdbqt, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if "REMARK VINA RESULT:" in line:
                    parts = line.split()
                    for tok in parts:
                        try:
                            scores.append(float(tok)); break
                        except ValueError:
                            continue
    except Exception:
        pass

    if not scores:
        # si --log supporté et fichier créé
        if _vina_supports("--log", vina_cmd) and os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if "REMARK VINA RESULT:" in line:
                        parts = line.split()
                        for tok in parts:
                            try:
                                scores.append(float(tok)); break
                            except ValueError:
                                continue
        else:
            # fallback vieux Vina : parfois les résultats sont sortis sur stdout
            for line in (vout or "").splitlines():
                if "REMARK VINA RESULT:" in line:
                    parts = line.split()
                    for tok in parts:
                        try:
                            scores.append(float(tok)); break
                        except ValueError:
                            continue

    # --- Extraction de la 1re pose en SDF (identique à avant) ---
    best_sdf = str(Path(tmpdir) / "best.sdf")
    vina_split = _find_vina_split(vina_cmd)
    if vina_split:
        code_s, out_s, err_s = _run(f'{vina_split} --input "{out_pdbqt}" --ligand', cwd=tmpdir)
        first_pose = os.path.join(tmpdir, "ligand_1.pdbqt")
        if code_s == 0 and os.path.exists(first_pose):
            code_b, out_b, err_b = _run(f'obabel -ipdbqt "{first_pose}" -osdf -O "{best_sdf}" -d')
            if code_b != 0 or (not os.path.exists(best_sdf)):
                best_sdf = None
        else:
            code_b, out_b, err_b = _run(f'obabel -ipdbqt "{out_pdbqt}" -osdf -O "{best_sdf}" -f 1 -l 1 -d')
            if code_b != 0 or (not os.path.exists(best_sdf)):
                best_sdf = None
    else:
        code_b, out_b, err_b = _run(f'obabel -ipdbqt "{out_pdbqt}" -osdf -O "{best_sdf}" -f 1 -l 1 -d')
        if code_b != 0 or (not os.path.exists(best_sdf)):
            tmp_pdb = str(Path(tmpdir) / "best.pdb")
            code_p, out_p, err_p = _run(f'obabel -ipdbqt "{out_pdbqt}" -opdb -O "{tmp_pdb}" -f 1 -l 1')
            if code_p == 0 and os.path.exists(tmp_pdb) and os.path.getsize(tmp_pdb) > 0:
                code_s, out_s, err_s = _run(f'obabel -ipdb "{tmp_pdb}" -osdf -O "{best_sdf}" -d')
                if code_s != 0 or (not os.path.exists(best_sdf)):
                    best_sdf = None
            else:
                best_sdf = None

    return best_sdf, scores







def _find_vina() -> str:
    """
    Retourne la commande Vina entre guillemets, en testant d'abord le chemin câblé.
    Lève RuntimeError si rien n'est trouvé.
    """
    # 0) chemin câblé prioritaire
    candidates = [
        r"C:\vina\vina.exe",                         # ton binaire
        r"C:\vina\vina_1.2.3_windows_x86_64.exe",    # autre nom fréquent
        os.environ.get("VINA_BIN", "").strip('"')    # variable d'env si définie
    ]

    for c in candidates:
        if c:
            p = Path(c)
            if p.is_file():
                return f'"{str(p)}"'

    # 1) via PATH
    for name in ("vina.exe", "vina"):
        p = shutil.which(name)
        if p:
            return f'"{p}"'

    # 2) derniers chemins “classiques”
    for c in (r"C:\Program Files\Vina\vina.exe",):
        if Path(c).is_file():
            return f'"{c}"'

    raise RuntimeError("AutoDock Vina introuvable. Vérifie C:\\vina\\vina.exe ou VINA_BIN / PATH.")


# =============== Exec helpers ===============
def _which(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def _run(cmd: str | list[str], cwd: str | None = None):
    """Exécute une commande shell, renvoie (code, stdout, stderr)."""
    if isinstance(cmd, list):
        cmd = " ".join(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, shell=True)
    out, err = p.communicate()
    return p.returncode, out.decode("utf-8", "ignore"), err.decode("utf-8", "ignore")



def _find_vina_split(vina_cmd_str: str) -> str | None:
    """Essaie de localiser vina_split à côté du binaire de vina."""
    raw = vina_cmd_str.strip().strip('"')
    folder = os.path.dirname(raw) if os.path.isfile(raw) else None
    if folder:
        cand = os.path.join(folder, "vina_split.exe" if os.name == "nt" else "vina_split")
        if os.path.exists(cand):
            return f'"{cand}"'
    # Sinon via PATH
    for n in ("vina_split", "vina_split.exe"):
        p = shutil.which(n)
        if p:
            return f'"{p}"'
    return None


# =============== Chimie utils ===============
def get_ligand_coords(mol: Chem.Mol) -> np.ndarray:
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError("Le ligand n'a pas de conformère 3D.")
    conf = mol.GetConformer(0)
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)

def get_protein_coords_and_text(uploaded_protein) -> tuple[np.ndarray, str]:
    pdb_bytes = uploaded_protein.getvalue()
    pdb_text = pdb_bytes.decode("utf-8", "ignore")
    pdb_stream = StringIO(pdb_text)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_stream)
    coords = [atom.coord for atom in structure.get_atoms()]
    if not coords:
        raise ValueError("Aucun atome détecté dans le PDB.")
    return np.array(coords, dtype=float), pdb_text

def distance_stats(prot_xyz: np.ndarray, lig_xyz: np.ndarray,
                   contact_cutoff: float = 4.0, clash_cutoff: float = 2.5):
    from scipy.spatial import cKDTree
    tree = cKDTree(prot_xyz)
    d_nn, _ = tree.query(lig_xyz, k=1)
    min_d = float(d_nn.min())
    mean_d = float(d_nn.mean())
    n_contacts = int(sum(tree.query_ball_point(lig_xyz, r=contact_cutoff, return_length=True)))
    n_clashs   = int(sum(tree.query_ball_point(lig_xyz, r=clash_cutoff,   return_length=True)))
    return min_d, mean_d, n_contacts, n_clashs

def interpret_affinity(min_dist: float) -> str:
    if min_dist < 3.0: return "🟢 Forte affinité potentielle (proximité élevée)"
    if min_dist < 6.0: return "🟡 Affinité moyenne (proximité modérée)"
    return "🔴 Faible affinité (proximité faible)"

def visualize(protein_pdb: str, ligand_mol: Chem.Mol | None) -> str:
    view = py3Dmol.view(width=700, height=520)
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
    ok = (props["MolWt"] <= 500 and props["LogP"] <= 5 and props["HBA"] <= 10 and props["HBD"] <= 5)
    return "✅ Conforme aux règles de Lipinski" if ok else "⚠️ Hors borne(s) (Lipinski)"

def center_molecules(protein_coords: np.ndarray, ligand_mol: Chem.Mol) -> Chem.Mol:
    protein_center = protein_coords.mean(axis=0)
    conf = ligand_mol.GetConformer(0)
    ligand_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(ligand_mol.GetNumAtoms())], dtype=float)
    ligand_center = ligand_coords.mean(axis=0)
    t = protein_center - ligand_center
    for i in range(ligand_mol.GetNumAtoms()):
        x, y, z = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (x + t[0], y + t[1], z + t[2]))
    return ligand_mol

def load_mol2_paths_from_zip(uploaded_zip) -> list[str]:
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
    if not smiles: return None
    m2d = Chem.MolFromSmiles(smiles)
    if m2d is None: return None
    m3d = Chem.AddHs(m2d)
    params = AllChem.ETKDGv3(); params.randomSeed = 0xF00D
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


# =============== Conversions PDB/PDBQT/SDF ===============
def receptor_to_pdbqt(pdb_text: str, tmpdir: str) -> str:
    """PDB (texte) -> PDBQT via OpenBabel (-h charges, -xr retire eau)."""
    if not _which("obabel"):
        raise RuntimeError("OpenBabel (obabel) n'est pas disponible dans l'environnement.")
    pdb_path = Path(tmpdir) / "receptor.pdb"
    pdbqt_path = Path(tmpdir) / "receptor.pdbqt"
    pdb_path.write_text(pdb_text)
    code, out, err = _run(["obabel", "-ipdb", str(pdb_path), "-opdbqt", "-O", str(pdbqt_path), "-h", "--partialcharge", "gasteiger", "-xr"])
    if code != 0 or (not pdbqt_path.exists()):
        raise RuntimeError(f"Échec conversion récepteur en PDBQT.\n{err}")
    return str(pdbqt_path)

def ligand_to_pdbqt_smart(mol: Chem.Mol, tmpdir: str) -> str:
    """
    RDKit Mol -> PDBQT : tente Meeko, sinon fallback OpenBabel via SDF.
    """
    # Essai Meeko
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        prep = MoleculePreparation()
        mk_mol = prep.prepare(mol)
        pdbqt_str = PDBQTWriterLegacy().write_string(mk_mol)
        out = Path(tmpdir) / "ligand.pdbqt"
        out.write_text(pdbqt_str)
        return str(out)
    except Exception as e:
        # Fallback OpenBabel
        if not _which("obabel"):
            raise RuntimeError("Meeko indisponible ET OpenBabel absent : impossible de produire un PDBQT.") from e
        # on écrit une SDF temporaire avec RDKit puis on convertit en PDBQT
        sdf = Path(tmpdir) / "ligand.sdf"
        w = Chem.SDWriter(str(sdf))
        w.write(mol); w.close()
        out = Path(tmpdir) / "ligand.pdbqt"
        code, outmsg, errmsg = _run(["obabel", "-isdf", str(sdf), "-opdbqt", "-O", str(out), "--gen3d"])
        if code != 0 or (not out.exists()):
            raise RuntimeError(f"Échec conversion ligand → PDBQT via OpenBabel.\n{errmsg}")
        return str(out)


def read_mol2_from_upload(uploaded_file):
    if uploaded_file is None or uploaded_file.size == 0:
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mol2") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        mol = Chem.MolFromMol2File(tmp_path, sanitize=True)
        if mol is None:
            # 2e tentative (certains .mol2 passent sans sanitize) :
            mol = Chem.MolFromMol2File(tmp_path, sanitize=False)
            if mol is not None:
                Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        st.error(f"Erreur lecture .mol2 : {e}")
        return None
    


# =============== STREAMLIT UI ===============
st.set_page_config(page_title="Docking simplifié (Vina obligatoire)", page_icon="🧬", layout="wide")
st.title("🧬 Docking **obligatoire** avec AutoDock Vina")
st.caption("Prépare le récepteur/ligand, calcule une box, lance Vina et affiche la meilleure pose. OpenBabel requis.")

with st.sidebar:
    st.header("Paramètres")
    uploaded_protein = st.file_uploader("📂 Protéine (.pdb)", type=["pdb"])
    uploaded_ligand  = st.file_uploader("💊 Charger un ligand candidat (.mol2)", type=["mol2"])
    smiles           = st.text_input("Ou SMILES (fallback)", "CCO")
    contact_cutoff   = st.slider("Seuil contacts (Å)", 3.0, 6.0, 4.0, 0.1)
    clash_cutoff     = st.slider("Seuil clashs (Å)",   1.8, 3.0, 2.5, 0.1)

    st.markdown("---")
    st.subheader("Vina")
    exhaust = st.number_input("Exhaustiveness", min_value=4, max_value=64, value=8, step=1)
    num_modes = st.number_input("Nombre de poses", min_value=1, max_value=20, value=9, step=1)

import sys

with st.sidebar:
    st.markdown("### Debug Vina")
    hard = r"C:\vina\vina.exe"
    st.write("Python exe:", sys.executable)
    st.write("Hardcoded path:", hard)
    st.write("os.path.exists:", os.path.exists(hard))
    try:
        st.write("Path.exists:", Path(hard).exists())
    except Exception as e:
        st.write("Path.exists err:", e)
    try:
        st.write("C:\\vina listing:", os.listdir(r"C:\vina"))
    except Exception as e:
        st.write("listdir err:", e)

    try:
        vina_cmd = _find_vina()
        st.success(f"Vina détecté : {vina_cmd}")
    except Exception as e:
        st.error(f"_find_vina() a échoué : {e}")

    st.write("VINA_BIN env:", os.environ.get("VINA_BIN", "(non défini)"))
    st.write("which(vina):", shutil.which("vina"))
    st.write("which(vina.exe):", shutil.which("vina.exe"))


    st.markdown("---")
    st.subheader("Boîte (center/size)")
    cx = st.number_input("Center X", value=0.0, step=0.5, format="%.2f")
    cy = st.number_input("Center Y", value=0.0, step=0.5, format="%.2f")
    cz = st.number_input("Center Z", value=0.0, step=0.5, format="%.2f")
    sx = st.number_input("Size X (Å)", min_value=8.0, max_value=60.0, value=20.0, step=1.0)
    sy = st.number_input("Size Y (Å)", min_value=8.0, max_value=60.0, value=20.0, step=1.0)
    sz = st.number_input("Size Z (Å)", min_value=8.0, max_value=60.0, value=20.0, step=1.0)

run_btn = st.button("🚀 Lancer le docking (Vina)")

# ---- Chargement ligand depuis MOL2 ou SMILES ----
ligand_mol = None
ligand_name_for_display = None



ligand_mol = None
if uploaded_ligand is not None and uploaded_ligand.size > 0:
    ligand_mol = read_mol2_from_upload(uploaded_ligand)


if ligand_mol is None and (smiles or "").strip():
    sm_mol = smiles_to_3d(smiles.strip())
    if sm_mol is not None:
        ligand_mol = sm_mol
        ligand_name_for_display = f"SMILES: {smiles}"
        st.sidebar.info("Ligand généré à partir du SMILES (3D).")
    else:
        st.sidebar.warning("SMILES invalide ou génération 3D impossible.")

# ---- Pipeline principal (Vina obligatoire) ----
if run_btn:
    if uploaded_protein is None:
        st.error("Téléverse une protéine .pdb"); st.stop()
    if ligand_mol is None:
        st.error("Fournis un ligand (.mol2 dans ZIP) ou un SMILES."); st.stop()

    try:
        protein_coords, pdb_text = get_protein_coords_and_text(uploaded_protein)
    except Exception as e:
        st.error(f"❌ Erreur parsing PDB : {e}"); st.stop()

    # centrage ligand pour l’affichage “avant docking”
    try:
        ligand_mol = center_molecules(protein_coords, ligand_mol)
        ligand_coords = get_ligand_coords(ligand_mol)
        min_d, mean_d, n_contacts, n_clashs = distance_stats(protein_coords, ligand_coords)
        props = mol_properties(ligand_mol)
    except Exception as e:
        st.warning(f"Analyse pré-docking partielle : {e}")

    with st.spinner("Docking avec Vina…"):
        with tempfile.TemporaryDirectory() as tdir:
            # récepteur → PDBQT
            receptor_pdbqt = receptor_to_pdbqt(pdb_text, tdir)
            # ligand → PDBQT (Meeko → fallback OB)
            ligand_pdbqt   = ligand_to_pdbqt_smart(ligand_mol, tdir)

            # Boîte : si (0,0,0) → centroïde protéine
            center = (cx, cy, cz)
            if center == (0.0, 0.0, 0.0):
                c = protein_coords.mean(axis=0)
                center = (float(c[0]), float(c[1]), float(c[2]))
            size = (sx, sy, sz)

            best_sdf, vina_scores = vina_run_and_extract_best_pose(
                receptor_pdbqt, ligand_pdbqt, center, size, int(exhaust), int(num_modes), tdir
            )

            # Lecture RDKit de la meilleure pose
            docked_mol = None
            if best_sdf and os.path.exists(best_sdf) and os.path.getsize(best_sdf) > 0:
                try:
                    suppl = Chem.SDMolSupplier(best_sdf, removeHs=False, sanitize=False, strictParsing=False)
                    for m in suppl:
                        if m is not None:
                            docked_mol = m; break
                except Exception as e:
                    st.warning(f"SDF généré mais lecture RDKit impossible : {e}")

    # ---- Affichage ----
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔬 Visualisation 3D (meilleure pose Vina)")
        mol_to_show = docked_mol if docked_mol is not None else ligand_mol
        st.components.v1.html(visualize(pdb_text, mol_to_show), height=560)

    with col2:
        st.subheader("📊 Résultats")
        if ligand_name_for_display:
            st.write(f"**Ligand :** {ligand_name_for_display}")
        try:
            st.write(f"**Distance min (pré-docking) :** {min_d:.2f} Å")
            st.write(f"**Distance moy (pré-docking) :** {mean_d:.2f} Å")
            st.write(f"**Contacts (<4 Å) :** {n_contacts} / **Clashs (<2.5 Å) :** {n_clashs}")
        except Exception:
            pass
        try:
            st.write("**Propriétés RDKit**"); st.write(pd.DataFrame([props]))
            st.info(lipinski_flags(props))
        except Exception:
            pass
        if vina_scores:
            best = min(vina_scores)
            st.write(f"**Vina score (meilleur, kcal/mol) :** {best:.2f}")
            st.caption("Plus le score est négatif, meilleure est l'affinité prévue (selon Vina).")

    st.caption("ℹ️ Nécessite OpenBabel (obabel) + AutoDock Vina installés. "
               "Si la box vaut (0,0,0) on utilise le centroïde protéine par défaut.")





















