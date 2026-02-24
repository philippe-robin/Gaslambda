"""
Molecular descriptor computation for thermal conductivity prediction.

Computes physics-informed descriptors from SMILES using RDKit:
- Molecular weight, atom counts (relevant to degrees of freedom)
- Topological descriptors (molecular shape/branching)
- Electronic descriptors (polarity, H-bonding)
- Size/shape descriptors (surface area, volume)

These descriptors are chosen based on physical understanding of gas thermal
conductivity: λ depends on molecular mass, shape (affecting collision cross-section),
degrees of freedom (translational + rotational + vibrational), and polarity
(affecting intermolecular forces).

References:
- Poling, Prausnitz & O'Connell, "Properties of Gases and Liquids", 5th ed., Ch. 10
- Gharagheizi et al., Ind. Eng. Chem. Res., 2013, 52, 7165-7174
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments, Crippen
from rdkit.Chem import AllChem, GraphDescriptors


def compute_descriptors(smiles: str, temperature_K: float = 300.0) -> dict:
    """
    Compute molecular descriptors for thermal conductivity prediction.
    
    Parameters
    ----------
    smiles : str
        SMILES representation of the molecule.
    temperature_K : float
        Temperature in Kelvin (used as a feature).
    
    Returns
    -------
    dict
        Dictionary of descriptor name -> value.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Add hydrogens for accurate 3D descriptors
    mol_h = Chem.AddHs(mol)
    
    descriptors = {}
    
    # ---- Temperature (key variable) ----
    descriptors["temperature_K"] = temperature_K
    descriptors["temperature_reduced"] = temperature_K / 1000.0  # normalized
    
    # ---- Basic molecular properties ----
    descriptors["molecular_weight"] = Descriptors.MolWt(mol)
    descriptors["heavy_atom_count"] = mol.GetNumHeavyAtoms()
    descriptors["num_atoms_total"] = mol_h.GetNumAtoms()  # including H
    descriptors["num_bonds"] = mol.GetNumBonds()
    
    # Degrees of freedom (critical for thermal conductivity)
    # For non-linear molecule: 3 translational + 3 rotational + (3N-6) vibrational
    n_atoms = mol_h.GetNumAtoms()
    descriptors["dof_total"] = 3 * n_atoms
    descriptors["dof_vibrational"] = max(3 * n_atoms - 6, 0)
    
    # ---- Atom type counts ----
    descriptors["num_C"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    descriptors["num_H"] = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1)
    descriptors["num_O"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    descriptors["num_N"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    descriptors["num_S"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
    descriptors["num_F"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    descriptors["num_Cl"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 17)
    descriptors["num_Br"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 35)
    
    # Halogen content (affects thermal conductivity significantly)
    descriptors["halogen_fraction"] = (
        descriptors["num_F"] + descriptors["num_Cl"] + descriptors["num_Br"]
    ) / max(descriptors["heavy_atom_count"], 1)
    
    # ---- Electronic / Polarity descriptors ----
    descriptors["TPSA"] = Descriptors.TPSA(mol)  # Topological polar surface area
    descriptors["MolLogP"] = Crippen.MolLogP(mol)  # Lipophilicity
    descriptors["MolMR"] = Crippen.MolMR(mol)  # Molar refractivity (~ polarizability)
    descriptors["NumHDonors"] = Descriptors.NumHDonors(mol)
    descriptors["NumHAcceptors"] = Descriptors.NumHAcceptors(mol)
    
    # ---- Topological descriptors (molecular shape/branching) ----
    descriptors["BertzCT"] = GraphDescriptors.BertzCT(mol)  # Complexity
    descriptors["Chi0"] = GraphDescriptors.Chi0(mol)  # Connectivity index
    descriptors["Chi1"] = GraphDescriptors.Chi1(mol)
    descriptors["Kappa1"] = GraphDescriptors.Kappa1(mol)  # Shape indices
    descriptors["Kappa2"] = GraphDescriptors.Kappa2(mol)
    descriptors["Kappa3"] = GraphDescriptors.Kappa3(mol)
    
    # Wiener index (path-based topological descriptor)
    try:
        from rdkit.Chem import rdmolops
        dm = rdmolops.GetDistanceMatrix(mol)
        descriptors["WienerIndex"] = np.sum(dm) / 2.0
    except:
        descriptors["WienerIndex"] = 0.0
    
    # ---- Ring / Aromaticity ----
    descriptors["num_aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
    descriptors["num_aliphatic_rings"] = rdMolDescriptors.CalcNumAliphaticRings(mol)
    descriptors["num_rings"] = rdMolDescriptors.CalcNumRings(mol)
    descriptors["fraction_CSP3"] = Descriptors.FractionCSP3(mol)  # sp3 carbon fraction
    
    # ---- Bond types ----
    descriptors["num_rotatable_bonds"] = Descriptors.NumRotatableBonds(mol)
    descriptors["num_double_bonds"] = sum(
        1 for b in mol.GetBonds() 
        if b.GetBondType() == Chem.BondType.DOUBLE
    )
    descriptors["num_triple_bonds"] = sum(
        1 for b in mol.GetBonds() 
        if b.GetBondType() == Chem.BondType.TRIPLE
    )
    
    # ---- Derived physics-informed features ----
    # 1/sqrt(M) appears in kinetic theory expression for λ
    descriptors["inv_sqrt_MW"] = 1.0 / np.sqrt(max(descriptors["molecular_weight"], 1.0))
    
    # Cv/R approximation (classical): 3 trans + 3 rot + (3N-6) vib ≈ (3N-3) at high T
    # At moderate T, vibrational modes are partially excited
    descriptors["Cv_R_classical"] = 3 * n_atoms - 3  # upper limit
    
    # Eucken factor proxy: f = 1 + (9/4)*(R/Cv)
    # Higher Cv -> lower Eucken factor -> different T dependence
    cv_r = max(descriptors["Cv_R_classical"], 2.5)
    descriptors["eucken_factor"] = 1.0 + 9.0 / (4.0 * cv_r)
    
    # T-dependent feature: λ ∝ T^n where n depends on molecule
    descriptors["T_power_0.7"] = temperature_K ** 0.7  # typical exponent for polyatomics
    descriptors["T_over_MW"] = temperature_K / max(descriptors["molecular_weight"], 1.0)
    
    return descriptors


def compute_descriptors_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptors for a DataFrame with 'smiles' and 'temperature_K' columns.
    
    Returns a new DataFrame with all descriptor columns.
    """
    all_desc = []
    errors = []
    
    for idx, row in df.iterrows():
        try:
            desc = compute_descriptors(row["smiles"], row["temperature_K"])
            desc["_index"] = idx
            all_desc.append(desc)
        except Exception as e:
            errors.append((idx, row.get("name", "?"), str(e)))
    
    if errors:
        print(f"Warning: {len(errors)} errors during descriptor computation:")
        for idx, name, err in errors[:5]:
            print(f"  [{idx}] {name}: {err}")
    
    desc_df = pd.DataFrame(all_desc).set_index("_index")
    return desc_df


# Feature names used by the model (in order)
FEATURE_NAMES = [
    "temperature_K", "temperature_reduced",
    "molecular_weight", "heavy_atom_count", "num_atoms_total", "num_bonds",
    "dof_total", "dof_vibrational",
    "num_C", "num_H", "num_O", "num_N", "num_S", "num_F", "num_Cl", "num_Br",
    "halogen_fraction",
    "TPSA", "MolLogP", "MolMR", "NumHDonors", "NumHAcceptors",
    "BertzCT", "Chi0", "Chi1", "Kappa1", "Kappa2", "Kappa3", "WienerIndex",
    "num_aromatic_rings", "num_aliphatic_rings", "num_rings", "fraction_CSP3",
    "num_rotatable_bonds", "num_double_bonds", "num_triple_bonds",
    "inv_sqrt_MW", "Cv_R_classical", "eucken_factor",
    "T_power_0.7", "T_over_MW",
]


if __name__ == "__main__":
    # Quick test
    test_smiles = ["C", "CC", "c1ccccc1", "CCO", "CC(=O)C", "ClCCl"]
    test_names = ["Methane", "Ethane", "Benzene", "Ethanol", "Acetone", "DCM"]
    
    for name, smi in zip(test_names, test_smiles):
        desc = compute_descriptors(smi, 300.0)
        print(f"{name:12s} | MW={desc['molecular_weight']:.1f} | "
              f"DOF_vib={desc['dof_vibrational']} | "
              f"TPSA={desc['TPSA']:.1f} | "
              f"Eucken={desc['eucken_factor']:.3f}")
