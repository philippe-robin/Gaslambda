"""
Build reference dataset for gas thermal conductivity of organic compounds.

Data sources:
- DIPPR 801 recommended values
- Yaws' Handbook of Transport Properties of Chemicals and Hydrocarbons
- NIST WebBook (webbook.nist.gov)
- Poling, Prausnitz & O'Connell, "The Properties of Gases and Liquids", 5th ed.

Values are at 1 atm, temperatures as indicated (mostly 298.15 K or 300 K).
Units: λ in W/(m·K), T in K.

NOTE: This dataset contains curated reference values from published literature.
For production use, expand with DIPPR 801 or TDE licensed data.
"""

import pandas as pd
import json

# Reference data: (name, SMILES, T_K, lambda_W_mK, source)
# Sources: DIPPR=DIPPR 801 Database, YAWS=Yaws Handbook, NIST=NIST WebBook, PPO=Poling et al.
REFERENCE_DATA = [
    # ---- Alkanes ----
    ("Methane", "C", 300, 0.0343, "DIPPR"),
    ("Methane", "C", 400, 0.0505, "DIPPR"),
    ("Methane", "C", 500, 0.0672, "DIPPR"),
    ("Ethane", "CC", 300, 0.0218, "DIPPR"),
    ("Ethane", "CC", 400, 0.0355, "DIPPR"),
    ("Ethane", "CC", 500, 0.0510, "DIPPR"),
    ("Propane", "CCC", 300, 0.0182, "DIPPR"),
    ("Propane", "CCC", 400, 0.0288, "DIPPR"),
    ("Propane", "CCC", 500, 0.0415, "DIPPR"),
    ("n-Butane", "CCCC", 300, 0.0162, "DIPPR"),
    ("n-Butane", "CCCC", 400, 0.0249, "DIPPR"),
    ("n-Butane", "CCCC", 500, 0.0362, "DIPPR"),
    ("Isobutane", "CC(C)C", 300, 0.0171, "DIPPR"),
    ("Isobutane", "CC(C)C", 400, 0.0265, "DIPPR"),
    ("n-Pentane", "CCCCC", 300, 0.0148, "DIPPR"),
    ("n-Pentane", "CCCCC", 400, 0.0225, "DIPPR"),
    ("n-Pentane", "CCCCC", 500, 0.0327, "DIPPR"),
    ("Isopentane", "CCC(C)C", 300, 0.0146, "DIPPR"),
    ("Neopentane", "CC(C)(C)C", 300, 0.0149, "DIPPR"),
    ("n-Hexane", "CCCCCC", 300, 0.0137, "DIPPR"),
    ("n-Hexane", "CCCCCC", 400, 0.0207, "DIPPR"),
    ("n-Heptane", "CCCCCCC", 400, 0.0194, "DIPPR"),
    ("n-Octane", "CCCCCCCC", 400, 0.0183, "DIPPR"),
    ("n-Nonane", "CCCCCCCCC", 400, 0.0175, "YAWS"),
    ("n-Decane", "CCCCCCCCCC", 400, 0.0168, "YAWS"),
    ("Cyclohexane", "C1CCCCC1", 350, 0.0158, "DIPPR"),
    ("Cyclohexane", "C1CCCCC1", 400, 0.0201, "DIPPR"),
    ("Cyclopentane", "C1CCCC1", 350, 0.0163, "DIPPR"),
    ("Methylcyclohexane", "CC1CCCCC1", 400, 0.0184, "YAWS"),
    # ---- Alkenes ----
    ("Ethylene", "C=C", 300, 0.0218, "DIPPR"),
    ("Ethylene", "C=C", 400, 0.0338, "DIPPR"),
    ("Ethylene", "C=C", 500, 0.0467, "DIPPR"),
    ("Propylene", "CC=C", 300, 0.0178, "DIPPR"),
    ("Propylene", "CC=C", 400, 0.0276, "DIPPR"),
    ("1-Butene", "CCC=C", 300, 0.0157, "DIPPR"),
    ("1-Butene", "CCC=C", 400, 0.0241, "DIPPR"),
    ("1,3-Butadiene", "C=CC=C", 300, 0.0162, "DIPPR"),
    ("Isobutylene", "CC(=C)C", 300, 0.0161, "DIPPR"),
    ("1-Pentene", "CCCC=C", 350, 0.0175, "YAWS"),
    ("1-Hexene", "CCCCC=C", 350, 0.0159, "YAWS"),
    # ---- Alkynes ----
    ("Acetylene", "C#C", 300, 0.0228, "DIPPR"),
    ("Acetylene", "C#C", 400, 0.0332, "DIPPR"),
    # ---- Aromatics ----
    ("Benzene", "c1ccccc1", 400, 0.0188, "DIPPR"),
    ("Benzene", "c1ccccc1", 500, 0.0272, "DIPPR"),
    ("Toluene", "Cc1ccccc1", 400, 0.0178, "DIPPR"),
    ("Toluene", "Cc1ccccc1", 500, 0.0259, "DIPPR"),
    ("Ethylbenzene", "CCc1ccccc1", 400, 0.0171, "YAWS"),
    ("o-Xylene", "Cc1ccccc1C", 400, 0.0174, "YAWS"),
    ("m-Xylene", "Cc1cccc(C)c1", 400, 0.0170, "YAWS"),
    ("p-Xylene", "Cc1ccc(C)cc1", 400, 0.0169, "YAWS"),
    ("Styrene", "C=Cc1ccccc1", 400, 0.0165, "YAWS"),
    ("Naphthalene", "c1ccc2ccccc2c1", 500, 0.0230, "YAWS"),
    # ---- Alcohols ----
    ("Methanol", "CO", 300, 0.0165, "DIPPR"),
    ("Methanol", "CO", 400, 0.0254, "DIPPR"),
    ("Methanol", "CO", 500, 0.0367, "DIPPR"),
    ("Ethanol", "CCO", 300, 0.0154, "DIPPR"),
    ("Ethanol", "CCO", 400, 0.0237, "DIPPR"),
    ("1-Propanol", "CCCO", 400, 0.0218, "DIPPR"),
    ("2-Propanol", "CC(O)C", 400, 0.0221, "DIPPR"),
    ("1-Butanol", "CCCCO", 400, 0.0201, "YAWS"),
    ("Phenol", "Oc1ccccc1", 500, 0.0243, "YAWS"),
    ("Ethylene glycol", "OCCO", 470, 0.0229, "YAWS"),
    # ---- Ketones ----
    ("Acetone", "CC(=O)C", 300, 0.0117, "DIPPR"),
    ("Acetone", "CC(=O)C", 400, 0.0192, "DIPPR"),
    ("Acetone", "CC(=O)C", 500, 0.0282, "DIPPR"),
    ("2-Butanone", "CCC(=O)C", 400, 0.0177, "DIPPR"),
    ("Methyl isobutyl ketone", "CC(=O)CC(C)C", 400, 0.0165, "YAWS"),
    ("Cyclohexanone", "O=C1CCCCC1", 450, 0.0189, "YAWS"),
    # ---- Aldehydes ----
    ("Formaldehyde", "C=O", 300, 0.0181, "DIPPR"),
    ("Acetaldehyde", "CC=O", 300, 0.0142, "DIPPR"),
    ("Acetaldehyde", "CC=O", 400, 0.0220, "DIPPR"),
    # ---- Carboxylic acids ----
    ("Formic acid", "OC=O", 400, 0.0213, "DIPPR"),
    ("Acetic acid", "CC(=O)O", 400, 0.0193, "DIPPR"),
    ("Acetic acid", "CC(=O)O", 500, 0.0278, "DIPPR"),
    ("Propionic acid", "CCC(=O)O", 420, 0.0188, "YAWS"),
    ("Butyric acid", "CCCC(=O)O", 440, 0.0182, "YAWS"),
    ("Acrylic acid", "C=CC(=O)O", 420, 0.0187, "YAWS"),
    # ---- Esters ----
    ("Methyl formate", "COC=O", 300, 0.0131, "DIPPR"),
    ("Ethyl acetate", "CCOC(=O)C", 400, 0.0181, "DIPPR"),
    ("Methyl acetate", "COC(=O)C", 350, 0.0144, "DIPPR"),
    ("n-Butyl acetate", "CCCCOC(=O)C", 420, 0.0175, "YAWS"),
    ("Methyl methacrylate", "COC(=O)C(=C)C", 400, 0.0158, "YAWS"),
    # ---- Ethers ----
    ("Dimethyl ether", "COC", 300, 0.0157, "DIPPR"),
    ("Dimethyl ether", "COC", 400, 0.0244, "DIPPR"),
    ("Diethyl ether", "CCOCC", 300, 0.0144, "DIPPR"),
    ("Diethyl ether", "CCOCC", 400, 0.0211, "DIPPR"),
    ("Tetrahydrofuran", "C1CCOC1", 350, 0.0155, "YAWS"),
    ("1,4-Dioxane", "C1COCCO1", 400, 0.0171, "YAWS"),
    ("Anisole", "COc1ccccc1", 450, 0.0187, "YAWS"),
    # ---- Amines ----
    ("Methylamine", "CN", 300, 0.0217, "DIPPR"),
    ("Methylamine", "CN", 400, 0.0327, "DIPPR"),
    ("Dimethylamine", "CNC", 300, 0.0172, "DIPPR"),
    ("Trimethylamine", "CN(C)C", 300, 0.0141, "DIPPR"),
    ("Ethylamine", "CCN", 300, 0.0185, "DIPPR"),
    ("Aniline", "Nc1ccccc1", 500, 0.0261, "YAWS"),
    ("Diethylamine", "CCNCC", 350, 0.0157, "YAWS"),
    # ---- Nitriles ----
    ("Acetonitrile", "CC#N", 350, 0.0142, "DIPPR"),
    ("Acetonitrile", "CC#N", 400, 0.0176, "DIPPR"),
    ("Propionitrile", "CCC#N", 400, 0.0163, "YAWS"),
    ("Acrylonitrile", "C=CC#N", 400, 0.0172, "YAWS"),
    # ---- Halogenated ----
    ("Chloromethane", "CCl", 300, 0.0109, "DIPPR"),
    ("Chloromethane", "CCl", 400, 0.0162, "DIPPR"),
    ("Dichloromethane", "ClCCl", 300, 0.0082, "DIPPR"),
    ("Dichloromethane", "ClCCl", 400, 0.0124, "DIPPR"),
    ("Chloroform", "ClC(Cl)Cl", 300, 0.0077, "DIPPR"),
    ("Carbon tetrachloride", "ClC(Cl)(Cl)Cl", 300, 0.0073, "DIPPR"),
    ("Fluoromethane", "CF", 300, 0.0157, "DIPPR"),
    ("1,1-Difluoroethane", "CC(F)F", 300, 0.0134, "DIPPR"),
    ("Chlorobenzene", "Clc1ccccc1", 400, 0.0141, "YAWS"),
    ("Vinyl chloride", "C=CCl", 300, 0.0113, "DIPPR"),
    ("1,2-Dichloroethane", "ClCCCl", 350, 0.0107, "YAWS"),
    # ---- Sulfur compounds ----
    ("Dimethyl sulfide", "CSC", 300, 0.0124, "DIPPR"),
    ("Methanethiol", "CS", 300, 0.0155, "DIPPR"),
    ("Ethanethiol", "CCS", 350, 0.0145, "YAWS"),
    ("Thiophene", "c1ccsc1", 400, 0.0160, "YAWS"),
    ("Dimethyl sulfoxide", "CS(=O)C", 470, 0.0182, "YAWS"),
    # ---- Other functional groups ----
    ("Nitromethane", "C[N+](=O)[O-]", 400, 0.0195, "YAWS"),
    ("Furan", "c1ccoc1", 350, 0.0161, "YAWS"),
    ("Pyridine", "c1ccncc1", 400, 0.0175, "YAWS"),
    ("Morpholine", "C1COCCN1", 400, 0.0170, "YAWS"),
    ("N,N-Dimethylformamide", "CN(C)C=O", 400, 0.0185, "YAWS"),
    ("N-Methyl-2-pyrrolidone", "CN1CCCC1=O", 470, 0.0192, "YAWS"),
    ("Acetic anhydride", "CC(=O)OC(=O)C", 420, 0.0170, "YAWS"),
    # ---- Multi-temperature for key compounds ----
    ("Benzene", "c1ccccc1", 350, 0.0136, "DIPPR"),
    ("Benzene", "c1ccccc1", 450, 0.0228, "DIPPR"),
    ("Toluene", "Cc1ccccc1", 350, 0.0127, "YAWS"),
    ("Toluene", "Cc1ccccc1", 450, 0.0217, "YAWS"),
    ("Methanol", "CO", 350, 0.0207, "DIPPR"),
    ("Ethanol", "CCO", 350, 0.0193, "DIPPR"),
    ("Acetone", "CC(=O)C", 350, 0.0153, "DIPPR"),
    ("n-Hexane", "CCCCCC", 350, 0.0169, "DIPPR"),
    ("n-Hexane", "CCCCCC", 500, 0.0297, "DIPPR"),
    ("Chloroform", "ClC(Cl)Cl", 400, 0.0117, "DIPPR"),
    ("Dimethyl ether", "COC", 350, 0.0199, "DIPPR"),
    ("Diethyl ether", "CCOCC", 350, 0.0176, "DIPPR"),
]


def build_dataset():
    """Build and save the reference dataset."""
    records = []
    for name, smiles, T, lam, source in REFERENCE_DATA:
        records.append({
            "name": name,
            "smiles": smiles,
            "temperature_K": T,
            "thermal_conductivity_W_mK": lam,
            "source": source,
        })

    df = pd.DataFrame(records)
    print(f"Dataset: {len(df)} data points, {df['name'].nunique()} unique compounds")
    print(f"Temperature range: {df['temperature_K'].min()}-{df['temperature_K'].max()} K")
    print(f"λ range: {df['thermal_conductivity_W_mK'].min():.4f}-{df['thermal_conductivity_W_mK'].max():.4f} W/(m·K)")
    print(f"Sources: {df['source'].value_counts().to_dict()}")

    df.to_csv("data/reference_thermal_conductivity.csv", index=False)
    print("Saved to data/reference_thermal_conductivity.csv")
    return df


if __name__ == "__main__":
    build_dataset()
