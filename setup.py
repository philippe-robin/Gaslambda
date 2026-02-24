from setuptools import setup, find_packages

setup(
    name="gaslambda",
    version="1.0.0",
    description="QSPR/ML predictor for gas-phase thermal conductivity of organic compounds",
    author="Alysophil SAS",
    author_email="contact@alysophil.com",
    url="https://github.com/alysophil/gaslambda",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24,<2.0",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "joblib>=1.3",
        "rdkit>=2023.9",
    ],
    extras_require={
        "app": ["streamlit>=1.30", "plotly>=5.18"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
