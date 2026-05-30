"""Sphinx configuration for the churn project docs."""

import os
import sys

# Make the package importable for autodoc.
sys.path.insert(0, os.path.abspath("../src"))

project = "RetailGenius Churn Prediction"
author = "EPITA AI PM - Group 10"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"

# Napoleon: we use Google-style docstrings throughout.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Don't fail the build if heavy deps aren't importable at doc-build time.
autodoc_mock_imports = ["mlflow", "shap", "xgboost", "sklearn"]
