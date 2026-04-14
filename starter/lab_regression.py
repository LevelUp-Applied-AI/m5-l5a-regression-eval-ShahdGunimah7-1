"""Compatibility wrapper for test import path.

Tests expect ``starter/lab_regression.py`` to exist. This shim loads the
project's root ``lab_regression.py`` and re-exports the lab functions.
"""

from pathlib import Path
import importlib.util

# Load root lab_regression.py
root_file = Path(__file__).resolve().parents[1] / "lab_regression.py"
spec = importlib.util.spec_from_file_location("lab_regression_root", root_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Re-export functions
load_data = module.load_data
split_data = module.split_data
build_logistic_pipeline = module.build_logistic_pipeline
build_ridge_pipeline = module.build_ridge_pipeline
evaluate_classifier = module.evaluate_classifier
evaluate_regressor = module.evaluate_regressor
run_cross_validation = module.run_cross_validation