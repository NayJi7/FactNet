"""Smoke test: the package and its submodules import cleanly."""

import importlib

import pytest

MODULES = [
    "factnet",
    "factnet.ingestion",
    "factnet.nlp",
    "factnet.graph",
    "factnet.scoring",
    "factnet.viz",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
