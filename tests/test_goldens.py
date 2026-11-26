"""Replay the library against the goldens recorded in tests/goldens/.

Each golden is the output of one public function on a fixed input, recorded from a named
commit with the verify-by-oracle skill's oracle.py (tests/goldens/manifest.json says which
commit, which library versions, and the comparison rule per case). A failure here means
the current code gives a different answer than that commit did. That is either a bug or
an intended change; only the second is a reason to refresh, and the refresh is manual
(tests/goldens/README.md). Run with ``pytest -s`` to see the measured spread of every
tolerance and angle case.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
GOLDENS_DIR = HERE / "goldens"


def _load(name):
    spec = importlib.util.spec_from_file_location("goldens_" + name, GOLDENS_DIR / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    if str(HERE.parent) not in sys.path:
        sys.path.insert(0, str(HERE.parent))
    spec.loader.exec_module(module)
    return module


cases = _load("cases")
compare = _load("compare")
MANIFEST = json.loads((GOLDENS_DIR / "manifest.json").read_text(encoding="ascii"))
CASE_NAMES = sorted(MANIFEST["cases"])


@pytest.fixture(scope="module")
def goldens():
    with np.load(GOLDENS_DIR / "goldens.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def test_cases_rules_and_goldens_name_the_same_set(goldens):
    """A new case needs a recorded golden; a stale golden needs its case back or a refresh."""
    assert set(cases.all_case_names()) == set(CASE_NAMES)
    assert set(cases.RULES) == set(CASE_NAMES)
    assert set(goldens) == set(CASE_NAMES)


def test_rules_in_code_match_the_manifest():
    """A changed rule is recorded with its reason (refresh.py --rules-only), never edited in one place."""
    for name in CASE_NAMES:
        compare.validate_rule(cases.RULES[name])
        assert MANIFEST["cases"][name]["comparison"] == cases.RULES[name], name
        assert MANIFEST["cases"][name].get("labels") == cases.LABELS.get(name), name


@pytest.mark.parametrize("name", CASE_NAMES)
def test_golden(name, goldens):
    new = np.asarray(getattr(cases, "case_" + name)())
    ok, message, measured = compare.compare(goldens[name], new, cases.RULES[name], labels=cases.LABELS.get(name))
    if measured:
        print("[golden %s] %s" % (name, measured))
    assert ok, "%s: %s\n(golden recorded at %s, %s; tests/goldens/README.md says what to do)" % (
        name,
        message,
        MANIFEST["git_ref"],
        MANIFEST["git_head"][:12],
    )
