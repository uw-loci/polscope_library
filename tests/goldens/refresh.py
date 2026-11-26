"""Re-record the goldens in this folder from the code as it is checked out.

This is the only way the goldens change, and it is deliberately manual. Run it from the
repository root with a reason, and put the same reason in the commit message:

    python3 tests/goldens/refresh.py --reason "1.6.0: hue_shift keeps float hue (#NN)"

It captures every ``case_<name>`` in cases.py with the verify-by-oracle skill's
``oracle.py capture`` (recording sha256, shape, dtype, library versions, git ref and the
comparison rule per case), then packs the arrays into ``goldens.npz`` and the record into
``manifest.json``. It refuses a tree with uncommitted changes to tracked files, because a
golden has to name the commit it came from.

``--rules-only`` rewrites the comparison rules in manifest.json from cases.RULES without
touching the recorded arrays, for a change to a rule that is not a change to the code.

Never run this to make a failing replay pass. A difference the replay found is either a
bug (fix the code) or an intended change; only the second earns a refresh, and its reason
is the changelog entry or issue that describes the new behavior. See README.md here.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOLDENS = HERE / "goldens.npz"
MANIFEST = HERE / "manifest.json"
DEFAULT_ORACLE = Path.home() / ".claude" / "skills" / "verify-by-oracle" / "scripts" / "oracle.py"


def load_cases():
    spec = importlib.util.spec_from_file_location("golden_cases", HERE / "cases.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO))
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def oracle_rule(rule):
    """The rule as oracle.py understands it; it has no circular comparison of its own."""
    if rule.get("kind") == "angle":
        return {
            "kind": "tolerance",
            "atol": rule["atol"],
            "rtol": 0.0,
            "reason": rule["reason"]
            + " (tests/test_goldens.py compares this on the circle, period %s)" % rule["period"],
        }
    return rule


def write_manifest(record):
    MANIFEST.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="ascii")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--reason", required=True, help="why the goldens change; goes into manifest.json and the commit"
    )
    parser.add_argument("--oracle", default=os.environ.get("ORACLE_PY", str(DEFAULT_ORACLE)), help="path to oracle.py")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="record from a tree with uncommitted tracked changes"
    )
    parser.add_argument("--rules-only", action="store_true", help="rewrite the rules in manifest.json; keep the arrays")
    args = parser.parse_args(argv)

    cases = load_cases()
    names = cases.all_case_names()
    missing = sorted(set(names) - set(cases.RULES))
    extra = sorted(set(cases.RULES) - set(names))
    if missing or extra:
        sys.exit("RULES and cases disagree: no rule for %s; rule without case %s" % (missing, extra))

    if args.rules_only:
        record = json.loads(MANIFEST.read_text(encoding="ascii"))
        for name in names:
            record["cases"][name]["comparison"] = cases.RULES[name]
            record["cases"][name]["labels"] = cases.LABELS.get(name)
        record["rules_updated"] = {"reason": args.reason, "git_head": git("rev-parse", "HEAD")}
        write_manifest(record)
        print("rules rewritten in %s (%d cases); arrays untouched" % (MANIFEST, len(names)))
        return 0

    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        sys.exit("uncommitted changes to tracked files; commit first so the goldens name a ref:\n" + dirty)
    oracle = Path(args.oracle)
    if not oracle.exists():
        sys.exit("oracle.py not found at %s (pass --oracle or set ORACLE_PY)" % oracle)

    describe = git("describe", "--tags", "--always", "--dirty")
    work = Path(tempfile.mkdtemp(prefix="goldens-"))
    try:
        spec = {
            "stage": REPO.name + "-goldens",
            "record_versions": cases.RECORD_VERSIONS,
            "comparison": {"kind": "exact"},
            "cases": [
                {
                    "name": n,
                    "callable": "%s:case_%s" % (HERE / "cases.py", n),
                    "comparison": oracle_rule(cases.RULES[n]),
                }
                for n in names
            ],
        }
        (work / "spec.json").write_text(json.dumps(spec, indent=1), encoding="ascii")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        capture = work / "capture"
        command = [
            sys.executable,
            str(oracle),
            "capture",
            "--spec",
            str(work / "spec.json"),
            "--out",
            str(capture),
            "--label",
            "%s goldens at %s: %s" % (REPO.name, describe, args.reason),
            "--git-ref",
            describe,
            "--repo",
            str(REPO),
        ]
        result = subprocess.run(command, env=env, cwd=str(REPO))
        if result.returncode != 0:
            sys.exit("oracle.py capture failed; nothing written")
        captured = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
        arrays = {}
        record_cases = {}
        for name, entry in captured["cases"].items():
            if "error" in entry:
                sys.exit("case %s failed: %s" % (name, entry["error"]))
            if entry["type"] != "ndarray":
                sys.exit("case %s returned %s; every case must return a numpy array" % (name, entry["type"]))
            arrays[name] = np.load(capture / entry["file"], allow_pickle=False)
            record_cases[name] = {
                "sha256": entry["sha256"],
                "shape": entry["shape"],
                "dtype": entry["dtype"],
                "seconds": entry["seconds"],
                "comparison": cases.RULES[name],
                "labels": cases.LABELS.get(name),
            }
        np.savez_compressed(GOLDENS, **arrays)
        record = {
            "label": captured["label"],
            "reason": args.reason,
            "created": captured["created"],
            "git_ref": describe,
            "git_head": captured["git_head"],
            "git_dirty": bool(dirty),
            "library_version": getattr(cases, "LIBRARY_VERSION", None),
            "python": captured["python"],
            "numpy": captured["numpy"],
            "platform": captured["platform"],
            "versions": captured["versions"],
            "seed": cases.SEED,
            "oracle": "verify-by-oracle oracle.py capture, packed by tests/goldens/refresh.py",
            "goldens_sha256": hashlib.sha256(GOLDENS.read_bytes()).hexdigest(),
            "cases": record_cases,
        }
        write_manifest(record)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("wrote %s (%d cases, %d bytes) and %s" % (GOLDENS, len(arrays), GOLDENS.stat().st_size, MANIFEST))
    return 0


if __name__ == "__main__":
    sys.exit(main())
