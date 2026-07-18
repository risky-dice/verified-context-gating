"""CLI tests: data loading, verify verdicts, and exit codes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextgates.cli import _load, _normalize_docs, cmd_verify, main  # noqa: E402


def _write(tmp, rows):
    p = tmp / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_normalize_docs_accepts_all_three_shapes():
    dict_form = _normalize_docs([{"title": "A", "sentences": ["x.", "y."]}])
    tuple_form = _normalize_docs([["A", ["x.", "y."]]])
    text_form = _normalize_docs([{"title": "A", "text": "x. y."}])
    assert dict_form[0]["sentences"] == ["x.", "y."]
    assert tuple_form[0]["sentences"] == ["x.", "y."]
    assert text_form[0]["sentences"] == ["x.", "y."]


def test_load_skips_bad_and_missing(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps({"question": "q", "docs": [["A", ["s."]]], "gold_answer": "a"})
        + "\n" + "{not json}\n"
        + json.dumps({"docs": []})  # missing question
        + "\n",
        encoding="utf-8",
    )
    rows, skipped = _load(p)
    assert len(rows) == 1 and skipped == 2


def test_verify_pass_and_fail(tmp_path, capsys):
    comp = {
        "question": "Which is older, Alpha Film or Beta Film?",
        "docs": [
            {"title": "Alpha Film", "sentences": ["Alpha Film is a 1990 drama."]},
            {"title": "Beta Film", "sentences": ["Beta Film is a 2001 comedy."]},
            {"title": "Noise One", "sentences": ["Trains carry freight."]},
            {"title": "Noise Two", "sentences": ["The weather was cold."]},
            {"title": "Noise Three", "sentences": ["Bananas are yellow."]},
        ],
        "gold_answer": "Alpha Film",
        "support": [["Alpha Film", 0], ["Beta Film", 0]],
    }
    p = _write(tmp_path, [comp] * 120)

    class A:
        data = str(p); gate = "entity_title"; json = None
    rc = cmd_verify(A())
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out


def test_verify_missing_file_returns_2(capsys):
    class A:
        data = "/nonexistent/x.jsonl"; gate = None; json = None
    assert cmd_verify(A()) == 2


def test_main_dispatch_and_json_output(tmp_path):
    comp = {
        "question": "Which is older, Alpha Film or Beta Film?",
        "docs": [["Alpha Film", ["Alpha Film is a 1990 drama."]],
                 ["Beta Film", ["Beta Film is a 2001 comedy."]]],
        "gold_answer": "Alpha Film",
    }
    p = _write(tmp_path, [comp] * 105)
    outp = tmp_path / "report.json"
    rc = main(["verify", "--data", str(p), "--gate", "entity_title", "--json", str(outp)])
    assert rc == 0
    rep = json.loads(outp.read_text())
    assert rep["entity_title"]["n_admitted"] == 105


if __name__ == "__main__":
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q"], cwd=str(Path(__file__).parent))
    sys.exit(r.returncode)
