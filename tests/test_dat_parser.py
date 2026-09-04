from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data"
FEA = Path(__file__).resolve().parents[1] / "FEA"

_PARSER_NAME = "cae_test_dat_parser"


def _load_parser():
    if _PARSER_NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(_PARSER_NAME, FEA / "dat_parser.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_PARSER_NAME] = mod
        spec.loader.exec_module(mod)
    return sys.modules[_PARSER_NAME]


def _minimal_dat(mat_id: str = "1", bad_pressure: bool = False) -> str:
    return f"""4 1
coordinates
1 0.0 0.0
2 1.0 0.0
3 1.0 1.0
4 0.0 1.0
end coordinates
element
1 1 2 3 4 {mat_id}
end element
material properties
1 1000.0 0.2 5.0 1.0
2 2000.0 0.3 6.0 2.0
end material properties
Moment-Load
Node, 1, UX, 0.0, UY, 0.0
end moment-load
presure
Node, 3, {("-0.1.0" if bad_pressure else "-0.1")}
end presure
"""


def test_parse_reference_c1():
    p = _load_parser()
    m = p.read_dat(DATA / "c1.dat")
    assert m.n_nodes == 3086
    assert m.n_elements == 2938
    assert set(m.materials) == {1}


def test_parse_reference_d1():
    p = _load_parser()
    m = p.read_dat(DATA / "d1.dat")
    assert m.n_nodes == 2092
    assert m.n_elements == 1989
    assert set(m.materials) == {1}


def test_empty_file_raises(tmp_path: Path):
    p = _load_parser()
    f = tmp_path / "empty.dat"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        p.read_dat(f)


def test_header_mismatch_raises(tmp_path: Path):
    p = _load_parser()
    f = tmp_path / "mismatch.dat"
    lines = _minimal_dat().splitlines(keepends=True)
    lines[0] = "99 1\n"  # header node count disagrees with parsed coordinates
    f.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(ValueError):
        p.read_dat(f)


def test_missing_material_raises(tmp_path: Path):
    p = _load_parser()
    f = tmp_path / "mat.dat"
    f.write_text(_minimal_dat(mat_id="99"), encoding="utf-8")
    with pytest.raises(ValueError):
        p.read_dat(f)


def test_malformed_pressure_raises(tmp_path: Path):
    p = _load_parser()
    f = tmp_path / "badp.dat"
    f.write_text(_minimal_dat(bad_pressure=True), encoding="utf-8")
    with pytest.raises(ValueError):
        p.read_dat(f)


def test_multimaterial_and_wall_parse(tmp_path: Path):
    p = _load_parser()
    body = _minimal_dat()
    body += "wall\nNode, 2\nend wall\n"
    f = tmp_path / "mm.dat"
    f.write_text(body, encoding="utf-8")
    m = p.read_dat(f)
    assert len(m.materials) == 2
    assert int(m.element_material_ids[0]) == 1
    assert m.wall_nodes == [2]
    assert m.constraints.get(1) == {"ux": 0.0, "uy": 0.0}
    assert m.prescribed_uy.get(3) == -0.1
