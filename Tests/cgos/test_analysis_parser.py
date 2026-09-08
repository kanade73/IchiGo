"""kata-genmove_analyze -> CGOS analysis JSON (docs/spec/03-engine.md §9 example line; reproduces
the pinned reference client's AnalyzeResultParser -- see ichigo_cgos_client module docstring)."""

import json

from ichigo_cgos_client import encode_analysis, parse_kata_info_line

SPEC_EXAMPLE_LINE = (
    "info move D4 visits 80 winrate 0.620000 scoreLead 2.300000 prior 0.120000 order 0 pv D4 E4"
)


def test_spec_example_line_matches_reference_json_shape():
    info = parse_kata_info_line(SPEC_EXAMPLE_LINE.split(" "))
    assert info == {
        "moves": [{"move": "D4", "visits": 80, "winrate": 0.62, "score": 2.3, "prior": 0.12, "pv": "E4"}],
        "visits": 80,
        "winrate": 0.62,
        "score": 2.3,
    }
    # Compact separators, no spaces -- exactly what the pinned server round-trips through
    # json.loads/json.dumps(..., separators=(",", ":")).
    encoded = encode_analysis(info)
    assert encoded == (
        '{"moves":[{"move":"D4","visits":80,"winrate":0.62,"score":2.3,"prior":0.12,"pv":"E4"}],'
        '"visits":80,"winrate":0.62,"score":2.3}'
    )
    assert " " not in encoded
    # And it must itself parse back as a JSON object, matching the pinned server's acceptance
    # check ("isinstance(info, dict)") for the analysis extension.
    assert isinstance(json.loads(encoded), dict)


def test_single_token_pv_is_dropped_entirely():
    line = "info move D4 visits 80 winrate 0.62 scoreLead 2.3 prior 0.12 order 0 pv D4"
    info = parse_kata_info_line(line.split(" "))
    assert "pv" not in info["moves"][0]


def test_order_and_other_unlisted_tokens_are_dropped():
    line = "info move D4 visits 80 winrate 0.62 scoreLead 2.3 prior 0.12 order 3 pv D4 E4 F4"
    info = parse_kata_info_line(line.split(" "))
    assert "order" not in info["moves"][0]
    assert info["moves"][0]["pv"] == "E4 F4"


def test_weighted_root_summary_matches_single_candidate():
    line = "info move Q16 visits 5 winrate 0.5 scoreLead 0.0 prior 0.05 order 0 pv Q16"
    info = parse_kata_info_line(line.split(" "))
    assert info["visits"] == 5
    assert info["winrate"] == 0.5
    assert info["score"] == 0.0
