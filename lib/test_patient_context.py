"""Unit tests for PatientContextAnalyzer.

Run: python -m pytest lib/test_patient_context.py -v
  or: python lib/test_patient_context.py
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.patient_context import PatientContextAnalyzer


def make_mock_llm(response: str):
    """Create a mock LLM that returns a fixed response."""
    async def mock_llm(messages):
        return response
    return mock_llm


def make_enriched_relation(target_name, relation, constraints):
    """Helper to build a relation dict with context_constraints."""
    return {
        "relation": relation,
        "target_name": target_name,
        "target_index": 999,
        "target_type": "disease",
        "direction": "outgoing",
        "context_constraints": constraints,
    }


def make_plain_relation(target_name, relation):
    """Helper to build a plain relation dict (no enrichment)."""
    return {
        "relation": relation,
        "target_name": target_name,
        "target_index": 888,
        "target_type": "disease",
        "direction": "outgoing",
    }


# ------------------------------------------------------------------
# Tests for analyze()
# ------------------------------------------------------------------

def test_no_enriched_relations():
    """Plain relations pass through unchanged."""
    relations = [
        make_plain_relation("Nausea", "drug_effect"),
        make_plain_relation("Headache", "drug_effect"),
    ]
    # LLM should not be called — use a mock that would fail
    analyzer = PatientContextAnalyzer(make_mock_llm("SHOULD NOT BE CALLED"))
    result = asyncio.run(analyzer.analyze(relations, "74yo male"))
    assert len(result) == 2
    assert result[0]["target_name"] == "Nausea"
    assert result[1]["target_name"] == "Headache"
    print("  PASS: plain relations pass through unchanged")


def test_definitely_not_applicable_removed():
    """Relations matched as 'Definitely NOT Applicable' are removed."""
    relations = [
        make_enriched_relation("Constipation", "drug_effect", [
            {"patient_group": "Elderly ≥65", "applicability": "Definitely NOT Applicable", "evidence": "..."},
        ]),
        make_plain_relation("Nausea", "drug_effect"),
    ]
    llm_response = json.dumps([
        {"applicability": "Definitely NOT Applicable", "reason": "Patient is 30yo, not elderly"}
    ])
    analyzer = PatientContextAnalyzer(make_mock_llm(llm_response))
    result = asyncio.run(analyzer.analyze(relations, "30yo female"))
    assert len(result) == 1
    assert result[0]["target_name"] == "Nausea"
    print("  PASS: Definitely NOT Applicable relation removed")


def test_definitely_applicable_emphasized():
    """Relations matched as 'Definitely Applicable' get _patient_relevance annotation."""
    relations = [
        make_enriched_relation("Constipation", "drug_effect", [
            {"patient_group": "Elderly ≥65", "applicability": "Definitely Applicable", "evidence": "anticholinergic"},
        ]),
    ]
    llm_response = json.dumps([
        {"applicability": "Definitely Applicable", "reason": "Patient is 74yo, matches elderly ≥65"}
    ])
    analyzer = PatientContextAnalyzer(make_mock_llm(llm_response))
    result = asyncio.run(analyzer.analyze(relations, "74yo male on clozapine"))
    assert len(result) == 1
    assert result[0]["_patient_relevance"] == "Definitely Applicable"
    assert "74yo" in result[0]["_relevance_reason"]
    assert "context_constraints" not in result[0], "context_constraints should be stripped"
    print("  PASS: Definitely Applicable relation emphasized, constraints stripped")


def test_increased_likelihood_emphasized():
    """Relations matched as 'Increased Likelihood' get _patient_relevance annotation."""
    relations = [
        make_enriched_relation("Weight gain", "drug_effect", [
            {"patient_group": "(general population)", "applicability": "Increased Likelihood", "evidence": "..."},
        ]),
    ]
    llm_response = json.dumps([
        {"applicability": "Increased Likelihood", "reason": "General population applies"}
    ])
    analyzer = PatientContextAnalyzer(make_mock_llm(llm_response))
    result = asyncio.run(analyzer.analyze(relations, "45yo male"))
    assert len(result) == 1
    assert result[0]["_patient_relevance"] == "Increased Likelihood"
    assert "context_constraints" not in result[0], "context_constraints should be stripped"
    print("  PASS: Increased Likelihood relation emphasized, constraints stripped")


def test_not_determinable_stripped():
    """Relations matched as 'Not Determinable' have context_constraints stripped."""
    relations = [
        make_enriched_relation("Rash", "drug_effect", [
            {"patient_group": "Unknown group", "applicability": "Not Determinable", "evidence": ""},
        ]),
    ]
    llm_response = json.dumps([
        {"applicability": "Not Determinable", "reason": "Insufficient evidence"}
    ])
    analyzer = PatientContextAnalyzer(make_mock_llm(llm_response))
    result = asyncio.run(analyzer.analyze(relations, "50yo female"))
    assert len(result) == 1
    assert "context_constraints" not in result[0]
    assert "_patient_relevance" not in result[0]
    print("  PASS: Not Determinable relation stripped of constraints")


def test_mixed_relations():
    """Mix of enriched and plain relations — correct filtering order."""
    relations = [
        make_plain_relation("Headache", "drug_effect"),           # plain, keep
        make_enriched_relation("Constipation", "drug_effect", [   # enriched, remove
            {"patient_group": "Elderly", "applicability": "Definitely NOT Applicable", "evidence": ""},
        ]),
        make_enriched_relation("Seizures", "drug_effect", [       # enriched, emphasize
            {"patient_group": "Elderly", "applicability": "Definitely Applicable", "evidence": ""},
        ]),
        make_plain_relation("Dizziness", "drug_effect"),          # plain, keep
    ]
    llm_response = json.dumps([
        {"applicability": "Definitely NOT Applicable", "reason": "Not elderly"},
        {"applicability": "Definitely Applicable", "reason": "Patient is elderly"},
    ])
    analyzer = PatientContextAnalyzer(make_mock_llm(llm_response))
    result = asyncio.run(analyzer.analyze(relations, "74yo male"))
    assert len(result) == 3  # Headache, Seizures, Dizziness (Constipation removed)
    names = [r["target_name"] for r in result]
    assert names == ["Headache", "Seizures", "Dizziness"]
    assert result[1]["_patient_relevance"] == "Definitely Applicable"
    assert "context_constraints" not in result[1], "context_constraints should be stripped from annotated"
    print("  PASS: mixed relations filtered/emphasized correctly")


def test_json_parse_failure_fallback():
    """If LLM returns invalid JSON, all enriched relations treated as Not Determinable."""
    relations = [
        make_enriched_relation("Constipation", "drug_effect", [
            {"patient_group": "Elderly", "applicability": "Definitely Applicable", "evidence": ""},
        ]),
        make_plain_relation("Nausea", "drug_effect"),
    ]
    analyzer = PatientContextAnalyzer(make_mock_llm("This is not valid JSON at all"))
    result = asyncio.run(analyzer.analyze(relations, "74yo male"))
    # Enriched relation should be kept but stripped of constraints (Not Determinable fallback)
    assert len(result) == 2
    assert "context_constraints" not in result[0]
    assert "_patient_relevance" not in result[0]
    assert result[1]["target_name"] == "Nausea"
    print("  PASS: JSON parse failure falls back to Not Determinable")


def test_json_with_markdown_fences():
    """LLM wraps JSON in ```json ... ``` — should still parse."""
    relations = [
        make_enriched_relation("Seizures", "drug_effect", [
            {"patient_group": "Elderly", "applicability": "Definitely Applicable", "evidence": ""},
        ]),
    ]
    llm_response = '```json\n[{"applicability": "Definitely Applicable", "reason": "matches"}]\n```'
    analyzer = PatientContextAnalyzer(make_mock_llm(llm_response))
    result = asyncio.run(analyzer.analyze(relations, "74yo male"))
    assert len(result) == 1
    assert result[0]["_patient_relevance"] == "Definitely Applicable"
    print("  PASS: JSON with markdown fences parsed correctly")


def test_batching():
    """With >15 enriched relations, analyze() batches LLM calls."""
    # Create 20 enriched relations
    relations = [
        make_enriched_relation(f"Effect_{i}", "drug_effect", [
            {"patient_group": "Elderly", "applicability": "Definitely Applicable", "evidence": ""},
        ])
        for i in range(20)
    ]
    # Mock LLM that records call count and returns correct-sized JSON
    call_count = [0]
    async def counting_llm(messages):
        call_count[0] += 1
        user_msg = messages[-1].content
        # Count [N] entries in the prompt to know batch size
        import re
        indices = re.findall(r'\[(\d+)\]', user_msg)
        n = len(indices)
        return json.dumps([
            {"applicability": "Definitely Applicable", "reason": f"match {i}"}
            for i in range(n)
        ])

    analyzer = PatientContextAnalyzer(counting_llm)
    result = asyncio.run(analyzer.analyze(relations, "74yo male"))
    assert call_count[0] == 2, f"Expected 2 batches (15+5), got {call_count[0]}"
    assert len(result) == 20
    assert all(r.get("_patient_relevance") == "Definitely Applicable" for r in result)
    assert all("context_constraints" not in r for r in result)
    print(f"  PASS: 20 enriched relations processed in {call_count[0]} batches, all annotated + stripped")


def test_batching_partial_failure():
    """If one batch fails JSON parse, only that batch falls back to Not Determinable."""
    relations = [
        make_enriched_relation(f"Effect_{i}", "drug_effect", [
            {"patient_group": "Elderly", "applicability": "Definitely Applicable", "evidence": ""},
        ])
        for i in range(20)
    ]
    call_count = [0]
    async def partial_fail_llm(messages):
        call_count[0] += 1
        if call_count[0] == 1:
            # First batch (0-14): valid JSON
            return json.dumps([
                {"applicability": "Definitely Applicable", "reason": f"match {i}"}
                for i in range(15)
            ])
        else:
            # Second batch (15-19): invalid JSON
            return "INVALID JSON"

    analyzer = PatientContextAnalyzer(partial_fail_llm)
    result = asyncio.run(analyzer.analyze(relations, "74yo male"))
    assert len(result) == 20
    # First 15: annotated
    annotated = [r for r in result if "_patient_relevance" in r]
    assert len(annotated) == 15, f"Expected 15 annotated, got {len(annotated)}"
    # Last 5: plain (Not Determinable fallback)
    plain = [r for r in result if "_patient_relevance" not in r]
    assert len(plain) == 5, f"Expected 5 plain, got {len(plain)}"
    assert all("context_constraints" not in r for r in result)
    print(f"  PASS: partial batch failure — 15 annotated, 5 fell back to Not Determinable")


def test_extract():
    """extract() calls LLM and returns the response."""
    analyzer = PatientContextAnalyzer(make_mock_llm("74yo male with diabetes on metformin"))
    result = asyncio.run(analyzer.extract("A 74-year-old man with type 2 diabetes..."))
    assert "74yo" in result
    print("  PASS: extract() returns LLM response")


# ------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("analyze: no enriched relations", test_no_enriched_relations),
        ("analyze: Definitely NOT Applicable removed", test_definitely_not_applicable_removed),
        ("analyze: Definitely Applicable emphasized", test_definitely_applicable_emphasized),
        ("analyze: Increased Likelihood emphasized", test_increased_likelihood_emphasized),
        ("analyze: Not Determinable stripped", test_not_determinable_stripped),
        ("analyze: mixed relations", test_mixed_relations),
        ("analyze: JSON parse failure fallback", test_json_parse_failure_fallback),
        ("analyze: JSON with markdown fences", test_json_with_markdown_fences),
        ("analyze: batching >15 relations", test_batching),
        ("analyze: partial batch failure", test_batching_partial_failure),
        ("extract: returns LLM response", test_extract),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            print(f"\n[TEST] {name}")
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")

    if failed > 0:
        sys.exit(1)
