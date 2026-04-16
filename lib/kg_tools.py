"""KG tool functions for PrimeKG + UMLS fallback.

Used by conditionKgTestAgentic.py (agentic eval) and any future consumers.
"""

import re
from pymongo import MongoClient
from lib.runtime_config import get_mongo_uri

# ============================================================
# MongoDB (PrimeKG)
# ============================================================

MONGO_URI = get_mongo_uri("primekg")

_mongo_client = None
_mongo_db = None


def get_db():
    global _mongo_client, _mongo_db
    if _mongo_db is None:
        _mongo_client = MongoClient(MONGO_URI)
        _mongo_db = _mongo_client["primeKG"]
    return _mongo_db


# ============================================================
# MongoDB (UMLS)
# ============================================================

UMLS_MONGO_URI = get_mongo_uri("umls")

_umls_client = None
_umls_db = None


def get_umls_db():
    global _umls_client, _umls_db
    if _umls_db is None:
        _umls_client = MongoClient(UMLS_MONGO_URI)
        _umls_db = _umls_client["umls_test"]
    return _umls_db


DRUG_DISEASE_RELATIONS = {"indication", "contraindication", "off-label use"}

# ============================================================
# KG Tool Functions
# ============================================================


def _umls_fallback_search(query: str, type: str = None, limit: int = 10) -> list[dict]:
    """Search UMLS for synonyms, return matching PrimeKG entities via CUI bridge."""
    umls_col = get_umls_db()["umls_strings_raw_test"]
    pkg_col = get_db()["entities"]

    # Search UMLS by source_name — exact match first, then substring
    umls_results = list(umls_col.find(
        {"source_name": {"$regex": f"^{re.escape(query)}$", "$options": "i"}},
        {"cui": 1, "_id": 0}
    ).limit(50))
    if not umls_results:
        umls_results = list(umls_col.find(
            {"source_name": {"$regex": re.escape(query), "$options": "i"}},
            {"cui": 1, "_id": 0}
        ).limit(50))
    if not umls_results:
        return []

    cuis = list({r["cui"] for r in umls_results if r.get("cui")})
    if not cuis:
        return []

    # Lookup PrimeKG entities with matching CUIs
    pkg_filter = {"cui": {"$in": cuis}}
    if type:
        pkg_filter["type"] = type
    projection = {"_id": 0, "index": 1, "name": 1, "type": 1, "source": 1, "id": 1, "cui": 1}
    results = list(pkg_col.find(pkg_filter, projection).limit(limit))

    for r in results:
        r["_matched_via"] = "umls_synonym"
    return results


def tool_search_entity(query: str, type: str = None, source: str = None, limit: int = 10) -> list[dict]:
    db = get_db()
    col = db["entities"]
    projection = {"_id": 0, "index": 1, "name": 1, "type": 1, "source": 1, "id": 1, "cui": 1}

    # Try exact match first (case-insensitive)
    exact_filter = {"name": {"$regex": f"^{re.escape(query)}$", "$options": "i"}}
    if type:
        exact_filter["type"] = type
    if source:
        exact_filter["source"] = source
    exact = list(col.find(exact_filter, projection).limit(1))
    if exact:
        return exact

    # Fall back to substring match
    mongo_filter = {"name": {"$regex": re.escape(query), "$options": "i"}}
    if type:
        mongo_filter["type"] = type
    if source:
        mongo_filter["source"] = source
    results = list(col.find(mongo_filter, projection).limit(limit))
    if results:
        return results

    # UMLS fallback: search synonyms via CUI bridge
    return _umls_fallback_search(query, type=type, limit=limit)


def tool_get_entity_relations(entity_index: int, relation_type: str = None, limit: int = 20) -> list[dict]:
    db = get_db()
    col = db["relations"]
    results = []

    # Outgoing
    q_out = {"x_index": entity_index}
    if relation_type:
        q_out["relation"] = relation_type
    for doc in col.find(q_out).limit(limit):
        results.append({
            "relation": doc["relation"],
            "display_relation": doc.get("display_relation", ""),
            "target_index": doc["y_index"],
            "target_name": doc["y_name"],
            "target_type": doc["y_type"],
            "direction": "outgoing",
        })

    # Incoming
    remaining = limit - len(results)
    if remaining > 0:
        q_in = {"y_index": entity_index}
        if relation_type:
            q_in["relation"] = relation_type
        for doc in col.find(q_in).limit(remaining):
            results.append({
                "relation": doc["relation"],
                "display_relation": doc.get("display_relation", ""),
                "target_index": doc["x_index"],
                "target_name": doc["x_name"],
                "target_type": doc["x_type"],
                "direction": "incoming",
            })

    return results


def _enrich_relations(results: list[dict]) -> list[dict]:
    """Batch-lookup relation_with_facts and merge context_constraints."""
    db = get_db()
    tuples_to_lookup = []
    for r in results:
        tuples_to_lookup.append((r.pop("_x"), r.pop("_y"), r["relation"]))

    if not tuples_to_lookup:
        return results

    fact_query = {"$or": [
        {"x_index": x, "y_index": y, "relation": rel}
        for x, y, rel in tuples_to_lookup
    ]}
    facts = {
        (f["x_index"], f["y_index"], f["relation"]): f
        for f in db["relation_with_facts"].find(fact_query, {"_id": 0, "audit_reasoning": 0})
    }

    for r, (x, y, rel) in zip(results, tuples_to_lookup):
        fact = facts.get((x, y, rel))
        if fact:
            r["context_constraints"] = [
                {
                    "patient_group": c.get("patient_characteristics"),
                    "applicability": c["applicability"],
                    "evidence": c.get("evidence", ""),
                }
                for c in fact.get("relation_constraints", [])
            ]

    return results


def tool_get_entity_relations_with_context(entity_index: int, relation_type: str = None, limit: int = 20) -> list[dict]:
    db = get_db()
    col = db["relations"]
    results = []

    # For drug-disease sub-types: fetch ALL drug-disease edges (reclassification needs the full set)
    is_drug_disease = relation_type in DRUG_DISEASE_RELATIONS
    if is_drug_disease:
        query_relation = {"$in": list(DRUG_DISEASE_RELATIONS)}
        query_limit = 0  # no limit — fetch all, filter after reclassification
    else:
        query_relation = relation_type
        query_limit = limit

    # Outgoing
    q_out = {"x_index": entity_index}
    if query_relation:
        q_out["relation"] = query_relation
    cursor_out = col.find(q_out)
    if query_limit:
        cursor_out = cursor_out.limit(query_limit)
    for doc in cursor_out:
        results.append({
            "relation": doc["relation"],
            "display_relation": doc.get("display_relation", ""),
            "target_index": doc["y_index"],
            "target_name": doc["y_name"],
            "target_type": doc["y_type"],
            "direction": "outgoing",
            "_x": entity_index,
            "_y": doc["y_index"],
        })

    # Incoming
    remaining = (query_limit - len(results)) if query_limit else 0
    if query_limit == 0 or remaining > 0:
        q_in = {"y_index": entity_index}
        if query_relation:
            q_in["relation"] = query_relation
        cursor_in = col.find(q_in)
        if query_limit:
            cursor_in = cursor_in.limit(remaining)
        for doc in cursor_in:
            results.append({
                "relation": doc["relation"],
                "display_relation": doc.get("display_relation", ""),
                "target_index": doc["x_index"],
                "target_name": doc["x_name"],
                "target_type": doc["x_type"],
                "direction": "incoming",
                "_x": doc["x_index"],
                "_y": entity_index,
            })

    results = _enrich_relations(results)

    # Tag drug-disease results so result_hook knows to reclassify
    if is_drug_disease:
        for r in results:
            r["_requested_relation"] = relation_type

    return results


def tool_check_relation(entity_a_index: int, entity_b_index: int) -> list[dict]:
    """Check all relations between two specific entities."""
    db = get_db()
    col = db["relations"]
    results = []

    for doc in col.find({"$or": [
        {"x_index": entity_a_index, "y_index": entity_b_index},
        {"x_index": entity_b_index, "y_index": entity_a_index},
    ]}):
        results.append({
            "relation": doc["relation"],
            "display_relation": doc.get("display_relation", ""),
            "x_index": doc["x_index"],
            "x_name": doc["x_name"],
            "x_type": doc["x_type"],
            "y_index": doc["y_index"],
            "y_name": doc["y_name"],
            "y_type": doc["y_type"],
        })

    return results


def tool_check_relation_with_context(entity_a_index: int, entity_b_index: int) -> list[dict]:
    """Check all relations between two entities, enriched with patient-group constraints."""
    db = get_db()
    col = db["relations"]
    results = []

    for doc in col.find({"$or": [
        {"x_index": entity_a_index, "y_index": entity_b_index},
        {"x_index": entity_b_index, "y_index": entity_a_index},
    ]}):
        results.append({
            "relation": doc["relation"],
            "display_relation": doc.get("display_relation", ""),
            "x_index": doc["x_index"],
            "x_name": doc["x_name"],
            "x_type": doc["x_type"],
            "y_index": doc["y_index"],
            "y_name": doc["y_name"],
            "y_type": doc["y_type"],
            "_x": doc["x_index"],
            "_y": doc["y_index"],
        })

    return _enrich_relations(results)


def tool_list_relation_types() -> dict:
    """Return all PrimeKG relation types, grouped by category. Static — PrimeKG schema is fixed."""
    return {
        "drug_relations": {
            "indication":            {"endpoints": "drug <-> disease",          "edges": 18776,   "description": "Approved therapeutic uses"},
            "contraindication":      {"endpoints": "drug <-> disease",          "edges": 61350,   "description": "Conditions where the drug should NOT be used"},
            "off-label use":         {"endpoints": "drug <-> disease",          "edges": 5136,    "description": "Non-approved but clinically used indications"},
            "drug_effect":           {"endpoints": "drug <-> effect/phenotype", "edges": 129568,  "description": "Adverse drug reactions / side effects (NOTE: use 'drug_effect', NOT 'side_effect')"},
            "drug_protein":          {"endpoints": "drug <-> gene/protein",     "edges": 51306,   "description": "Drug-protein interactions (display_relation: target/enzyme/transporter/carrier)"},
            "drug_drug":             {"endpoints": "drug <-> drug",             "edges": 2672628, "description": "Drug-drug synergistic interactions"},
        },
        "disease_relations": {
            "disease_phenotype_positive": {"endpoints": "disease <-> effect/phenotype", "edges": 300634, "description": "Phenotypes/symptoms associated WITH the disease"},
            "disease_phenotype_negative": {"endpoints": "disease <-> effect/phenotype", "edges": 2386,   "description": "Phenotypes NOT associated with the disease"},
            "disease_protein":            {"endpoints": "disease <-> gene/protein",     "edges": 160822, "description": "Disease-gene associations"},
            "disease_disease":            {"endpoints": "disease <-> disease",           "edges": 64388,  "description": "Disease ontology hierarchy (parent-child)"},
        },
        "protein_relations": {
            "protein_protein":   {"endpoints": "gene/protein <-> gene/protein",       "edges": 642150, "description": "Protein-protein interactions"},
            "bioprocess_protein": {"endpoints": "biological_process <-> gene/protein", "edges": 289610, "description": "Gene-biological process annotation"},
            "molfunc_protein":   {"endpoints": "molecular_function <-> gene/protein",  "edges": 139060, "description": "Gene-molecular function annotation"},
            "cellcomp_protein":  {"endpoints": "cellular_component <-> gene/protein",  "edges": 166804, "description": "Gene-cellular component annotation"},
            "pathway_protein":   {"endpoints": "pathway <-> gene/protein",             "edges": 85292,  "description": "Gene-pathway membership"},
            "phenotype_protein": {"endpoints": "effect/phenotype <-> gene/protein",    "edges": 6660,   "description": "Phenotype-gene associations"},
        },
        "anatomy_relations": {
            "anatomy_protein_present": {"endpoints": "anatomy <-> gene/protein", "edges": 3036406, "description": "Gene expressed in tissue/organ"},
            "anatomy_protein_absent":  {"endpoints": "anatomy <-> gene/protein", "edges": 39774,   "description": "Gene NOT expressed in tissue/organ"},
            "anatomy_anatomy":         {"endpoints": "anatomy <-> anatomy",       "edges": 28064,   "description": "Anatomical ontology hierarchy"},
        },
        "ontology_hierarchy": {
            "bioprocess_bioprocess": {"endpoints": "biological_process <-> biological_process", "edges": 105772, "description": "GO hierarchy"},
            "molfunc_molfunc":       {"endpoints": "molecular_function <-> molecular_function", "edges": 27148,  "description": "GO hierarchy"},
            "cellcomp_cellcomp":     {"endpoints": "cellular_component <-> cellular_component", "edges": 9690,   "description": "GO hierarchy"},
            "phenotype_phenotype":   {"endpoints": "effect/phenotype <-> effect/phenotype",     "edges": 37472,  "description": "HPO hierarchy"},
            "pathway_pathway":       {"endpoints": "pathway <-> pathway",                       "edges": 5070,   "description": "Reactome hierarchy"},
        },
        "exposure_relations": {
            "exposure_disease":    {"endpoints": "exposure <-> disease",            "edges": 4608, "description": "Exposure-disease links"},
            "exposure_protein":    {"endpoints": "exposure <-> gene/protein",       "edges": 2424, "description": "Exposure-protein interactions"},
            "exposure_bioprocess": {"endpoints": "exposure <-> biological_process", "edges": 3250, "description": "Exposure-process interactions"},
            "exposure_molfunc":    {"endpoints": "exposure <-> molecular_function", "edges": 90,   "description": "Exposure-function interactions"},
            "exposure_cellcomp":   {"endpoints": "exposure <-> cellular_component", "edges": 20,   "description": "Exposure-component interactions"},
            "exposure_exposure":   {"endpoints": "exposure <-> exposure",            "edges": 4140, "description": "Exposure ontology hierarchy"},
        },
    }
