"""
Create primeKG.entities collection by:
1. Deriving all unique entities from primeKG.relations
2. Mapping to UMLS CUI via:
   a. Existing entity_to_umls_map.json (name-based mapping prepared offline)
   b. Direct source_code lookup in umls_strings_raw_test
3. If both fail, cui = null
"""

import json
from pymongo import MongoClient, ASCENDING
from collections import defaultdict
from lib.runtime_config import get_mongo_uri, get_path_config

MONGO_URI = get_mongo_uri("primekg")
ENTITY_MAP_FILE = get_path_config("entity_to_umls_map_json")
BATCH_SIZE = 5000

# PrimeKG source -> UMLS source vocabulary + ID transform
SOURCE_MAP = {
    "DrugBank": ("DRUGBANK", lambda x: x),          # DB00001 -> DB00001
    "NCBI":     ("NCBI",     lambda x: x),           # 1 -> 1
    "HPO":      ("HPO",      lambda x: f"HP:{int(x):07d}"),  # 1 -> HP:0000001
    "GO":       ("GO",       lambda x: f"GO:{int(x):07d}"),  # 1 -> GO:0000001
    "REACTOME": ("REACTOME", lambda x: x),           # R-HSA-xxx -> R-HSA-xxx (unlikely to match)
}
# MONDO, UBERON, CTD, MONDO_grouped have no direct UMLS source vocab


def main():
    client = MongoClient(MONGO_URI)
    rel_col = client["primeKG"]["relations"]
    ent_col = client["primeKG"]["entities"]
    umls_col = client["umls_test"]["umls_strings_raw_test"]

    # -------------------------------------------------------
    # Step 0: Ensure source_code index on UMLS
    # -------------------------------------------------------
    existing_indexes = [name for name in umls_col.index_information()]
    if "source_1_source_code_1" not in existing_indexes:
        print("Creating index (source, source_code) on umls_strings_raw_test...")
        umls_col.create_index([("source", 1), ("source_code", 1)], name="source_1_source_code_1")
        print("  Done.")

    # -------------------------------------------------------
    # Step 1: Load existing entity_to_umls_map.json
    # -------------------------------------------------------
    print(f"Loading {ENTITY_MAP_FILE}...")
    with open(ENTITY_MAP_FILE) as f:
        name_to_cui = json.load(f)
    print(f"  {len(name_to_cui)} name->CUI mappings loaded.")

    # -------------------------------------------------------
    # Step 2: Extract all unique entities from relations
    # -------------------------------------------------------
    print("Extracting unique entities from primeKG.relations...")

    entities = {}  # index -> {index, id, type, name, source}

    # x-side
    pipeline_x = [
        {"$group": {
            "_id": "$x_index",
            "id": {"$first": "$x_id"},
            "type": {"$first": "$x_type"},
            "name": {"$first": "$x_name"},
            "source": {"$first": "$x_source"},
        }}
    ]
    for doc in rel_col.aggregate(pipeline_x, allowDiskUse=True):
        idx = doc["_id"]
        entities[idx] = {
            "index": idx,
            "id": doc["id"],
            "type": doc["type"],
            "name": doc["name"],
            "source": doc["source"],
        }

    print(f"  After x-side: {len(entities)} entities")

    # y-side (fill in any missing)
    pipeline_y = [
        {"$group": {
            "_id": "$y_index",
            "id": {"$first": "$y_id"},
            "type": {"$first": "$y_type"},
            "name": {"$first": "$y_name"},
            "source": {"$first": "$y_source"},
        }}
    ]
    for doc in rel_col.aggregate(pipeline_y, allowDiskUse=True):
        idx = doc["_id"]
        if idx not in entities:
            entities[idx] = {
                "index": idx,
                "id": doc["id"],
                "type": doc["type"],
                "name": doc["name"],
                "source": doc["source"],
            }

    print(f"  Total unique entities: {len(entities)}")

    # -------------------------------------------------------
    # Step 3: Map each entity to CUI
    # -------------------------------------------------------
    print("Mapping entities to UMLS CUI...")

    stats = defaultdict(int)
    results = []

    for i, (idx, ent) in enumerate(entities.items()):
        cui = None
        cui_score = None
        cui_method = None

        name = ent["name"]
        source = ent["source"]
        eid = ent["id"]

        # Strategy A: name lookup in entity_to_umls_map.json
        if name in name_to_cui:
            mapping = name_to_cui[name]
            cui = mapping["cui"]
            cui_score = mapping["score"]
            cui_method = "entity_map"
            stats["entity_map"] += 1

        # Strategy B: direct source_code lookup in UMLS
        if cui is None and source in SOURCE_MAP:
            umls_source, id_transform = SOURCE_MAP[source]
            try:
                umls_code = id_transform(eid)
            except (ValueError, TypeError):
                umls_code = str(eid)

            umls_doc = umls_col.find_one(
                {"source": umls_source, "source_code": umls_code},
                {"cui": 1}
            )
            if umls_doc:
                cui = umls_doc["cui"]
                cui_score = 0.0
                cui_method = "source_code"
                stats["source_code"] += 1

        if cui is None:
            stats["unmapped"] += 1

        results.append({
            "index": ent["index"],
            "id": ent["id"],
            "type": ent["type"],
            "name": ent["name"],
            "source": ent["source"],
            "cui": cui,
            "cui_score": cui_score,
            "cui_method": cui_method,
        })

        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(entities)} mapped...")

    print(f"\nMapping stats:")
    for method, count in sorted(stats.items()):
        print(f"  {method}: {count}")

    # -------------------------------------------------------
    # Step 4: Write to primeKG.entities
    # -------------------------------------------------------
    print(f"\nDropping existing primeKG.entities...")
    ent_col.drop()

    print(f"Inserting {len(results)} entities...")
    for i in range(0, len(results), BATCH_SIZE):
        batch = results[i:i + BATCH_SIZE]
        ent_col.insert_many(batch)
        if (i + BATCH_SIZE) % 50000 == 0 or i + BATCH_SIZE >= len(results):
            print(f"  {min(i + BATCH_SIZE, len(results))}/{len(results)} inserted")

    # -------------------------------------------------------
    # Step 5: Create indexes
    # -------------------------------------------------------
    print("Creating indexes...")
    ent_col.create_index("index", unique=True, name="index_1")
    ent_col.create_index([("source", 1), ("id", 1)], name="source_1_id_1")
    ent_col.create_index("name", name="name_1")
    ent_col.create_index("cui", name="cui_1")
    ent_col.create_index("type", name="type_1")
    print("  Done.")

    # -------------------------------------------------------
    # Verify
    # -------------------------------------------------------
    count = ent_col.count_documents({})
    mapped = ent_col.count_documents({"cui": {"$ne": None}})
    print(f"\nVerification: {count} entities, {mapped} with CUI ({mapped/count*100:.1f}%)")


if __name__ == "__main__":
    main()
