"""Import relation_facts_all_cleaned.jsonl into MongoDB collection relation_with_facts."""
import json
from pymongo import MongoClient, IndexModel
from lib.runtime_config import get_mongo_uri, get_path_config

MONGO_URI = get_mongo_uri("primekg")
INPUT_FILE = get_path_config("relation_with_facts_jsonl")
COLLECTION = "relation_with_facts"
BATCH_SIZE = 1000

client = MongoClient(MONGO_URI)
db = client["primeKG"]

# Drop existing collection if any
if COLLECTION in db.list_collection_names():
    db[COLLECTION].drop()
    print(f"Dropped existing '{COLLECTION}' collection")

col = db[COLLECTION]

batch = []
total = 0

with open(INPUT_FILE) as f:
    for line in f:
        d = json.loads(line)
        t = d["triplet"]

        doc = {
            "x_index": int(t["x"]["index"]),
            "y_index": int(t["y"]["index"]),
            "relation": t["relation"],
            "display_relation": t.get("display_relation", ""),
            "corrected_relation": t.get("corrected_relation", ""),
            "x_name": t["x"]["name"],
            "y_name": t["y"]["name"],
            "x_type": t["x"]["type"],
            "y_type": t["y"]["type"],
            "audit_reasoning": d.get("audit_reasoning", ""),
            "relation_constraints": d.get("relation_constraints", []),
        }

        batch.append(doc)
        if len(batch) >= BATCH_SIZE:
            col.insert_many(batch)
            total += len(batch)
            batch = []
            if total % 10000 == 0:
                print(f"  Inserted {total}...")

if batch:
    col.insert_many(batch)
    total += len(batch)

print(f"\nInserted {total} documents into '{COLLECTION}'")

# Create indexes
print("Creating indexes...")
col.create_indexes([
    IndexModel([("x_index", 1), ("y_index", 1), ("relation", 1)]),
    IndexModel([("y_index", 1), ("x_index", 1), ("relation", 1)]),
    IndexModel([("corrected_relation", 1)]),
])
print("Indexes created.")

# Verify
count = col.count_documents({})
print(f"Verification: {count} documents in '{COLLECTION}'")

# Spot check
doc = col.find_one({"x_index": 15003, "y_index": 36562, "relation": "indication"})
if doc:
    print(f"Spot check OK: {doc['x_name']} -> {doc['y_name']}, corrected: {doc['corrected_relation']}")
else:
    print("Spot check FAILED: expected Cortisone acetate / exostosis not found")
