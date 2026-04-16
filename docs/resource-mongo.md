# MongoDB Resources

This document lists only the MongoDB collections required by the public reproduction path described in [README.md](../README.md).

## Database: `primeKG`

### Collection: `relations`

Loaded from official `PrimeKg.csv`.

Purpose:
- base PrimeKG relation graph used during path search and neighborhood lookup

Key fields:
- `relation`
- `display_relation`
- `x_index`
- `x_id`
- `x_type`
- `x_name`
- `x_source`
- `y_index`
- `y_id`
- `y_type`
- `y_name`
- `y_source`

### Collection: `entities`

Loaded from the QKG-published `qkg-primekg-entities-with-cui` artifact.

Purpose:
- unique PrimeKG entities with QKG-provided UMLS CUI annotations for entity lookup

Key fields:
- `index`
- `id`
- `type`
- `name`
- `source`
- `cui`
- `cui_score`
- `cui_method`

### Collection: `relation_with_facts`

Loaded from the QKG-published `qkg-relation-with-facts` artifact.

Purpose:
- patient-aware fact annotations retrieved at inference time for relation filtering

## Database: `umls_test`

### Collection: `umls_strings_raw_test`

Loaded from official UMLS `MRCONSO.RRF` via `python tools/import_umls_strings.py`.

Purpose:
- UMLS synonym and abbreviation fallback used by `search_entity`

Key fields:
- `_id`
- `cui`
- `language`
- `source`
- `source_code`
- `source_name`
- `source_term_type`
- `aui`

For setup commands, use the steps in [README.md](../README.md).
