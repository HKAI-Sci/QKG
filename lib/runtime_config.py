"""Runtime configuration helpers for public-safe QKG scripts."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path("conf/config.yaml")
EXAMPLE_CONFIG_PATH = Path("conf/config.example.yaml")


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return _expand_env_vars(data)


def _expand_env_vars(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    return value


def load_local_config() -> dict:
    config_path = Path(os.environ.get("QKG_CONFIG", DEFAULT_CONFIG_PATH))
    data = _read_yaml(config_path)
    if data:
        return data
    example = _read_yaml(EXAMPLE_CONFIG_PATH)
    if example:
        return example
    return {}


def apply_optional_aws_env(config: dict | None = None) -> None:
    cfg = config or load_local_config()
    aws = cfg.get("aws") or {}
    mapping = {
        "AWS_ACCESS_KEY_ID": aws.get("access_key_id"),
        "AWS_SECRET_ACCESS_KEY": aws.get("secret_access_key"),
        "AWS_REGION_NAME": aws.get("region"),
    }
    for env_name, env_value in mapping.items():
        if env_value:
            os.environ[env_name] = str(env_value)


def get_mongo_uri(which: str) -> str:
    cfg = load_local_config()
    mongo = cfg.get("mongo") or {}
    env_map = {
        "primekg": os.environ.get("QKG_PRIMEKG_MONGO_URI"),
        "umls": os.environ.get("QKG_UMLS_MONGO_URI"),
    }
    if env_map.get(which):
        return env_map[which]
    key_map = {
        "primekg": "primekg_uri",
        "umls": "umls_uri",
    }
    value = mongo.get(key_map[which])
    if not value:
        raise ValueError(
            f"Missing Mongo URI for '{which}'. Set it in conf/config.yaml or the matching environment variable."
        )
    return str(value)


def get_path_config(key: str) -> str:
    cfg = load_local_config()
    paths = cfg.get("paths") or {}
    env_map = {
        "primekg_csv": os.environ.get("QKG_PRIMEKG_CSV"),
        "umls_mrconso_rrf": os.environ.get("QKG_UMLS_MRCONSO_RRF"),
        "relation_with_facts_jsonl": os.environ.get("QKG_RELATION_FACTS_JSONL"),
        "entity_to_umls_map_json": os.environ.get("QKG_ENTITY_TO_UMLS_MAP_JSON"),
        "qa_eval_jsonl": os.environ.get("QKG_QA_EVAL_JSONL"),
        "primekg_entities_jsonl": os.environ.get("QKG_PRIMEKG_ENTITIES_JSONL"),
    }
    if env_map.get(key):
        return env_map[key]
    value = paths.get(key)
    if not value:
        raise ValueError(
            f"Missing configured path '{key}'. Set it in conf/config.yaml or the matching environment variable."
        )
    return str(value)


def get_optional_path_config(key: str, default: str) -> str:
    cfg = load_local_config()
    paths = cfg.get("paths") or {}
    env_map = {
        "primekg_csv": os.environ.get("QKG_PRIMEKG_CSV"),
        "umls_mrconso_rrf": os.environ.get("QKG_UMLS_MRCONSO_RRF"),
        "relation_with_facts_jsonl": os.environ.get("QKG_RELATION_FACTS_JSONL"),
        "entity_to_umls_map_json": os.environ.get("QKG_ENTITY_TO_UMLS_MAP_JSON"),
        "qa_eval_jsonl": os.environ.get("QKG_QA_EVAL_JSONL"),
        "primekg_entities_jsonl": os.environ.get("QKG_PRIMEKG_ENTITIES_JSONL"),
    }
    return str(env_map.get(key) or paths.get(key) or default)
