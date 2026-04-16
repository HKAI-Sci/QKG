"""Patient context analysis for constraint-aware KG reasoning."""

import asyncio
import json
from lib.react_agent import Message


class PatientContextAnalyzer:
    """Uses LLM to extract patient context and filter KG relations by applicability."""

    def __init__(self, llm_call):
        """
        Args:
            llm_call: async (list[Message]) -> str
        """
        self.llm_call = llm_call

    async def extract(self, question: str) -> str:
        """Extract patient characteristics from a medical question."""
        messages = [
            Message(role="system", content=(
                "Extract the key patient characteristics from this medical question. "
                "Include: age, sex, medical history, current medications, symptoms, "
                "lab values, vital signs. Output a concise summary in 2-3 sentences."
            )),
            Message(role="user", content=question),
        ]
        return await self.llm_call(messages)

    _BATCH_SIZE = 15  # max relations per LLM analysis call

    async def analyze(self, relations: list[dict], patient_context: str) -> list[dict]:
        """Filter/emphasize relations based on patient-specific applicability.

        For each enriched relation (has context_constraints):
        - Definitely NOT Applicable → removed from result
        - Definitely Applicable / Increased Likelihood → kept with _patient_relevance annotation
        - Not Determinable → treated as plain relation

        All kept relations have context_constraints stripped (analysis is
        captured in _patient_relevance / _relevance_reason instead).
        Non-enriched relations pass through unchanged.
        """
        enriched = [r for r in relations if "context_constraints" in r]
        if not enriched:
            return relations

        # Run all batches concurrently (per-agent semaphore controls LLM parallelism)
        batch_coros = []
        for batch_start in range(0, len(enriched), self._BATCH_SIZE):
            batch = enriched[batch_start:batch_start + self._BATCH_SIZE]
            batch_coros.append(self._analyze_batch(batch, batch_start, patient_context))
        batch_results = await asyncio.gather(*batch_coros)
        matched = {}
        for br in batch_results:
            matched.update(br)

        # Apply filtering/emphasis
        filtered = []
        enriched_set = {id(r) for r in enriched}
        enriched_idx = 0

        for r in relations:
            if id(r) not in enriched_set:
                filtered.append(r)
                continue

            decision = matched.get(enriched_idx, {})
            applicability = decision.get("applicability", "Not Determinable")
            reason = decision.get("reason", "")
            enriched_idx += 1

            if applicability == "Definitely NOT Applicable":
                continue  # remove — don't contaminate agent memory
            elif applicability in ("Definitely Applicable", "Increased Likelihood"):
                r.pop("context_constraints", None)
                r["_patient_relevance"] = applicability
                r["_relevance_reason"] = reason
                filtered.append(r)
            else:
                r.pop("context_constraints", None)
                filtered.append(r)

        return filtered

    async def analyze_drug_disease(
        self, relations: list[dict], patient_context: str, requested_relation: str = None
    ) -> list[dict]:
        """Reclassify drug-disease relations for a specific patient.

        Single-step judgment: LLM outputs suggested_relation per relation.
        - no_relation → removed
        - Others → kept (if requested_relation is None, keep all; otherwise filter to match)
        Non-enriched relations pass through with PrimeKG's original label.
        """
        enriched = [r for r in relations if "context_constraints" in r]
        if not enriched:
            if requested_relation:
                return [r for r in relations if r.get("relation") == requested_relation]
            return relations

        # Run all batches concurrently (per-agent semaphore controls LLM parallelism)
        batch_coros = []
        for batch_start in range(0, len(enriched), self._BATCH_SIZE):
            batch = enriched[batch_start:batch_start + self._BATCH_SIZE]
            batch_coros.append(
                self._analyze_drug_disease_batch(batch, batch_start, patient_context)
            )
        batch_results = await asyncio.gather(*batch_coros)
        matched = {}
        for br in batch_results:
            matched.update(br)

        # Apply judgments
        filtered = []
        enriched_set = {id(r) for r in enriched}
        enriched_idx = 0

        for r in relations:
            if id(r) not in enriched_set:
                # Plain relation — keep if matches requested, or keep all
                if requested_relation and r.get("relation") != requested_relation:
                    continue
                filtered.append(r)
                continue

            decision = matched.get(enriched_idx, {})
            suggested = decision.get("suggested_relation", r.get("relation", "no_relation"))
            reason = decision.get("reason", "")
            enriched_idx += 1

            if suggested == "no_relation":
                continue  # remove

            r.pop("context_constraints", None)
            r["relation"] = suggested  # overwrite with reclassified type
            r["_relevance_reason"] = reason

            if requested_relation and suggested != requested_relation:
                continue  # reclassified to different type than requested

            filtered.append(r)

        return filtered

    async def _analyze_drug_disease_batch(
        self, batch: list[dict], offset: int, patient_context: str
    ) -> dict[int, dict]:
        """Run LLM analysis on a batch of enriched drug-disease relations.

        Returns {global_index: {"suggested_relation": ..., "reason": ...}}.
        """
        constraint_lines = []
        for i, r in enumerate(batch):
            # Build entity names for context
            x_name = r.get("x_name", r.get("target_name", "?"))
            y_name = r.get("y_name", r.get("target_name", "?"))
            parts = [f"[{i}] {x_name} → {y_name} (PrimeKG label: {r['relation']}):"]
            for j, c in enumerate(r.get("context_constraints", [])):
                pg = c.get("patient_group") or "(general population)"
                parts.append(f"  {j}. [{c['applicability']}] {pg}")
                if c.get("evidence"):
                    parts.append(f"     Evidence: {c['evidence']}")
            constraint_lines.append("\n".join(parts))

        messages = [
            Message(role="system", content=(
                "You are a medical analyst. Given a patient description and drug-disease "
                "relations with patient-group-specific constraints, determine the most "
                "accurate relationship type for this specific patient.\n\n"
                "For each relation, choose exactly one:\n"
                '- "indication": the drug is a treatment for this condition in this patient\n'
                '- "contraindication": the drug should NOT be used for this patient\'s condition\n'
                '- "off-label use": the drug is used for this condition outside approved indications\n'
                '- "no_relation": no clinically meaningful relationship exists for this patient\n\n'
                "Output a JSON list with one object per relation, in order:\n"
                '[{"suggested_relation": "indication", "reason": "..."}, ...]\n\n'
                "Rules:\n"
                "- Consider the patient's specific characteristics when choosing the relation type\n"
                "- If the patient matches a specific patient_group, use that group's context\n"
                "- If no patient_group matches, use the general population entry\n"
                "- Output ONLY the JSON list, nothing else"
            )),
            Message(role="user", content=(
                f"Patient:\n{patient_context}\n\n"
                f"Relations with constraints:\n" + "\n\n".join(constraint_lines)
            )),
        ]
        raw = await self.llm_call(messages)

        result = {}
        try:
            cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
            items = json.loads(cleaned)
            for i, item in enumerate(items):
                result[offset + i] = item
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # fallback: keep PrimeKG's original labels

        return result

    async def _analyze_batch(
        self, batch: list[dict], offset: int, patient_context: str
    ) -> dict[int, dict]:
        """Run LLM analysis on a batch of enriched relations.

        Returns {global_index: {"applicability": ..., "reason": ...}}.
        """
        constraint_lines = []
        for i, r in enumerate(batch):
            parts = [f"[{i}] {r['target_name']} ({r['relation']}):"]
            for j, c in enumerate(r.get("context_constraints", [])):
                pg = c.get("patient_group") or "(general population)"
                parts.append(f"  {j}. [{c['applicability']}] {pg}")
                if c.get("evidence"):
                    parts.append(f"     Evidence: {c['evidence']}")
            constraint_lines.append("\n".join(parts))

        messages = [
            Message(role="system", content=(
                "You are a medical analyst. Given a patient description and "
                "knowledge graph relations with patient-group-specific constraints, "
                "determine which patient_group best matches this patient for each relation.\n\n"
                "Output a JSON list with one object per relation, in order:\n"
                '[{"applicability": "Definitely Applicable", "reason": "..."}, ...]\n\n'
                "Rules:\n"
                '- "applicability" must be copied exactly from the best-matching '
                "patient_group's applicability value\n"
                '- "reason" is a brief sentence explaining why this patient_group '
                "matches the patient\n"
                "- If no patient_group matches, use the general population entry\n"
                "- Output ONLY the JSON list, nothing else"
            )),
            Message(role="user", content=(
                f"Patient:\n{patient_context}\n\n"
                f"Relations with constraints:\n" + "\n\n".join(constraint_lines)
            )),
        ]
        raw = await self.llm_call(messages)

        result = {}
        try:
            cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
            items = json.loads(cleaned)
            for i, item in enumerate(items):
                result[offset + i] = item
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # fallback: all Not Determinable

        return result
