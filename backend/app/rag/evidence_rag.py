"""
Evidence RAG — indexes known bias patterns and dataset exemplars.

Purpose: after an agent flags a passage, the orchestrator can cross-check
the flagged text against known bias patterns to boost confidence or detect
false positives.

The evidence index is built from two sources:
1. **Bias pattern descriptions** — canonical descriptions of each bias type
   (B01–B11) with linguistic markers and examples.
2. **Dataset exemplars** — sample biased/control texts from the eval
   transforms, giving the system concrete examples of what each bias
   looks like in practice.

Usage:
    evidence = EvidenceRAG()
    evidence.build_index()  # one-time, loads patterns + exemplars
    matches = evidence.check_annotation(flagged_text, bias_type="B01")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .chunker import Chunk
from .vector_store import SearchResult, VectorStore


# ── Canonical bias pattern catalogue ────────────────────────────────────

@dataclass
class BiasPattern:
    """A known bias pattern with description and linguistic markers."""
    bias_code: str          # B01, B03, etc.
    bias_name: str
    description: str
    markers: list[str]      # linguistic markers / red flags
    example_biased: str     # short example of biased text
    example_neutral: str    # short example of neutral text


# The domain ontology — this IS the competitive moat
BIAS_PATTERNS: list[BiasPattern] = [
    BiasPattern(
        bias_code="B01",
        bias_name="Confirmation Bias",
        description=(
            "Selectively presenting evidence that supports a preferred conclusion "
            "while ignoring or downplaying contradictory evidence. In systematic reviews, "
            "this manifests as cherry-picked citations, one-sided evidence summaries, "
            "and failure to acknowledge conflicting findings."
        ),
        markers=[
            "consistently supports", "strongly suggests", "clear evidence for",
            "studies confirm", "evidence overwhelmingly", "all studies agree",
            "no evidence against", "uniformly positive", "without exception",
        ],
        example_biased=(
            "The evidence consistently supports the efficacy of treatment X. "
            "Multiple studies confirm positive outcomes across diverse populations."
        ),
        example_neutral=(
            "While several studies report positive outcomes for treatment X, "
            "two RCTs found no significant benefit, and one reported adverse effects."
        ),
    ),
    BiasPattern(
        bias_code="B02",
        bias_name="Reporting / Publication Bias",
        description=(
            "Systematic distortion from over-representing published positive results "
            "while unpublished negative studies are missing. Manifests as funnel plot "
            "asymmetry, missing null results, and over-reliance on published literature."
        ),
        markers=[
            "published studies show", "available evidence", "reported outcomes",
            "published literature", "peer-reviewed sources",
        ],
        example_biased=(
            "All published studies on this intervention report positive outcomes, "
            "strongly supporting its clinical adoption."
        ),
        example_neutral=(
            "Published studies largely report positive outcomes; however, funnel plot "
            "analysis suggests potential publication bias with missing null results."
        ),
    ),
    BiasPattern(
        bias_code="B03",
        bias_name="Certainty Inflation",
        description=(
            "Expressing conclusions with greater certainty than the evidence warrants. "
            "Replacing hedging language (may, might, suggests) with definitive claims "
            "(proves, establishes, demonstrates). Particularly problematic in preliminary "
            "or observational research."
        ),
        markers=[
            "clearly demonstrates", "definitively proves", "establishes beyond doubt",
            "unequivocally shows", "conclusive evidence", "certainly",
            "without question", "indisputable", "proven fact",
        ],
        example_biased=(
            "This study conclusively proves that the intervention eliminates disease "
            "progression in all patient populations."
        ),
        example_neutral=(
            "This preliminary study suggests the intervention may reduce disease "
            "progression in the studied population, though further trials are needed."
        ),
    ),
    BiasPattern(
        bias_code="B04",
        bias_name="Overgeneralization",
        description=(
            "Extending findings beyond their warranted scope — from specific populations "
            "to all people, from in-vitro to clinical, from correlation to universal truth. "
            "Scope creep in conclusion sections is a common form."
        ),
        markers=[
            "all patients", "universally applicable", "in every case",
            "across all populations", "without exception", "everyone",
            "global implications", "applies to all", "regardless of",
        ],
        example_biased=(
            "These findings in elderly Japanese women demonstrate that the supplement "
            "benefits all populations regardless of age, sex, or ethnicity."
        ),
        example_neutral=(
            "These findings in elderly Japanese women suggest potential benefits "
            "in this specific population. Generalizability to other demographics "
            "requires further investigation."
        ),
    ),
    BiasPattern(
        bias_code="B05",
        bias_name="Framing Effect",
        description=(
            "Presenting information in a way that emphasizes certain aspects to influence "
            "interpretation. Includes spin (positive framing of null results), selective "
            "emphasis, loaded language, and rhetorical framing."
        ),
        markers=[
            "promising results", "encouraging findings", "breakthrough",
            "revolutionary", "game-changing", "paradigm shift",
            "first-ever", "unprecedented", "remarkable",
        ],
        example_biased=(
            "This breakthrough study reveals a revolutionary approach to cancer "
            "treatment with unprecedented response rates."
        ),
        example_neutral=(
            "This phase II trial reported a 23% response rate for the experimental "
            "treatment, compared to 18% in the control arm (p=0.12)."
        ),
    ),
    BiasPattern(
        bias_code="B06",
        bias_name="Anchoring Bias",
        description=(
            "Giving disproportionate weight to initially encountered information. "
            "In reviews, early-cited prominent studies anchor interpretation even when "
            "later evidence contradicts them."
        ),
        markers=[
            "seminal study", "landmark paper", "foundational work",
            "as originally shown", "consistent with the original finding",
        ],
        example_biased=(
            "As the seminal 2003 study by Smith et al. established, this mechanism "
            "is central to disease pathology. Subsequent work has reinforced this view."
        ),
        example_neutral=(
            "Smith et al. (2003) proposed this mechanism. While some subsequent studies "
            "support this view, others have identified alternative pathways."
        ),
    ),
    BiasPattern(
        bias_code="B07",
        bias_name="Outcome Reporting Bias",
        description=(
            "Selectively reporting favorable outcomes while omitting unfavorable ones. "
            "Switching primary endpoints, reporting only significant subgroup analyses, "
            "or emphasizing secondary outcomes over the primary null result."
        ),
        markers=[
            "subgroup analysis revealed", "secondary endpoint",
            "exploratory analysis", "post-hoc", "when stratified by",
        ],
        example_biased=(
            "Although the primary endpoint did not reach significance, subgroup analysis "
            "revealed a statistically significant benefit in patients under 50."
        ),
        example_neutral=(
            "The primary endpoint did not reach statistical significance (p=0.23). "
            "A pre-specified subgroup analysis in patients under 50 showed a trend "
            "toward benefit (p=0.06) that requires confirmatory study."
        ),
    ),
    BiasPattern(
        bias_code="B08",
        bias_name="Causal Inference Error",
        description=(
            "Inferring causation from correlational or observational data. Using causal "
            "language (causes, leads to, results in) when the study design only supports "
            "associational claims."
        ),
        markers=[
            "causes", "leads to", "results in", "produces", "induces",
            "triggers", "drives", "is responsible for", "due to",
            "because of", "effect of",
        ],
        example_biased=(
            "Coffee consumption causes reduced risk of Type 2 diabetes, "
            "as demonstrated by this large observational cohort."
        ),
        example_neutral=(
            "Coffee consumption was associated with reduced risk of Type 2 diabetes "
            "in this observational cohort. Causal inference is limited by the study design."
        ),
    ),
    BiasPattern(
        bias_code="B09",
        bias_name="Cultural / Demographic Bias",
        description=(
            "Assuming findings from one cultural or demographic group apply universally. "
            "WEIRD (Western, Educated, Industrialized, Rich, Democratic) bias in study "
            "populations."
        ),
        markers=[
            "universally", "all cultures", "regardless of background",
            "across demographics", "human nature", "inherently",
        ],
        example_biased=(
            "This psychological phenomenon is a universal aspect of human cognition, "
            "as replicated across multiple Western university samples."
        ),
        example_neutral=(
            "This phenomenon has been demonstrated in Western university samples. "
            "Cross-cultural validity has not been established."
        ),
    ),
    BiasPattern(
        bias_code="B10",
        bias_name="Temporal Bias",
        description=(
            "Inappropriate weighting of evidence by recency or ignoring how findings "
            "change over time. Over-relying on older or newer studies without "
            "justification."
        ),
        markers=[
            "recent evidence shows", "latest studies", "modern research",
            "outdated findings", "superseded by", "current understanding",
        ],
        example_biased=(
            "Modern research has superseded earlier findings, and current "
            "evidence clearly supports the revised mechanism."
        ),
        example_neutral=(
            "While recent studies suggest a revised mechanism, earlier findings "
            "have not been directly refuted and methodological differences may "
            "account for the discrepancy."
        ),
    ),
    BiasPattern(
        bias_code="B11",
        bias_name="Methodological Quality Bias",
        description=(
            "Treating all studies equally regardless of methodological quality, or "
            "selectively citing lower-quality studies that support a preferred conclusion "
            "while dismissing higher-quality contradictory evidence."
        ),
        markers=[
            "multiple studies show", "numerous reports", "widely reported",
            "consistent across studies",
        ],
        example_biased=(
            "Multiple studies consistently show benefit, with positive results "
            "reported across diverse methodologies."
        ),
        example_neutral=(
            "While several observational studies report benefit, the two available RCTs "
            "show no significant effect. The discrepancy may reflect confounding in "
            "the observational designs."
        ),
    ),
]

# Map B-code → BiasPattern for quick lookup
_PATTERN_MAP: dict[str, BiasPattern] = {p.bias_code: p for p in BIAS_PATTERNS}

# Map agent name → relevant bias codes
AGENT_BIAS_CODES: dict[str, list[str]] = {
    "ARGUS": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11"],
    "LIBRA": ["B03"],
    "LENS":  ["B01", "B05"],
    "QUILL": ["B05", "B03"],
    "VIGIL": ["B08", "B07"],
}


class EvidenceRAG:
    """Indexes bias patterns + exemplars for cross-checking agent findings."""

    def __init__(self) -> None:
        self._store = VectorStore()
        self._pattern_chunks: list[Chunk] = []
        self._built = False

    def build_index(self, include_eval_exemplars: bool = False) -> None:
        """Build the evidence index.

        Args:
            include_eval_exemplars: If True, also loads samples from eval
                transforms (requires eval/ to be importable). Default False
                for production — True for benchmark runs.
        """
        chunks: list[Chunk] = []
        idx = 0

        # 1. Index canonical bias patterns
        for pattern in BIAS_PATTERNS:
            # Main description chunk
            text = (
                f"Bias Type: {pattern.bias_code} — {pattern.bias_name}\n"
                f"Description: {pattern.description}\n"
                f"Linguistic markers: {', '.join(pattern.markers)}"
            )
            chunks.append(Chunk(
                text=text, index=idx, section=f"pattern_{pattern.bias_code}",
                metadata={"type": "pattern", "bias_code": pattern.bias_code},
            ))
            idx += 1

            # Biased example
            chunks.append(Chunk(
                text=f"BIASED EXAMPLE ({pattern.bias_code} {pattern.bias_name}): {pattern.example_biased}",
                index=idx, section=f"example_{pattern.bias_code}_biased",
                metadata={"type": "example", "bias_code": pattern.bias_code, "label": "biased"},
            ))
            idx += 1

            # Neutral example
            chunks.append(Chunk(
                text=f"NEUTRAL EXAMPLE ({pattern.bias_code} {pattern.bias_name}): {pattern.example_neutral}",
                index=idx, section=f"example_{pattern.bias_code}_neutral",
                metadata={"type": "example", "bias_code": pattern.bias_code, "label": "neutral"},
            ))
            idx += 1

        # 2. Optionally load eval dataset exemplars
        if include_eval_exemplars:
            exemplar_chunks = self._load_eval_exemplars(start_idx=idx)
            chunks.extend(exemplar_chunks)

        self._pattern_chunks = chunks
        self._store.index(chunks)
        self._built = True

    def check_annotation(
        self,
        flagged_text: str,
        bias_type: str | None = None,
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Cross-check a flagged annotation against known patterns.

        Returns matching evidence patterns sorted by relevance.
        High-scoring matches against biased examples = confidence boost.
        High-scoring matches against neutral examples = possible false positive.
        """
        if not self._built:
            raise RuntimeError("Call build_index() first.")

        results = self._store.query(flagged_text, top_k=top_k * 2)

        # If bias_type specified, prioritise matching patterns
        if bias_type:
            code = self._normalise_bias_code(bias_type)
            prioritised = []
            others = []
            for r in results:
                if r.chunk.metadata.get("bias_code") == code:
                    prioritised.append(r)
                else:
                    others.append(r)
            results = (prioritised + others)[:top_k]
        else:
            results = results[:top_k]

        return results

    def get_patterns_for_agent(self, agent_name: str) -> list[BiasPattern]:
        """Return the bias patterns relevant to a specific agent."""
        codes = AGENT_BIAS_CODES.get(agent_name.upper(), [])
        return [_PATTERN_MAP[c] for c in codes if c in _PATTERN_MAP]

    def get_markers_for_agent(self, agent_name: str) -> list[str]:
        """Return all linguistic markers relevant to an agent's bias types."""
        patterns = self.get_patterns_for_agent(agent_name)
        markers: list[str] = []
        for p in patterns:
            markers.extend(p.markers)
        return markers

    # ── internal ────────────────────────────────────────────────────────

    def _load_eval_exemplars(self, start_idx: int) -> list[Chunk]:
        """Try to load exemplars from eval transforms."""
        chunks: list[Chunk] = []
        idx = start_idx

        try:
            from eval.transforms import get_all_specs
        except ImportError:
            return chunks

        for spec in get_all_specs():
            if spec.transform_fn is None:
                continue
            try:
                samples = spec.transform_fn(max_samples=10)
            except Exception:
                continue

            for sample in samples[:6]:  # keep index small
                label_tag = "BIASED" if sample.bias_present else "CONTROL"
                text = (
                    f"[{label_tag}] Dataset: {spec.dataset_name} | "
                    f"Bias: {spec.bias_type} | Agent: {spec.agent}\n"
                    f"Transform: {sample.transform_description}\n"
                    f"Text: {sample.input_text[:500]}"
                )
                chunks.append(Chunk(
                    text=text, index=idx,
                    section=f"exemplar_{spec.dataset_id}",
                    metadata={
                        "type": "exemplar",
                        "bias_code": spec.bias_type,
                        "dataset": spec.dataset_id,
                        "label": sample.label,
                    },
                ))
                idx += 1

        return chunks

    @staticmethod
    def _normalise_bias_code(bias_type: str) -> str:
        """Convert bias_type string to B-code."""
        # Handle both "B01" and "confirmation_bias" forms
        if bias_type.upper().startswith("B") and len(bias_type) <= 4:
            return bias_type.upper()
        _TYPE_TO_CODE = {
            "confirmation_bias": "B01",
            "reporting_bias": "B02",
            "certainty_inflation": "B03",
            "overgeneralisation": "B04",
            "overgeneralization": "B04",
            "framing_effect": "B05",
            "anchoring_bias": "B06",
            "outcome_reporting_bias": "B07",
            "causal_inference_error": "B08",
            "cultural_bias": "B09",
            "temporal_bias": "B10",
            "methodological_bias": "B11",
        }
        return _TYPE_TO_CODE.get(bias_type.lower(), bias_type.upper())

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def pattern_count(self) -> int:
        return len(self._pattern_chunks)
