# AI Risk Classifier — Atlas Nexus Demo

A lightweight, interpretable multi-label AI risk classifier that identifies applicable risks from AI use-case descriptions.

## Quick Start

### Installation

```bash
pip install scikit-learn numpy pandas
```

### Basic Usage

```python
from classifier.ai_risk_classifier import AIRiskClassifier

# Initialize classifier
clf = AIRiskClassifier()

# Identify risks for use cases
results = clf.identify_risks_from_usecases([
    "An LLM answers patient questions about medications."
])

# Print results
for result in results:
    print(result)
```

### Output Example

```
Use case: An LLM answers patient questions about medications.
Identified 5 risk(s):
  [HIGH  ] Hallucination / Confabulation             score=0.578  ███████████████████████████████████ kw=[medical, health, patient, medication]
  [HIGH  ] Harmful Output                             score=0.401  ██████████████████████ kw=[medical, patient, medication, safety]
  [MEDIUM] Privacy Violation / Data Leakage           score=0.168  █████████ kw=[patient, health, medical, personal]
  [MEDIUM] Regulatory Non-Compliance                  score=0.157  █████████ kw=[medical, healthcare, patient, health]
  [MEDIUM] Lack of Explainability / Transparency     score=0.143  ████████
```

## Interactive Demo

Run the comprehensive Jupyter notebook for hands-on examples:

```bash
jupyter notebook AI_Risk_Classifier_Demo.ipynb
```

The notebook includes:
- Taxonomy overview
- Single use-case analysis
- Batch processing multiple use cases
- Score matrix generation for downstream ML
- Custom threshold tuning
- Keyword matching deep dive
- Results export

## Architecture

### Hybrid Scoring

The classifier combines two complementary signals:

1. **TF-IDF Cosine Similarity** — Surface-level vocabulary overlap
   - Use-case is vectorized and compared against per-risk anchor sentences
   - Score = max cosine similarity across a risk's anchors (configurable)

2. **Domain Keyword Matching** — Semantic coverage for domain terminology
   - Use-case is tokenized into unigrams, bigrams, and light-stemmed forms
   - Intersection with curated per-risk keyword vocabulary
   - Score = hits / normalizer (capped at 1.0)

**Final Score**: `alpha * tfidf_score + (1 - alpha) * keyword_score`

### Risk Taxonomy

16 curated risks across 7 categories:

| Category | Risks |
|----------|-------|
| **Reliability** | Hallucination, Overreliance |
| **Safety** | Harmful Output, Toxic Content, Misinformation, Social Engineering |
| **Fairness** | Output Bias / Discrimination |
| **Security** | Jailbreaking / Prompt Injection, Data/Model Poisoning |
| **Privacy** | Privacy Violation / Data Leakage |
| **Transparency** | Lack of Explainability |
| **Legal** | Copyright / IP Violation, Regulatory Non-Compliance |
| **Governance** | Third-Party / Supply Chain Risk, Lack of Accountability, Environmental Impact |

**Sources**: IBM AI Risk Atlas, NIST AI RMF, MIT Risk Repository, OWASP Top 10 for LLMs

## API Reference

### `AIRiskClassifier`

```python
clf = AIRiskClassifier(
    threshold_high=0.30,       # Score ≥ this → "high" relevance
    threshold_medium=0.12,     # Score ≥ this → "medium" relevance
    alpha=0.35,                # Weight of TF-IDF (0–1)
    taxonomy=None,             # Custom risk taxonomy (None = built-in 16 risks)
    aggregation="max"          # "max" or "mean" for per-anchor TF-IDF aggregation
)
```

#### Methods

**`identify_risks_from_usecases(usecases: List[str]) → List[UsecaseRiskResult]`**

Identify applicable risks for one or more use-case descriptions.

Returns a list of `UsecaseRiskResult` objects with properties:
- `.risks` — all identified risks (sorted by score, descending)
- `.high` — high-relevance risks only
- `.medium` — medium-relevance risks only
- `.low` — low-relevance risks only (empty if threshold_medium is set)
- `.to_dict()` — convert to JSON-serializable dict

**`score(usecases: List[str]) → np.ndarray`**

Return raw score matrix (n_usecases × n_risks). Suitable for sklearn pipelines and downstream analysis.

**`risk_ids() → List[str]`**

Return list of risk IDs in taxonomy order.

### Data Classes

**`IdentifiedRisk`**
- `id` — Risk identifier
- `label` — Human-readable risk label
- `group` — Risk category
- `sources` — Authoritative sources (IBM, NIST, OWASP, MIT)
- `score` — Final hybrid score (0–1)
- `tfidf_score` — TF-IDF component (0–1)
- `keyword_score` — Keyword component (0–1)
- `relevance` — "high" or "medium"
- `matched_keywords` — Keywords that contributed to the score

**`UsecaseRiskResult`**
- `usecase` — Original use-case string
- `risks` — List of `IdentifiedRisk` objects

## Configuration

### Threshold Tuning

Adjust thresholds to match your risk tolerance:

```python
# Conservative: fewer false positives
conservative = AIRiskClassifier(threshold_high=0.40, threshold_medium=0.20)

# Aggressive: catch marginal risks
aggressive = AIRiskClassifier(threshold_high=0.20, threshold_medium=0.08)
```

### TF-IDF / Keyword Weighting

Control the balance between vocabulary similarity and domain terminology:

```python
# More keyword-driven (good for domains with standard terminology)
keyword_heavy = AIRiskClassifier(alpha=0.25)

# More TF-IDF (good for novel use-case descriptions)
tfidf_heavy = AIRiskClassifier(alpha=0.50)
```

### Custom Taxonomy

Use domain-specific risks instead of the built-in set:

```python
custom_taxonomy = [
    {
        "id": "my-risk",
        "label": "My Custom Risk",
        "group": "custom",
        "sources": ["Internal Policy"],
        "anchors": ["description of when this risk applies"],
        "keywords": ["key", "domain", "terms"],
        "kw_norm": 3,
    },
    # ... more risks
]

clf = AIRiskClassifier(taxonomy=custom_taxonomy)
```

## Use Cases

### 1. AI Audit Workflow
Screen proposed AI applications for risk surface before development:
```python
proposed_usecases = ["...", "...", "..."]
results = clf.identify_risks_from_usecases(proposed_usecases)
high_risk = [r for r in results if r.high]
```

### 2. Compliance Checks
Automatically flag applications in regulated domains:
```python
# Identify healthcare use cases with regulatory risk
for result in results:
    if any(r.id == "regulatory-non-compliance" and r.relevance == "high" 
           for r in result.risks):
        print(f"⚠️  Regulatory review needed: {result.usecase}")
```

### 3. ML Pipeline Integration
Use scores as features in broader governance systems:
```python
score_matrix = clf.score(usecases)  # (n_usecases, 16)
# Feed into downstream risk models, dashboards, alerting, etc.
```

### 4. Threshold Calibration
If you have labelled data, optimize thresholds for your deployment:
```python
from sklearn.model_selection import cross_val_score

# Test threshold combinations
for threshold in np.arange(0.1, 0.5, 0.05):
    clf_tuned = AIRiskClassifier(threshold_high=threshold)
    scores = cross_val_score(clf_tuned, X, y, cv=5)
    print(f"Threshold {threshold}: {scores.mean():.3f}")
```

## Limitations

- **No training data required**, but thresholds are calibrated to general AI risk landscapes
- **Anchor sentences and keywords are static** — cannot adapt to novel risk formulations
- **Vocabulary overlap bias** — may miss risks that use different terminology
- **No temporal reasoning** — cannot reason about sequential or causal relationships in use cases
- **Assumes English text** — TF-IDF vectorizer optimized for English

## References

- [IBM AI Risk Atlas](https://www.ibm.com/ai-risk-atlas)
- [NIST AI Risk Management Framework](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)
- [OWASP Top 10 for Large Language Models](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [MIT AI Risk Repository](https://air.mit.edu/)
- [IBM Granite Guardian](https://research.ibm.com/blog/granite-guardian)

## License

Same as parent repository (see `LICENSE` in root).

---

**Part of**: [AI Atlas Nexus Demos](https://github.com/IBM/risk-atlas-nexus-demos)
