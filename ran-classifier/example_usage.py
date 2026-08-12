#!/usr/bin/env python3
"""
example_usage.py — AIRiskClassifier demonstrations

Quick start (static taxonomy):
    python example_usage.py

With IBM Risk Atlas (requires ai-atlas-nexus):
    pip install ai-atlas-nexus
    python example_usage.py --nexus
"""

import sys
sys.path.insert(0, 'classifier')

from ai_risk_classifier import AIRiskClassifier
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nexus', action='store_true', help='Use IBM Risk Atlas from ai-atlas-nexus')
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Example 1: IBM Risk Atlas (from nexus or custom)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 80)
if args.nexus:
    print("EXAMPLE 1: IBM Risk Atlas from ai-atlas-nexus (~99 risks)")
    print("=" * 80)
    clf = AIRiskClassifier.from_nexus(taxonomy='ibm-risk-atlas')
else:
    print("EXAMPLE 1: IBM Risk Atlas from ai-atlas-nexus (~99 risks)")
    print("(Note: Custom static taxonomy has been removed)")
    print("=" * 80)
    print("\nTo use this demo, run with --nexus flag or provide a custom taxonomy.")
    print("Example: python example_usage.py --nexus")
    sys.exit(0)

print(f"\n{clf}\n")

usecases = [
    "An LLM answers patient questions about medications.",
    "A hiring system screens job applications.",
]

for uc in usecases:
    results = clf.identify_risks_from_usecases([uc])
    print(results[0])


# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Custom thresholds
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("EXAMPLE 2: Custom Thresholds (More Sensitive)")
print("=" * 80)

if args.nexus:
    clf_sensitive = AIRiskClassifier.from_nexus(
        taxonomy='ibm-risk-atlas',
        threshold_high=0.15,
        threshold_medium=0.06,
        alpha=0.4
    )
else:
    clf_sensitive = AIRiskClassifier(
        threshold_high=0.15,
        threshold_medium=0.06,
        alpha=0.4
    )

usecase = "A chatbot helps users invest in stocks."
results = clf_sensitive.identify_risks_from_usecases([usecase])
print(f"\n{results[0]}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Raw score matrix for downstream analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("EXAMPLE 3: Score Matrix (for ML pipelines)")
print("=" * 80)

usecases_batch = [
    "An automated hiring system screens job applications and ranks candidates.",
    "A generative AI assistant creates marketing content.",
    "A medical AI diagnoses diseases from imaging.",
]

score_matrix = clf.score(usecases_batch)
print(f"\nScore matrix shape: {score_matrix.shape}")
print(f"  Rows: {score_matrix.shape[0]} use cases")
print(f"  Cols: {score_matrix.shape[1]} risks")
print(f"\nRaw scores (first 3 risks across 3 use cases):")
print(score_matrix[:, :3])


# ─────────────────────────────────────────────────────────────────────────────
# Example 4: Explore other taxonomies (nexus only)
# ─────────────────────────────────────────────────────────────────────────────

if args.nexus:
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Other Available Taxonomies")
    print("=" * 80)

    taxonomies = [
        ('nist-ai-rmf', 'NIST AI RMF'),
        ('mit-ai-risk-repository', 'MIT Risk Repository'),
        ('owasp-llm-2.0', 'OWASP Top 10 for LLMs'),
    ]

    for tax_name, tax_label in taxonomies:
        try:
            clf_tax = AIRiskClassifier.from_nexus(taxonomy=tax_name)
            print(f"  ✓ {tax_label:<30} {len(clf_tax.taxonomy)} risks")
        except Exception as e:
            print(f"  ✗ {tax_label:<30} (error: {str(e)[:40]})")

print("\n" + "=" * 80)
