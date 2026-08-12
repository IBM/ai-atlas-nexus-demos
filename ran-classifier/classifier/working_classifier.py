from ai_risk_classifier import AIRiskClassifier

clf = AIRiskClassifier()
results = clf.identify_risks_from_usecases(
    usecases=["An LLM answers customer questions about product recalls."]
)
for r in results:
    print(r)

usecases = ["an automated hiring system", "an llm assisted probation decider", "a medical system for cats"]

results = clf.identify_risks_from_usecases(usecases=usecases)
print(results)

# For sklearn pipelines — returns (n_usecases, n_risks) score matrix
scores = clf.score(usecases)
print(scores)