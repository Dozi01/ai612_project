from scoring.scorer import Scorer
import json

with open("data/valid_data.json", "r") as f:
    data = json.load(f)

with open("data/valid_label.json", "r") as f:
    gold_labels = json.load(f)

with open("results/result_20250310_192408.json", "r") as f:
    predictions = json.load(f)

print(predictions)
scorer = Scorer(
    data=data,
    predictions=predictions,
    gold_labels=gold_labels,
    score_dir="results"
)

score = scorer.get_scores()

print(score)
