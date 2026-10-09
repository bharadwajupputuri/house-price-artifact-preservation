
import json
import sys

MINIMUM_R2 = 0.70

print("HOUSE PRICE ML QUALITY GATE")
print("---------------------------")

try:
    with open("metrics.json", "r") as file:
        metrics = json.load(file)

    r2_score = metrics["r2_score"]

except (FileNotFoundError, KeyError, json.JSONDecodeError) as error:
    print("ERROR: Could not read model metrics.")
    print(error)
    sys.exit(1)

print("Model R2 Score:", round(r2_score, 4))
print("Minimum Required R2 Score:", MINIMUM_R2)

if r2_score >= MINIMUM_R2:
    print("QUALITY GATE PASSED")
    print("Model meets the minimum performance requirement.")
    sys.exit(0)
else:
    print("QUALITY GATE FAILED")
    print("Model performance is below the required threshold.")
    sys.exit(1)
