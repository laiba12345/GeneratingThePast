import json
from collections import Counter
from pathlib import Path

json_path = Path("/Data/Laiba-Maddy-Project/Project2/Dataset/something_something_v2/labels/train.json")   # change if needed

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

templates = Counter(item["template"] for item in data)

for template, count in templates.most_common():
    print(f"{template}: {count}")

print(f"\nTotal samples: {len(data)}")
print(f"Total unique templates: {len(templates)}")