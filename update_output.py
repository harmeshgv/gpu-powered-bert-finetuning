import json
from tabulate import tabulate
import re

with open("outputs.json", 'r', encoding="utf-8") as f:
    data = json.load(f)


rows = [
    [
        accuracy.get("stage", ""),
        accuracy.get("description", ""),
        accuracy.get("model_used", ""),
        f"{accuracy.get("accuracy", "")}%"
    ]
    for accuracy in data.get("fine-tune", [])
]

headers = ["Stage", "Description", "Model Used", "Accuracy"]

table_md = tabulate(rows, headers = headers, tablefmt="github")
with open("README.md", 'r', encoding="utf-8") as f:
    readme_content = f.read()

start_placeholder = '<!--start-->'
stop_placeholder = '<!--stop-->'

new_readme_content = re.sub(
    f"({re.escape(start_placeholder)})[\\s\\S]*({re.escape(stop_placeholder)})",
    f"\\1\n{table_md}\n\\2",
    readme_content
)
   
with open("README.md", 'w', encoding='utf-8') as f:
    f.write(new_readme_content)

print("README file updated successfully with the description and scores.")
   