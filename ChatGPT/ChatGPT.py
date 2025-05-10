import json
from openai import OpenAI


client = OpenAI(
  api_key=""
)


def get_cvss_score_from_func(func_code):
    prompt = (
        "You are a Senior Software Engineer scanning a codebase for vulnerabilities. Here is a snippet of C/C++ code. Please analyze this code for vulnerabilities, and return 1 if the code contains a vulnerability, or 0 if it is safe. Respond with nothing other than the classification of 0 or 1. Do not provide reasoning. "f"Code: {func_code}"
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        score = completion.choices[0].message.content.strip()
        return score
    except Exception as e:
        print(f"Error processing func: {e}")
        return "0.01"


path = "C:/Users/saram/OneDrive/Desktop/primevul_for_gpt.json"

with open(path, 'r', encoding='utf-8') as file:
    data = json.load(file)

new_data = []


for i, entry in enumerate(data):
    print(f"Processing entry {i} in the dataset...")
    idx = entry["idx"]
    func = entry["func"]
    score = get_cvss_score_from_func(func)
    new_data.append({"idx": idx, "func": func, "cvss_score": score})


output_file = f"new_gpt_evaluations"
try:
    with open(output_file, 'r', encoding='utf-8') as file:
        existing_data = json.load(file)
except FileNotFoundError:
    existing_data = []

existing_data.extend(new_data)

with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(existing_data, file, indent=4)

print(f"Saved data entries to '{output_file}'")