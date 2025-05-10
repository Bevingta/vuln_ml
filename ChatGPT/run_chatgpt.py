import json
from openai import OpenAI

first_half = "sk-proj-c0_Wi9MdPbgh7ACk2o1xLpKPmMiQUNJmbWXACCPcBA3lnLFDxUgT7YxrvmhtJpFS9LNfqlvj-JT3BlbkFJoFQfXeOySpFM"
second_half = "-g1jv3Dz1gh8j5i_cjASBcgjjAgpcbih4owTW78AXUEPnKxE2WKfdszFbbkVQA"

api_key = first_half + second_half


client = OpenAI(
  api_key=api_key
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


path = "primevul_for_gpt.json"

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

import json

with open("C:/Users/saram/OneDrive/Desktop/ChatGPT/new_gpt_evaluations.txt") as f1:
    first_data = json.load(f1)

with open("C:/Users/saram/OneDrive/Desktop/ChatGPT/primevul_for_gpt.json") as f2:
    second_data = json.load(f2)

counter = 0
for item1, item2 in zip(first_data, second_data):
    score = item1["cvss_score"]

    if score == "0.01":
        counter += 1
    else:
        if int(score) != int(item2["target"]):
            counter += 1

print("Number of elements with gpt evaluation different from the real target:", counter)
print("Performance:", 1 - counter/7204)
