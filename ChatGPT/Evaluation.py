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
