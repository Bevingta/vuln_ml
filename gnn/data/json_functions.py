import json

class JsonFuncs:
    def load_json_array(path):
        try:
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except FileNotFoundError:
            print("DB not found")
            data = []
        return data
    def save_array_to_json(data, path):
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)