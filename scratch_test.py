import json
with open('data/weather_data.json') as f:
    data = json.load(f)
    print(json.dumps(data[-1], indent=2))
