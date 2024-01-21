import json

r = json.load(open("results.json"))
for time, res in r.items():
    print(time)
    for features, scores in res.items():
        if "classifier" in features:
            continue
        print(f"\t- {features}:")
        for metric, score in scores['test'].items():
            print(f"\t\t- {metric}:")
            print(f"\t\t\t{score}")
