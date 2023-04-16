import json

pre = json.load(open('pre_relations.json', 'r'))
desc2name = {v : k for k, v in pre.items()}
rel = json.load(open('relations.json', 'r'))

path = 'valid.txt.json'

x = json.load(open(path, 'r'))
final = []
for i, temp in enumerate(x):
    name = desc2name[temp['relation']]
    temp['relation'] = rel[name]
    final.append(temp)
json.dump(final, open(path, 'w'))