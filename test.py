import json
import os

files = [file for file in os.listdir() if file.endswith('json')]
print(files)

for file in files:
    with open(file) as f:
        content = json.load(f)['loss_mean']
    print(file)
    print(sum(content)/len(content))