import pandas as pd

from classifier import KNearestNeighbor

data = pd.read_csv('glass.data.txt')


def normalize(key):
    max_val = data[key].max()
    data[key] = data[key].apply(lambda x: x / max_val)


normalize('ri')
normalize('na')
normalize('mg')
normalize('al')
normalize('si')
normalize('k')
normalize('ca')
normalize('ba')
normalize('fe')

values = data.values

classfier = KNearestNeighbor(data)

test = {'ri': 1.51747, 'na': 12.84, 'mg': 3.50, 'al': 1.14, 'si': 73.27, 'k': 0.56, "ca": 8.55, 'ba': 0.00, 'fe': 0.00}

correct = 0
incorrect = 0

for index, row in data.iterrows():
    result = classfier.predict(row.to_dict(), 5)
    if result == int(row['class']):
        correct += 1
    else:
        incorrect += 1

print(correct,incorrect)
