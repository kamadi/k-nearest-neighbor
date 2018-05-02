import pandas as pd

from classifier import KNearestNeighbor

data = pd.read_csv('glass.csv')


def normalize(key):
    max_val = data[key].max()
    min_val = data[key].min()
    data[key] = data[key].apply(lambda x: (x - min_val) / (max_val - min_val))


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

train = data[:150]

test = data[150:]

classfier = KNearestNeighbor(train)

for i in range(1, 11):
    correct = 0

    incorrect = 0

    for index, row in test.iterrows():
        result = classfier.predict(row.to_dict(), i)
        if result == int(row['class']):
            correct += 1
        else:
            incorrect += 1
    accuracy = correct / (correct + incorrect)
    print(i, accuracy)
