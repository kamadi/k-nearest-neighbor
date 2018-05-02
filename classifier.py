import pandas as pd
import math


class KNearestNeighbor:

    def __init__(self, data):
        """

        :type data: pandas
        """
        self.data = data

    # id,ri,na,mg,al,si,k,ca,ba,fe,class

    def predict(self, arr, k):
        info_list = []

        for index, row in self.data.iterrows():
            distance = math.sqrt(((arr['ri'] - row['ri']) ** 2) + ((arr['na'] - row['na']) ** 2)
                                 + ((arr['mg'] - row['mg']) ** 2) + ((arr['al'] - row['al']) ** 2)
                                 + ((arr['si'] - row['si']) ** 2) + ((arr['k'] - row['k']) ** 2)
                                 + ((arr['ca'] - row['ca']) ** 2) + ((arr['ba'] - row['ba']) ** 2)
                                 + ((arr['fe'] - row['fe']) ** 2))
            info_list.append([distance, int(row['class'])])

        info_list.sort(key=lambda x: x[0])

        part = info_list[:k]

        result = []

        for i in range(0, 8):
            result.append([i, 0])

        for item in part:
            result[item[1]][1] = result[item[1]][1] + 1

        result.sort(key=lambda x: x[1], reverse=True)

        if k > 1 and result[0][1] == result[1][1]:
            for item in part:
                if item[1] == result[0][1] or item[1] == result[1][1]:
                    return item[1]
        return result[0][0]
