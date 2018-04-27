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

        info_list.sort(key=lambda x: x[0], reverse=True)

        # print(info_list[0])
        # print(info_list[len(info_list) - 1])

        part = info_list[:k]

        result = [0] * 8

        for item in part:
            result[item[1]] = result[item[1]] + 1

        max_value = result[0]
        max_index = 0

        for index, item in enumerate(result):
            if item > max_value:
                max_value = item
                max_index = index

        # print(max_index)
        return max_index
