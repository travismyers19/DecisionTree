import pandas as pd
from numpy import argmax
from statistics import mode

class DecisionTree:
    def __init__(self, datafile, max_tree_depth=5, min_node_records=5):
        self.datafile = datafile
        self.data = pd.read_csv(datafile, header=None).values.tolist()
        self.max_tree_depth = max_tree_depth
        self.min_node_recordes = min_node_records

    def split(self, attribute_index, value, data):
        left = []
        right = []

        for index in range(len(data)):
            if data[index][attribute_index] < value:
                left.append(data[index])
            else:
                right.append(data[index])

        return [left, right]

    def get_gini_index(self, groups, classes):
        gini = 0.
        total_instances = 0.

        for group in groups:
            if len(group) == 0:
                continue

            total_instances += len(group)
            class_vals = [row[-1] for row in group]
            score = 0.

            for class_val in classes:
                proportion = class_vals.count(class_val)/len(group)
                score += proportion*proportion

            gini += (1. - score)*len(group)

        return gini/total_instances


    def get_split(self, data):
        classes = list(set([row[-1] for row in data]))
        best_gini = None

        for attribute_index in range(len(data[0]) - 1):
            for row in data:
                groups = self.split(attribute_index, row[attribute_index], data)
                gini = self.get_gini_index(groups, classes)
                if best_gini is None or gini < best_gini:
                    best_index = attribute_index
                    best_value = row[attribute_index]
                    best_gini = gini
                    best_groups = groups

        return {'index': best_index, 'value': best_value, 'groups': best_groups}

    def get_terminal_class(self, group):
        return mode([row[-1] for row in group])

my_tree = DecisionTree('data_banknote_authentication.txt')
print(my_tree.data)