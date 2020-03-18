import pandas as pd
from numpy import argmax
from statistics import mode

class DecisionTree:
    def __init__(self, datafile, max_tree_depth=5, min_node_records=5):
        self.datafile = datafile
        self.data = pd.read_csv(datafile, header=None).values.tolist()
        self.max_tree_depth = max_tree_depth
        self.min_node_records = min_node_records
        self.grow_tree()

    def split(self, attribute_index, value, data):
        left = []
        right = []

        for index in range(len(data)):
            if data[index][attribute_index] < value:
                left.append(data[index])
            else:
                right.append(data[index])

        return [left, right]

    def recursive_split(self, node, depth):
        left = node['groups'][0]
        right = node['groups'][1]
        del(node['groups'])

        if not left or not right:
            node['left'] = self.get_terminal_class(left + right)
            node['right'] = node['left']
            return

        if depth >= self.max_tree_depth:
            node['left'] = self.get_terminal_class(left)
            node['right'] = self.get_terminal_class(right)
            return

        if len(left) <= self.min_node_records:
            node['left'] = self.get_terminal_class(left)
        else:
            node['left'] = self.get_split(left)
            self.recursive_split(node['left'], depth + 1)

        if len(right) <= self.min_node_records:
            node['right'] = self.get_terminal_class(right)
        else:
            node['right'] = self.get_split(right)
            self.recursive_split(node['right'], depth + 1)

    def grow_tree(self):
        self.root = self.get_split(self.data)
        self.recursive_split(self.root, 0)

    def predict(self, row, node=None):
        if node is None:
            node = self.root
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(row, node['left'])
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(row, node['right'])
            else:
                return node['right']

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
        row_classes = [row[-1] for row in group]
        classes = {}
        for label in row_classes:
            if label in classes:
                classes[label] += 1
            else:
                classes[label]  = 1
        highest = 0
        best = None
        for label in classes:
            if classes[label] > highest:
                best = label
                highest = classes[label]

        return best

    #def print_tree(self,):


my_tree = DecisionTree('data_banknote_authentication.txt',max_tree_depth=0)
for i in range(100):
    print(my_tree.predict(my_tree.data[i]))
    print(my_tree.data[i][-1])
    print('')