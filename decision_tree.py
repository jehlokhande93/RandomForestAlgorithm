from util import entropy, information_gain, partition_classes
import numpy as np
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
        pass


    def branch(self, X, y):
        splitlist = list()
        ig = 0
        xleft=[]
        xright=[]
        split_attr='0'
        split_val ='0'

        X=np.asarray(X,dtype=object)
        y=np.asarray([y], dtype= object)
        X = np.concatenate((X,y.T),axis=1)
        m,n = np.shape(X)
        for column in range(n-1): #Enumerate through each attribute in X
            #print('yo',X[0,column])
            if type((X[0,column])) not in (int, long, float, complex) and len(set(X[:,column]))>1: #If X is categorical, enumerate through each row in X[column]
                #print('entered alpha')
                #print(len(set([x[0] for x in X])))
                for row in set(X[:,column]):
                    #print(row)
                    X_left, X_right, y_left, y_right = partition_classes(X, y, column, row)
                    #print(y,y_left, y_right)
                    #print(ig,information_gain(np.ndarray.tolist(y[0]), [y_left,y_right]))
                    #print('ig calc')
                    #print(information_gain(np.ndarray.tolist(y[0]), [y_left,y_right]))
                    if information_gain(np.ndarray.tolist(y[0]), [y_left,y_right]) > ig:
                        #print ('alpha',ig,information_gain(np.ndarray.tolist(y[0]), [y_left,y_right]))
                        ig = information_gain(np.ndarray.tolist(y[0]), [y_left,y_right])
                        xleft = X_left
                        xright = X_right
                        yleft = y_left
                        yright=y_right
                        split_attr = column
                        split_val = str(row)

            elif type((X[0,column])) in (int, long, float, complex) and len(set(X[:,column]))>1:
                #mean = sum([x[column] for x in X])/len([x[column] for x in X])
                for value in [np.percentile(X[:,column],25),np.percentile(X[:,column],50),np.percentile(X[:,column],75), np.mean(X[:,column])]:
                    #np.percentile(X[:,column],25),np.percentile(X[:,column],50),np.percentile(X[:,column],75),
                    X_left, X_right, y_left, y_right = partition_classes(X, y, column, value)
                    #print(value,np.ndarray.tolist(y[0]), [y_left,y_right])
                    if information_gain(np.ndarray.tolist(y[0]), [y_left,y_right]) > ig:
                        #print ('num',ig,information_gain(np.ndarray.tolist(y[0]), [y_left,y_right]))
                        ig = information_gain(np.ndarray.tolist(y[0]), [y_left,y_right])
                        #print('newIG', ig)
                        xleft = X_left
                        xright = X_right
                        yleft = y_left
                        yright=y_right
                        split_attr = column
                        split_val = value



        if ig==0:
            y1=np.ndarray.tolist(y[0])
            return max(y1, key=y1.count)

        else:
            #print xleft, X_left
            return {'left':self.branch(xleft,yleft), 'right':self.branch(xright,yright), 'splitattr':split_attr, 'splitval':split_val}

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree

        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        self.tree=self.branch(X,y)

    def preclassify(self, selftree, record):
        if type(selftree) is dict:
            if type(selftree['splitval']) not in (int, long, float, complex):
                if record[selftree['splitattr']]==selftree['splitval']:
                    return self.preclassify(selftree['left'], record)
                else:
                    return self.preclassify(selftree['right'], record)
            else:
                if record[selftree['splitattr']]<=float(selftree['splitval']):
                    return self.preclassify(selftree['left'], record)
                else:
                    return self.preclassify(selftree['right'], record)

        else:
            returnvalue = selftree
        return returnvalue

    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        selftree=self.tree
        return self.preclassify(selftree, record)

# X= [[3, 'aa', 10], [1, 'bb', 22], [2, 'cc', 28], [5, 'bb', 32], [4, 'cc', 32]]
# y = [1, 1, 0, 0, 1]
