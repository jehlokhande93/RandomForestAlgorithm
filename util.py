from scipy import stats
import numpy as np


# This method computes entropy for information gain
def entropy(class_y):
    # Input:
    #   class_y         : list of class labels (0's and 1's)

    # TODO: Compute the entropy for a list of classes
    #
    # Example:
    #    entropy([0,0,0,1,1,1,1,1,1]) = 0.92

    entropy = 0

    length = len(class_y)
    ones = sum(class_y)
    zeros = length-ones

    pones = float(ones)/float(length)
    pzeros = float(zeros)/float(length)

    if pones == 0 or pzeros == 0:
        entropy = 0
    else:
        entropy = -pones*np.log2(pones)-pzeros*np.log2(pzeros)


    return entropy


def partition_classes(X, y, split_attribute, split_val):

    X_left = np.asarray([], dtype=object)
    X_right = np.asarray([], dtype=object)

    y_left = np.asarray([], dtype=object)
    y_right = np.asarray([], dtype=object)

    if type(split_val) not in (int, float, complex, long):
        X_left = X[X[:,split_attribute]==split_val,:-1]
        X_right = X[X[:,split_attribute]!=split_val,:-1]
        y_left = X[X[:,split_attribute]==split_val,-1]
        y_right = X[X[:,split_attribute]!=split_val,-1]

    else:
        X_left =X[X[:,split_attribute]<=split_val,:-1]
        X_right = X[X[:,split_attribute]>split_val,:-1]
        y_left = X[X[:,split_attribute]<=split_val,-1]
        y_right = X[X[:,split_attribute]>split_val,-1]

    #print(list(X_left), list(X_right))
    return (np.ndarray.tolist(X_left), np.ndarray.tolist(X_right), np.ndarray.tolist(y_left), np.ndarray.tolist(y_right))


def information_gain(previous_y, current_y):
    # Inputs:
    #   previous_y: the distribution of original labels (0's and 1's)
    #   current_y:  the distribution of labels after splitting based on a particular
    #               split attribute and split value

    # TODO: Compute and return the information gain from partitioning the previous_y labels
    # into the current_y labels.
    # You will need to use the entropy function above to compute information gain
    # Reference: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf

    """
    Example:

    previous_y = [0,0,0,1,1,1]
    current_y = [[0,0], [1,1,1,0]]

    info_gain = 0.45915
    """
    if len(current_y[0])==0 or len(current_y[1])==0:
        return 0

    else:
        entropymain = entropy(previous_y)

        entropybranch0 = entropy(current_y[0])
        entropybranch1 = entropy(current_y[1])

        p0 = float(len(current_y[0]))/float((len(current_y[0])+len(current_y[1])))
        p1 = float(len(current_y[1]))/float((len(current_y[0])+len(current_y[1])))


        info_gain = entropymain - p0*entropybranch0 - p1*entropybranch1

        return info_gain


