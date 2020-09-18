import numpy as np

class DecisionTreeClassifier:
    #: INFORMATION_GAIN, GINI constants (criterion) :
    (ENTROPY, GINI) = list(range(2))

    # attributes informations
    INDEX = 0
    
    def __init__(self, max_depth= float('+inf'), data = None, attributes_list= None, criterion = None, root = None): # data -> (X, Y), classes -> list of the classes
        if root != None:
            self.__root = root
            return 
        self.__root = None
        self.__X = data[0] # X: NxD
        self.__Y = data[1] # Y: Nx1
        self.__attributes_list = attributes_list # list of tuples like (index)
        self.__classes = np.unique(data[1])
        self.__criterion = criterion
        self.__max_depth = max_depth 
            
    def train(self): #It starts the algorithm

        if self.__root != None:
            print("self.__root")
            return 
        
        if self.__criterion == self.ENTROPY:
            self.__root = Entropy(self.__X, self.__Y, self.__attributes_list, self.__classes, self.__max_depth -1)

        else:
            self.__root = Gini(self.__X, self.__Y, self.__attributes_list, self.__classes, self.__max_depth -1 )


    def prediction(self, Xte, dtype = float): # predict the label of Xte
        order = [dtype(i) for i in range(Xte.shape[0])] #dtype can have effect the approximation
        order = np.reshape(order, (Xte.shape[0], 1))
        Xte = np.append(Xte, order, axis=1)
        ordIdx, Ypred = self.__root.test(Xte)
        Ypred = np.array(Ypred)
        idx = np.argsort(ordIdx)
        Ypred = Ypred[idx]
        return Xte, Ypred
    
    def test(self, Xte, Yte, dtype = float): 
        Xte_p, Ypred = self.prediction(Xte, dtype)
        return Xte_p, Ypred, self.__calcError(Ypred, Yte)
    
    def __calcError(self, Ypred, Yte): #percentage of misclassified tuples
        misclassified = list(Ypred==Yte).count(False)
        return misclassified/Ypred.shape[0] if Ypred.shape[0]>0 else 0.0 # if there are no tuples in the test set it returns 0.0
        
    def pruning(self, Xte, Yte):
        if Xte.shape[0] <= 0:
            return self.__root
        self.__root.pruning(Xte, Yte)
        return self.__root
    
    def getMaxDepth(self):
        return self.__root.getMaxDepth()
    
    def getBalance(self): # the abs balance of the tree
        return np.abs(self.__root.getBalance())
    
    def getRoot(self):
        return self.__root

class LeafRequiredException(Exception):
    pass

class Split: # define and execute the split operations
    def __init__(self, index):
        self.__index = index
        

    def getIndex(self):
        return self.__index
    
    def __str__(self):
        return "{0}".format(self.__index)
       
class Continuous(Split):
    
    def __init__(self, index, threshold):
        Split.__init__(self, index)
        self.__threshold = threshold
        
    def __str__(self):
        return "{0}".format(self._Split__index) + ": {}".format(self.__threshold)
    
    def __partitioningX(self, X, minorization = True):
        if minorization:
            return X[X[:, self._Split__index] <= self.__threshold, :]
        return X[X[:, self._Split__index] > self.__threshold, :]

    def partitioningY(self, data):
        X, Y = data
        Yp0 = Y[np.where(X[:, self._Split__index]<= self.__threshold)]
        Yp1 = Y[np.where(X[:, self._Split__index] > self.__threshold)]
        return [Yp0,Yp1]

    def partitioningTest(self, X):
    
        left = self.__partitioningX(X)
        right = self.__partitioningX(X, False)
        return left, right
    
    def partitioningTrain(self, data, train = True ): #data -> (X, Y)
        Yp0, Yp1 = self.partitioningY(data)
        X, Y = data
        Xp0 = self.__partitioningX(X)
        Xp1 = self.__partitioningX(X, False)

        if train and (np.size(Xp0) == 0 or np.size(Xp1) == 0):
            raise LeafRequiredException('Leaf required') # a partition is empty
                
        right = [Xp1, Yp1]
        left = [Xp0, Yp0]
        return left, right

class Node:
    def __init__(self, X, Y, attributes, classes, max_depth, measure):
        self.__leaf = False
        self.__X = X
        self.__Y = Y
        self.__n = X.shape[0]
        self.__d = X.shape[1]
        self.__attributes = attributes
        self.__classes = classes
        self.__classDistribution = None
        self.__max_depth = max_depth
        self.__measure = measure

    def isLeaf(self):
        return self.__leaf
    
    def getBalance(self): #get the depths of the children
        left = self.__left.getMaxDepth()
        right = self.__right.getMaxDepth()
        return  left - right
    
    def getMaxDepth(self):
        if self.isLeaf():
            return 0
        depths = [self.__left.getMaxDepth() +1, self.__right.getMaxDepth() +1]
        return np.amax(depths)

    def __mostCommonClass(self, Y): # it returns the most common class and the class distribution of the node
        labels, counts= np.unique(Y, return_counts=True)
        i = np.argmax(counts)
        distribution = { label : 0.0 for label in self.__classes}
        for idx, label in enumerate(labels):
            distribution[label] = counts[idx]/Y.shape[0]
        return labels[i], distribution

    def __measureAttribute(self, outcomes): #expected information required to classify a tuple from X based on the partitioning by A
        return np.sum([ (np.size(outcome)/self.__n) * self.__measure(outcome) for outcome in outcomes])

    def __continuousAttribute(self, attribute): # if the set of the attribute is continuous
        index = attribute[DecisionTreeClassifier.INDEX]
        sequence = np.sort(self.__X[:,index]) #increasing order with no repeats
        nSplit = sequence.size -1 
        split_list = np.zeros(nSplit)
        measure_split = np.zeros(nSplit)
        
        for i in range(nSplit):
            split_list[i] = (sequence[i] + sequence[i+1])/2
            left, right = Continuous(index, split_list[i]).partitioningY((self.__X, self.__Y))
            measure_split[i] = self.__measureAttribute([left, right])
        i = np.argmin(measure_split)
        return measure_split[i], split_list[i]

    
    def __measureGain(self):
        measureA = []
        splits = [] # for eventual split involved
        for attribute in self.__attributes:
            measure, split = self.__continuousAttribute( attribute)
            measureA = np.append(measureA, measure)
            splits = np.append(splits, split)
        measureY = self.__measure(self.__Y)
        measureGain = measureY - measureA # the expected reduction in the information requirement caused by knowing the value of A
        return measureGain, splits
    
    def __bestAttribute(self): # it finds the best attribute 
        gainA, splits = self.__measureGain()
        i = np.argmax(gainA)
        split = Continuous(i, splits[i])
        return split.__str__(), gainA[i], split

    def test(self, Xte):
        if self.isLeaf():
            Ypred = [self.__label for i in range(Xte.shape[0])]
            ordIdx = Xte.shape[1]-1
            return list(Xte[:, ordIdx]), Ypred
        
        left, right = self.__split.partitioningTest(Xte)
        ordLeft, Yleft = (self.__left).test(left)  # recursive calls left node
        ordRight, Yright = (self.__right).test(right)  # recursive calls right node
        ordX = ordLeft + ordRight
        Ypred = Yleft + Yright
        return ordX, Ypred
    
    def pruning(self, Xte, Yte): #prune the subtree
        
        if self.isLeaf():
            return self.__classDistribution
        
        errorSubTree = DecisionTreeClassifier(root = self).test(Xte, Yte)[2]
        
        left, right = self.__split.partitioningTrain((Xte, Yte), False)
        left = (self.__left).pruning(left[0], left[1] )  if left[1].shape[0] != 0 else {label : 0.0 for label in self.__classes} # recursive calls left node
        right = (self.__right).pruning(right[0], right[1]) if right[1].shape[0] != 0 else {label : 0.0 for label in self.__classes} # recursive calls right node
        dictDistribution = { label : left[label] + right[label] for label in self.__classes}
        mostCommonClass = list(dictDistribution.keys())[np.argmax(list(dictDistribution.values()), axis=0)] # most common class in the subtree
        errorAsLeaf=list(mostCommonClass==Yte).count(False) / Yte.shape[0] 
        if errorAsLeaf < errorSubTree:
            self.__leaf=True
            self.__label, self.__classDistribution = self.__mostCommonClass(Yte)
            
        return dictDistribution

class Gini(Node):
    
    def __init__(self, X, Y, attributes, classes, max_depth):
        Node.__init__(self, X, Y, attributes, classes, max_depth, self.__gini)
        
        if self._Node__max_depth <= 0: # the maximum depth is reached
            self._Node__leaf = True
            self._Node__label, self._Node__classDistribution = self._Node__mostCommonClass(self._Node__Y)
            return
        
        if all(y == self._Node__Y[0] for y in self._Node__Y) : # All the tuples belong to the same class and there are less or equal to 1 tuples
            self._Node__leaf = True
            self._Node__label, self._Node__classDistribution = self._Node__mostCommonClass(self._Node__Y)
            return 
        
        self._Node__label, self._Node__gain, self._Node__split = self._Node__bestAttribute() # sets the features of the current node
        
        try:
            left, right = self._Node__split.partitioningTrain((self._Node__X, self._Node__Y))
            self._Node__left = Gini(left[0], left[1], self._Node__attributes, self._Node__classes, self._Node__max_depth - 1) # <=
            self._Node__right = Gini(right[0], right[1], self._Node__attributes, self._Node__classes, self._Node__max_depth - 1) # >
        except LeafRequiredException as err: # a partition is empty
            self._Node__leaf = True
            self._Node__label, self._Node__classDistribution = self._Node__mostCommonClass(self._Node__Y)
            return


    def __gini(self, Y): # the average amount of information needed to identify the class label of a tuple in X
        label, counts= np.unique(Y, return_counts=True)
        p = counts[counts > 0] / self._Node__n
        return np.sum(p * (1-p))
         
class Entropy(Node):
    
    def __init__(self, X, Y, attributes, classes, max_depth):
        Node.__init__(self, X, Y, attributes, classes, max_depth, self.__entropy)
        
        if self._Node__max_depth <= 0: # the maximum depth is reached
            self._Node__leaf = True
            self._Node__label, self._Node__classDistribution = self._Node__mostCommonClass(self._Node__Y)
            return 
        
        if all(y == self._Node__Y[0] for y in self._Node__Y) : # All the tuples belong to the same class and there are less or equal to 1 tuples
            self._Node__leaf = True
            self._Node__label, self._Node__classDistribution = self._Node__mostCommonClass(self._Node__Y)
            return 
        
        
        self._Node__label, self._Node__gain, self._Node__split = self._Node__bestAttribute() # sets the features of the current node
        
        try:
            left, right = self._Node__split.partitioningTrain((self._Node__X, self._Node__Y))
            self._Node__left = Entropy(left[0], left[1], self._Node__attributes, self._Node__classes, self._Node__max_depth -1) # <=
            self._Node__right = Entropy(right[0], right[1], self._Node__attributes, self._Node__classes, self._Node__max_depth -1) # >
        except LeafRequiredException as err: # a partition is empty
            self._Node__leaf = True
            self._Node__label, self._Node__classDistribution = self._Node__mostCommonClass(self._Node__Y)
            return

    
    def __entropy(self, Y): # the average amount of information needed to identify the class label of a tuple in X
        label, counts= np.unique(Y, return_counts=True)
        p = counts[counts > 0] / self._Node__n # if p_i=0 then p_i*log2⁡(p_i)=0 
        return -np.sum(p * np.log2(p))
    
def KFoldCVBestDepth(Xtr, Ytr, attributes_list, criterion, KF, max_depth_list):
    
    if KF <= 0:
        print("Please supply a positive number of repetitions")
        return -1

    # Ensures that k_list is a numpy array
    max_depth_list = np.array(max_depth_list)
    num_max_depth = max_depth_list.size

    n_tot = Xtr.shape[0]
    n_val = int(np.ceil(n_tot/KF))

    Tm = np.zeros(num_max_depth)
    Ts = np.zeros(num_max_depth)
    Vm = np.zeros(num_max_depth)
    Vs = np.zeros(num_max_depth)

    # Random permutation of training data
    rand_idx = np.random.choice(n_tot, size=n_tot, replace=False)
    
    
    for kdx, depth in enumerate(max_depth_list):
        first = 0
        for fold in range(KF):
           
            flags = np.zeros(Xtr.shape[0])
            flags[first:first+n_val]=1;
            
            X = Xtr[rand_idx[flags==0]]
            Y = Ytr[rand_idx[flags==0]]
            X_val = Xtr[rand_idx[flags==1]]
            Y_val = Ytr[rand_idx[flags==1]]

            # Compute the training error for the given value of depth
            Dt = DecisionTreeClassifier( depth, (X,Y), attributes_list, criterion)
            Dt.train()
            
            trError = Dt.test(X,Y)[2]
            Tm[kdx] = Tm[kdx] + trError
            Ts[kdx] = Ts[kdx] + trError ** 2

            # Compute the validation error for the given value of depth
            valError = Dt.test(X_val,Y_val)[2]
            Vm[kdx] = Vm[kdx] + valError
            Vs[kdx] = Vs[kdx] + valError ** 2
            
            first = first+n_val                

    Tm = Tm / KF
    Ts = Ts / KF - Tm ** 2

    Vm = Vm / KF
    Vs = Vs / KF - Vm ** 2

    best_max_depth_idx = np.argmin(Vm)
    best_max_depth = max_depth_list[best_max_depth_idx]

    return [best_max_depth], Vm, Vs, Tm, Ts