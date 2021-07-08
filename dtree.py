import numpy as np

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        return self.rchild.predict(x_test)
        

class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        # return prediction
        return self.prediction
        
class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.std or gini

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressors predict the average y
        for samples in that leaf.  
              
        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)
        
    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621. create_leaf() depending
        on the type of self.
        
        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(X) <= self.min_samples_leaf:
            return self.create_leaf(y)
        col, split = self.bestsplit(X, y)
        if col == -1:
            return self.create_leaf(y)
        X_col = X[:, col]
        lchild = self.fit_(X[X_col<=split], y[X_col<=split])
        rchild = self.fit_(X[X_col>split], y[X_col>split])
        return DecisionNode(col, split, lchild, rchild)
    
    def bestsplit(self, X, y):
        best = {'col':-1, 'split':-1, 'loss':self.loss(y)}
        p = X.shape[1]   # p: number of features
        for col in range(p):
            X_col = X[:, col]
            #candidates = np.random.choice(X_col, size=11)
            candidates = np.random.choice(X_col, size=min(len(X_col),11), replace=False)
            for split in candidates:
                yl = y[X_col <= split]
                yr = y[X_col > split]
                yl_length = len(yl)
                yr_length = len(yr)
                if yl_length < self.min_samples_leaf or yr_length < self.min_samples_leaf:
                    continue
                l = (yl_length*self.loss(yl)+yr_length*self.loss(yr)) / len(y)
                if l == 0:
                    return col, split
                if l < best['loss']:
                    best['col'],  best['split'], best['loss']= col, split, l
        return best['col'], best['split']
       
        
    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """    
        if len(X_test.shape) == 1:
            return np.array(self.root.predict(X_test))
        return np.array([self.root.predict(record) for record in X_test])
    
    
class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.std)
        
    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        return 1 - ((y_test - y_pred)** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
        
    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))
        
        
class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)
        
    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        return sum(y_test == y_pred) / len(y_test)
        
    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.bincount(y).argmax())
    
    
def gini(y):
    """
    Return the gini impurity score for values in y"
    Reference: https://github.com/parrt/msds621/blob/master/notebooks/trees/gini-impurity.ipynb
        """
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p**2)