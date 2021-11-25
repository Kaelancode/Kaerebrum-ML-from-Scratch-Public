import numpy as np
import numexpr as ne
from numba import njit, jit, vectorize
from numba.experimental import jitclass
import pandas as pd
from utilsGit import check_x, check_y
from graphviz import Digraph
import Rfcython2
#from mlcode.RF import Rfcython2
#import os
#os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"]="0"


@njit(fastmath=True, cache=True)
def _gini(n_labels, sum_of_labels):
    totals = sum_of_labels
    if totals[:, -1] == 0:
        totals[:, -1] = 1
    else:
        totals[:, -1]
    #impurity = 1 - (np.sum(np.power(n_labels, 2), axis=0, keepdims=True)/np.power(totals, 2))
    impurity = 1 - (np.sum((n_labels*n_labels), axis=0).reshape(1,-1)/(totals * totals))
    #impurity = ne.evaluate("1 - (sum((n_labels*n_labels), axis=0).reshape(1,-1)/(totals * totals))")
    return impurity


class DecisionTreeClassifier:  # entropy or gini,
    def __init__(self, min_samples=2, max_depth=10, min_leaf=1, max_leaf_nodes=np.inf, n_feats=None, split='best', measure='gini', cat_thres=5):
        self.min_samples, self.max_depth, self._current_depth = min_samples, max_depth, 0
        self.tree = None
        self.n_feats = n_feats
        self.splitter = split
        self.min_leaf = 1 if not min_leaf else max(min_leaf, 1)
        self.cat_thres = cat_thres

        self.max_leaf_nodes = np.inf if not max_leaf_nodes else min(max_leaf_nodes, np.inf)
        self._node_count = 0
        self._leaf_count = 0
        self.track = 0

        self._inf_funct = self._entropy if measure == 'entropy' else self._gini
        self.scoring = self._entropy_score if measure == 'entropy' else self._gini_score

    def _gini_score(self, label_bins, total):
        return 1 - (np.sum(np.power(label_bins, 2))/np.power(total, 2))

    def _entropy_score(self, label_bins, total):
        prob = label_bins/total
        prob = np.where(prob != 0, prob, 1)
        return np.sum((prob) * -np.log2(prob))
    '''
    def check_categorical(self, X, dataframe):
        data_type = []
        cols = range(len(X.columns)) if dataframe else X.T
        for col in cols:
            values = X.iloc[:, col].values if dataframe else col
            values = np.unique(X.iloc[:, col].values) if dataframe else np.unique(col)
            if type(values[0]) == str or type(values[0]) == np.str_ or len(values) <= self.cat_thres:
                data_type.append('Cat')
            else:
                data_type.append('Cont')
        return data_type
    '''
    def check_categorical(self, X, dataframe):
        data_type = []
        cols = range(len(X.columns)) if dataframe else X.T
        for col in cols:
            values = X.iloc[:, col].values if dataframe else col
            if type(values[0]) == str or type(values[0]) == np.str_ or len(values) <= self.cat_thres:
                data_type.append('Cat')
            elif len(np.unique(values)) <= self.cat_thres:
                data_type.append('Cat')
            else:
                data_type.append('Cont')
        return data_type

    def train(self, X, y, data_types=None):
        # check and fillup all self variables
        self.columnheads = X.columns if isinstance(X, (pd.DataFrame)) else None
        self.X, self.n, self.m = check_x(X)
        self.n_feats = self.n if not self.n_feats else min(self.n_feats, self.n)
        self.y = check_y(y)
        self.labels = np.unique(self.y)
        self._trace = []
        self._feature_trace = []

        if data_types is not None:
            self._data_types = data_types
        else:
            self._data_types = self.check_categorical(X, True) if np.any(self.columnheads) else self.check_categorical(X, False)

        if type(self.labels[0]) == str or type(self.labels[0]) == np.str_:
            Y_encode = np.sum([(self.y == m)*i for i, m in enumerate(self.labels)], axis=0)
            Y_encode = np.atleast_2d(Y_encode)
        else:
            Y_encode = self.y

        Y_hotcoded = (np.arange(len(self.labels)) == np.atleast_2d(Y_encode))*1
        Y_hotcoded = np.asfortranarray(Y_hotcoded)
        self.tree = self._grow_tree(self.X, Y_encode, Y_hotcoded)
        self._trace = np.atleast_2d(self._trace)
        self.record = [np.insert(np.sum(self._trace[self._trace[:, 0] == i][:, 1:-1], axis=0), 0, i) for i in range(self._current_depth+1)]
        #if self.max_leaf_nodes is not np.inf:
            #self.tree = self._max_leaf_node(self.tree, self.max_leaf_nodes)
        self.tree = self._max_leafs(self.tree, self.max_leaf_nodes) if self.max_leaf_nodes is not np.inf else self.tree

    def _grow_tree(self, X, y, y_hot, depth=0, feat=None, node_impurity=None):
        self._node_count += 1
        self._current_depth = max(self._current_depth, depth)
        self.impurity = np.inf
        self.value = np.inf
        self.split_list = [0]
        self.y = y
        self.y_hot = y_hot

        feat_list = np.random.choice(self.n, self.n_feats, replace=False)
        #Y_hotcoded = y_hot
        sample_bins = np.sum(y_hot, axis=0)
        m = X.shape[0]
        #labels, n_of_each_label = np.unique(y, return_counts=True)
        #d = 0
        labels = sum([1 for i in sample_bins if i != 0])
        #if labels <= 0:
            #return
        #impurity = 1 - (np.sum(np.power(sample_bins, 2))/np.power(m, 2))
        #impurity = self.scoring(sample_bins, m)
        impurity = node_impurity if node_impurity is not None else self.scoring(sample_bins, m)
        #print(impurity)
        if(depth >= self.max_depth) or (labels == 1) or (m < self.min_samples):
            idx_max = np.argmax(sample_bins)
            majority_label = self.labels[idx_max]
            self._leaf_count += 1
            #print('Node Count:', self._node_count, '  Dep', self._current_depth,'Tracl leaf', self._track_leaf, 'leafs:', self._leaf_count,'Classleaf', DecisionTreeClassifier.llc)
            self._trace.append(np.array([depth, 0, 1, 2]))
            return Node(feat, value=majority_label, depth=depth, columns=self.columnheads, cat=None if feat is None else self._data_types[feat], n_samples=m, label_bins=sample_bins, score=impurity)
        else:
            #split_list = [self._find_best_split(X[:, feature], y)+(feature,) for feature in feat_list]
            '''
            split_list = [self._find_best_split(X[:, feature], y, sample_bins)+(feature,) if self._data_types[feature] == 'Cont' else self._find_best_split_Cat(X[:, feature], y_hot, sample_bins)+(feature,) for feature in feat_list]
            '''
            split_list = [self._find_best_split(X[:, feature], sample_bins, feature) if self._data_types[feature] == 'Cont' else self._find_best_split_Cat(X[:, feature], sample_bins, feature) for feature in feat_list]
            #print(split_list)
            #best_list = self.split_list
            #low_impurity, high_impurity, split_impurity, split_value, feature = best_list
            '''
            altered_list = split_list.copy()
            for i, z in enumerate(altered_list):
                #print('z', z[3])
                if type(z[3]) == str or type(z[3]) == np.str_:
                    altered_list[i] = list(altered_list[i])
                    altered_list[i][3] = 0
                    altered_list[i] = tuple(altered_list[i])
            altered_list.sort(key=lambda x: x[2:5:2])
            '''
            #dtype = [('low', float), ('high', float), ('impurity', float),('value', int), ('feature', int)]
            #altered_list = np.array(altered_list, dtype=dtype)
            #altered_list.sort(order=['impurity', 'value', 'feature'])
            #choicelist = altered_list.argsort(order=['impurity', 'value', 'feature'])

            '''
            altered_list = [x[2:] for x in split_list]
            for i, z in enumerate(altered_list):
                #print('z', z[3])
                if type(z[0]) == str or type(z[0]) == np.str_:
                    altered_list[i] = list(altered_list[i])
                    altered_list[i][0] = 0
                    altered_list[i] = tuple(altered_list[i])

            dtype = [('impurity', float), ('value', int), ('feature', int)]
            altered_list = np.array(altered_list, dtype=dtype)
            choicelist = altered_list.argsort()
            '''
            #altered_list.sort(order=['impurity', 'value', 'feature'])
            #altered_list.sort(key=lambda x: x[2:5:2])
            #np.sort(altered_list, kind='stable', order=['impurity', 'feature'])
            #print('a', altered_list.shape,  altered_list)
            #split_list = [np.insert(self._find_best_split(X[:, feature], y), 3, feature) for feature in feat_list]
            '''
            choice = np.random.choice(len(split_list), 1)[0] if self.splitter == 'rand' else 0
            split_impurity, split_value, feature = split_list[choice]
            below = X[:, feature] <= split_value
            above = X[:, feature] > split_value
            #print('Below', y[below].shape[0],  X[below].shape)
            #print('above', y[above].shape[0])
            '''
            while True:
                list_length = len(split_list)
                choice = np.random.choice(list_length, 1)[0] if self.splitter == 'rand' else 0
                #chosen = choicelist[choice]
                #split_impurity, low_impurity, high_impurity, split_value, feature = split_list[choice]

                low_impurity, high_impurity, split_impurity, split_value, feature = split_list[-1]
                #low_impurity, high_impurity, split_impurity, split_value, feature = best_list
                #for x in split_list:
                #    if x[-1] == feature:
                #        split_value = x[3]

                #split_value = [x[3] for x in split_list if x[-1] == feature][0]

                below = X[:, feature] <= split_value if self._data_types[feature] == 'Cont' else X[:, feature] == split_value
                above = X[:, feature] > split_value if self._data_types[feature] == 'Cont' else X[:, feature] != split_value
                y_below = y[below]
                y_above = y[above]
                if self._check_split(y_below, y_above):
                    #print('Below', y[below].shape[0])
                    #print('above', y[above].shape[0])
                    break
                else: #list_length == 1: #
                    #print('TURN node', y[below].shape[0])
                    #print('TURN node', y[above].shape[0])
                    idx_max = np.argmax(sample_bins)
                    majority_label = self.labels[idx_max]
                    self._leaf_count += 1
                    self._trace.append(np.array([depth, 0, 1, 2]))
                    return Node(feat, value=majority_label, depth=depth, columns=self.columnheads, cat=None if feat is None else self._data_types[feat], n_samples=m, label_bins=sample_bins, score=impurity)
                #else:
                    #altered_list.pop(choice)
                    #altered_list = np.delete(altered_list, choice)
                    #choicelist = np.delete(choicelist, choice)

        left_node = self._grow_tree(X[below], y[below], y_hot[below,:], depth+1, feature, low_impurity)
        right_node = self._grow_tree(X[above], y[above], y_hot[above, :], depth+1, feature, high_impurity)
        self._trace.append(np.array([depth, 1, 0, split_impurity]))
        self._feature_trace.append(np.array([feature, m, impurity, split_impurity]))
        return Node(feature, split_value, left_node, right_node, depth=depth, columns=self.columnheads, cat=self._data_types[feature], n_samples=m, label_bins=sample_bins, score=impurity, split_score=split_impurity)

    def _check_split(self, low, high):
        check = False if len(low) < self.min_leaf or len(high) < self.min_leaf else True
        return check

    def _find_best_split_Cat(self, X, labels_bins, feature):
        m = len(self.y_hot)
        m_frac = 1/m
        Y_hotcoded = self.y_hot
        '''
        labels, n_of_labels = np.unique(y, return_counts=True)
        #print(n_of_labels.shape)
        if type(labels[0]) == str or type(labels[0]) == np.str_:
            Y_encode = np.sum([(y == m)*i for i, m in enumerate(labels)], axis=0)
            Y_encode = np.atleast_2d(Y_encode)
            #encode_labels = [i for i in range(len(labels))]
        else:
            Y_encode = y
            #encode_labels = labels

        Y_hotcoded = (np.arange(len(self.labels)) == np.atleast_2d(Y_encode))*1
        '''
        x_labels = np.unique(X)

        lowest_impurity = 10
        split_value = None
        lowest_low_impurity = None
        lowest_high_impurity = None

        for xlabel in x_labels:
            idx_equal_x = X == xlabel
            idx_notequal_x = X != xlabel

            #labels_equal = np.sum(Y_hotcoded[idx_equal_x, :], axis=0, keepdims=True)
            #labels_not_equal = np.sum(Y_hotcoded[idx_notequal_x, :], axis=0, keepdims=True)
            labels_equal = np.sum(Y_hotcoded[idx_equal_x, :], axis=0)
            labels_not_equal = np.sum(Y_hotcoded[idx_notequal_x, :], axis=0)

            '''
            y_equal = y[idx_equal_x]
            y_not_equal = y[idx_notequal_x]

            _, labels_equal = np.unique(y_equal, return_counts=True)
            _, labels_not_equal = np.unique(y_not_equal, return_counts=True)
            '''
            #sum_labels_equal = np.atleast_2d(np.sum(labels_equal))
            #sum_labels_not_equal = np.atleast_2d(np.sum(labels_not_equal))
            sum_labels_equal = np.sum(labels_equal)
            sum_labels_not_equal = np.sum(labels_not_equal)


            #print('Cat', sum_labels_not_equal, 'labels',labels_not_equal.shape, 'typ shape',sum_labels_not_equal.shape)
            high_impurity = self.scoring(labels_not_equal, sum_labels_not_equal) if sum_labels_not_equal else 0
            low_impurity = self.scoring(labels_equal, sum_labels_equal) if sum_labels_equal else 0
            #node_impurity = (sum_labels_not_equal*m_frac)*high_impurity + (sum_labels_equal*m_frac)*low_impurity
            node_impurity = (sum_labels_not_equal*m_frac)*high_impurity + (sum_labels_equal*m_frac)*low_impurity
            if node_impurity < lowest_impurity:
                lowest_impurity = node_impurity
                split_value = xlabel
                lowest_low_impurity = low_impurity
                lowest_high_impurity = high_impurity

        if lowest_impurity < self.impurity:
            self.impurity = lowest_impurity
            self.split_list = [lowest_low_impurity, lowest_high_impurity, lowest_impurity, split_value, feature]
            return self.split_list
        elif lowest_impurity == self.impurity:
            try:
                if split_value != self.value:
                    self.value = split_value
                    self.split_list = [lowest_low_impurity, lowest_high_impurity, lowest_impurity, split_value, feature]
                    return self.split_list
            except TypeError:
                pass
        else:
            pass
        return self.split_list
        #return lowest_low_impurity, lowest_high_impurity, lowest_impurity, split_value

    def _find_best_split(self, X, labels_bins, feature):
        idx = np.argsort(X)
        X_sorted, y_sorted = X[idx], self.y[idx]
        m = len(y_sorted)
        frac_m = 1/m
        #labels, n_of_labels = np.unique(y_sorted, return_counts=True)
        Y_encode = y_sorted
        encode_labels = np.where(labels_bins != 0)[0]
        tail = m-1
        #labels, n_of_labels = np.unique(y_sorted, return_counts=True)
        #print('labels',labels.shape)
        '''
        # FOR y is y_hotcoded
        n = y_sorted.shape[1]
        encode_labels = list(range(n))
        for i in encode_labels:
            y_sorted[:, i] = y_sorted[:, i]*i
        Y_encode = np.sum(y_sorted, axis=1, keepdims=True)
        '''
        '''
        if type(labels[0]) == str or type(labels[0]) == np.str_:
            Y_encode = y_hot_sorted
            print(Y_encode, Y_encode.shape)
            encode_labels = [i for i in range(len(labels))]
        else:
            Y_encode = y_sorted
            encode_labels = labels
        '''
        '''
        # for normal y
        if type(labels[0]) == str or type(labels[0]) == np.str_:
            Y_encode = np.sum([(y_sorted == m)*i for i, m in enumerate(labels)], axis=0)
            Y_encode = np.atleast_2d(Y_encode)
            encode_labels = [i for i in range(len(labels))]
        else:
            Y_encode = y_sorted
            encode_labels = labels
        '''
        #low_list = [(Y_encode == label).cumsum().reshape(-1, 1) for label in encode_labels]
        #high_list = [(low_list[i][-1] - low_list[i]).reshape(-1, 1) for i, label in enumerate(encode_labels)]

        low_list = [(Y_encode == label).cumsum() for label in encode_labels]
        #low_list = [np.cumsum(Y_encode == label, dtype=int) for label in encode_labels]
        #low_list = list(map(lambda x: (Y_encode == x).cumsum(), encode_labels))
        high_list = [(low_list[i][tail] - low_list[i]) for i, label in enumerate(encode_labels)]

        #low_n_labels_list = np.asarray(low_list).T[0]
        #high_n_labels_list = np.asarray(high_list).T[0]

        low_n_labels_list = np.asarray(low_list)
        high_n_labels_list = np.asarray(high_list)
        #print('part',high_n_labels_list[0], 'type', type(high_n_labels_list), high_n_labels_list[0].dtype)
        '''
        high_n_labels_list = np.zeros([len(y), 1], dtype=int)
        low_n_labels_list = np.zeros([len(y), 1], dtype=int)

        #print('Ysort', y_sorted, 's', y_sorted.shape)
        #print('Y_encode', Y_encode, 's', Y_encode.shape)
        for label in encode_labels:
            low_list = ((Y_encode == label)).cumsum().reshape(-1, 1)
            high_list = (low_list[-1] - low_list).reshape(-1, 1)


            low_n_labels_list = np.hstack((low_n_labels_list, low_list))
            high_n_labels_list = np.hstack((high_n_labels_list, high_list))

        high_n_labels_list = np.delete(high_n_labels_list, 0, 1)
        low_n_labels_list = np.delete(low_n_labels_list, 0, 1)
        '''
        #_, uni_idx = np.unique(X_sorted[::-1], return_index=True)  # index of all unique values in reverse order
        _, uni_idx = np.unique(np.flipud(X_sorted), return_index=True)
        uni_idx = (m-1)-uni_idx  # get the idx of the last occurance of each unique value in X array
        uni_x_sorted = X_sorted[uni_idx]  # get array of last occurance of unique values in X
        high_n_labels_list = high_n_labels_list[:, uni_idx]
        low_n_labels_list = low_n_labels_list[:, uni_idx]

        sum_high_labels = np.sum(high_n_labels_list, axis=0, keepdims=True)
        sum_low_labels = np.sum(low_n_labels_list, axis=0, keepdims=True)
        #sum_high_labels = ne.evaluate("sum(high_n_labels_list, axis=1)")
        #sum_low_labels = ne.evaluate("sum(low_n_labels_list, axis=1)")
        #sum_high_labels = sum_high_labels.reshape(-1, 1)
        #sum_low_labels = sum_low_labels.reshape(-1, 1)
        #print('cvont', 's', high_n_labels_list.shape, high_n_labels_list, )
        #print('cvont', 's', sum_high_labels.shape, sum_high_labels)

        high_impurity = self._inf_funct(high_n_labels_list, sum_high_labels)
        #print('h', high_impurity, high_impurity.shape)
        low_impurity = self._inf_funct(low_n_labels_list, sum_low_labels)
        node_impurity = (sum_high_labels * frac_m)*high_impurity + (sum_low_labels * frac_m)*low_impurity
        #print('Nimpure1', node_impurity.shape, node_impurity)
        '''
        _, uni_idx = np.unique(X_sorted[::-1], return_index=True)  # index of all unique values in reverse order
        uni_idx = (m-1)-uni_idx  # get the idx of the last occurance of each unique value in X array
        uni_x_sorted = X_sorted[uni_idx]  # get array of last occurance of unique values in X
        node_impurity = node_impurity[uni_idx]
        #print('Nimpure2', node_impurity.shape)
        node_impurity = node_impurity[uni_idx]
        high_impurity = high_impurity[uni_idx]
        low_impurity = low_impurity[uni_idx]
        '''
        best_split_idx = np.argmin(node_impurity)
        #print('best split',best_split_idx, best_split_idx.shape)
        lowest_impurity = node_impurity[:, best_split_idx]
        lowest_high_impurity = high_impurity[:, best_split_idx]
        lowest_low_impurity = low_impurity[:, best_split_idx]
        split_value = uni_x_sorted[best_split_idx]
        #print('Unixsorted',uni_x_sorted[best_split_idx], uni_x_sorted .shape)
        #print('low low',lowest_low_impurity.shape)
        #print('split value',split_value)
        #print('Cont', lowest_low_impurity[0], lowest_high_impurity[0], lowest_impurity[0], split_value)
        if lowest_impurity[0] < self.impurity:
            self.impurity = lowest_impurity
            self.split_list = [lowest_low_impurity[0], lowest_high_impurity[0], lowest_impurity[0], split_value, feature]
            return self.split_list
        elif lowest_impurity[0] == self.impurity:
            try:
                if split_value <= self.value:
                    self.value = split_value
                    self.split_list = [lowest_low_impurity[0], lowest_high_impurity[0], lowest_impurity[0], split_value, feature]
                    return self.split_list
            except TypeError:
                self.value = split_value
                self.split_list = [lowest_low_impurity[0], lowest_high_impurity[0], lowest_impurity[0], split_value, feature]
                return self.split_list
        else:
            pass
        return self.split_list
        #return lowest_low_impurity[0], lowest_high_impurity[0], lowest_impurity[0], split_value

    def _gini(self, n_labels, sum_of_labels):
        totals = sum_of_labels
        totals[:, -1] = 1 if totals[:, -1] == 0 else totals[:, -1]
        #impurity = 1 - (np.sum(np.power(n_labels, 2), axis=0, keepdims=True)/np.power(totals, 2))
        #impurity = 1 - (np.sum((n_labels*n_labels), axis=0, keepdims=True)/(totals * totals))
        impurity = 1 - np.divide(np.sum((n_labels*n_labels), axis=0, keepdims=True),(totals * totals))
        #impurity = ne.evaluate("1 - (sum((n_labels*n_labels), axis=0).reshape(1,-1)/(totals * totals))")
        return impurity

    def _entropy(self, n_labels, sum_of_labels):
        totals = sum_of_labels
        totals[:, -1] = 1 if totals[:, -1] == 0 else totals[-1]
        prob = np.divide(n_labels, totals)
        #print('start 1',prob)
        prob = np.where(prob != 0, prob, 1)
        #print('EWnd', prob)
        entropy = np.sum((prob) * -np.log2(prob), axis=0, keepdims=True)
        return entropy

    def _information_gain(self, parent, high_split, low_split):
        pass

    def legend(self):
        return pd.DataFrame([np.array(self._data_types)], columns=self.columnheads)

    def records(self):
        df = pd.DataFrame(self.record, columns=['Depth', 'Split Node', 'Leaf'])
        df.set_index('Depth', inplace=True)
        return df

    def top_feature(self, n_features):
        self._feature_trace = np.atleast_2d(self._feature_trace)
        self._feature_trace[:, 1] = self._feature_trace[:, 1]/self.m # turn into weight: no of samples in node / total samples
        self._feature_trace[:, 3] = self._feature_trace[:, 2] - self._feature_trace[:, 3] # to get inf gain from the split
        impt_score = self._feature_trace[:, 1]*self._feature_trace[:, 3]
        impt_score = impt_score.reshape(-1, 1)
        self._feature_trace = np.append(self._feature_trace, impt_score, axis=1)
        features = np.unique(self._feature_trace[:, 0])
        features = features.astype(int)
        #cols = self.columnheads.values[features] if self.columnheads is not None else features
        scores = [[i, np.sum(self._feature_trace[self._feature_trace[:, 0] == i][:, -1])] for i in features]
        scores.sort(key=lambda x: x[1], reverse=True)
        scores = np.atleast_2d(scores)
        if self.columnheads is not None:
            r = scores[:, 0]
            r = r.astype(int)
            cols = self.columnheads.values[r]
            cols = cols.reshape(-1, 1)
            scores = np.append(scores, cols, axis=1)
            scores[:,[1, 2]] = scores[:,[2, 1]]  # switch cols between Feature names  and Scores
            top_feature_df = pd.DataFrame(scores, columns=['Features', 'Feature Names', 'Scores'])

        else:
            top_feature_df = pd.DataFrame(scores, columns=['Features', 'Scores'])
        top_feature_df.index.name = 'Impt'
        return top_feature_df

    def predict(self, X):
        X, n, m = check_x(X)
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if (self._data_types[node.feature] == 'Cont'):
            if (x[node.feature] <= node.threshold):
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:
            if (x[node.feature] == node.threshold):
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)

    def _convert_node(self, node, value, threshold=None):
        if node is None:
            return
        if threshold is None:
            if node.depth == value and node.value is None:
                idx_max = np.argmax(node.label_bins)
                majority_label = self.labels[idx_max]
                node.value = majority_label
        else:
            if node.depth == value and node.value is None and node.split_score == threshold:
                idx_max = np.argmax(node.label_bins)
                majority_label = self.labels[idx_max]
                node.value = majority_label
                return node

        if node.left is not None:
            self._convert_node(node.left, value, threshold)
        if node.right is not None:
            self._convert_node(node.right, value, threshold)
        return node

    def _max_leafs(self, node, max):
        if node is None:
            return node
        #self.trace = np.atleast_2d(self.trace)
        #self.record = [np.sum(self.trace[self.trace[:, 0] == i], axis=0) for i in range(self._current_depth+1)]
        for i, depth in enumerate(self.record):
            if (depth[1]+depth[2]) >= max:
                self.list = self._convert_node(node, i)
                self.record = self.record[:i+1]
                self.record[i][2] = self.record[i][2] + self.record[i][1]
                self.record[i][1] = 0
                if (depth[1]+depth[2]) > max and i != 0:
                    round = (depth[1]+depth[2]) - max
                    for j in range(int(round)):
                        thres = np.sort(self._trace[self._trace[:, 0] == (i-1)][:, -1])[j]
                        self._convert_node(node, i-1, thres)
                        self.record[i-1][1] = self.record[i-1][1]-1
                        self.record[i-1][2] = self.record[i-1][2]+1
                        self.record[i][2] = self.record[i][2]-2
                    return node
                return node
            else:
                max = max - depth[2]
        return node


class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, depth=None, columns=None, cat=None, n_samples=None, label_bins=None, score=None, split_score=None):
        self.feature, self.threshold, self.left, self.right = feature, threshold, left, right
        self.value, self.depth, self.dot, self.columns, self.cat = value, depth, None, columns, cat
        self.n_samples, self.label_bins = n_samples, label_bins
        self.score = score
        self.split_score = split_score
        self.tag = 0

    def is_leaf_node(self):
        return self.value is not None

    def is_left_node(self):
        return self.left is not None

    def is_right_node(self):
        return self.right is not None

    def formatted(self, indent=0, Left=False, Right=False):
        side = "Left" if Left else "Right" if Right else ""
        X_header = "" if self.columns is not None else "X"
        operator = "==" if self.cat == 'Cat' else "<="
        Feature_header = self.columns[self.feature] if self.columns is not None else self.feature
        if self.is_leaf_node():
            s = f'{side} Leaf depth {self.depth} N:{self.n_samples} score:{self.score:0.6f} value ={self.value} '
        else:
            left = self.left.formatted(indent+1, Left=True)
            right = self.right.formatted(indent+1, Right=True)
            s = f'{side} Node depth {self.depth} N:{self.n_samples} score:{self.score:0.6f} ({X_header} {Feature_header} {operator} {self.threshold})\n {left}\n {right}'

        return "               " * indent + s

    def view(self, res=64, engine='dot', output='png', font='Rubik'):
        '''
        Engines: dot, neato, twopi, circo
                fdp, osage, patchwork, sfdp

        res : In multiples of 32. 32, 64, 96, 128...to 300
        '''
        layout = engine
        resolut = str(res)
        self.dot = Digraph(comment='Decision Tree', engine=layout)
        self.dot.attr(dpi=resolut)
        self.dot.attr('node', fontname=font)

        #self.dot = Digraph(comment='Decision Tree')
        #self.dot.node('A', 'Root')
        dot = self._draw(dot=self.dot)
        dot.format = output
        a = dot
        dot.render('my_graph', view=True)

        return a

    def _draw(self, previous=None, dot=None, Left=False, Right=False):
        # string formats
        side = "Left" if Left else "Right" if Right else ""
        X_header = "" if self.columns is not None else "X"
        Feature_header = self.columns[self.feature] if self.columns is not None else self.feature
        operator = "==" if self.cat == 'Cat' else "<="
        root_splitscore = self.split_score if self.split_score is not None else 0
        # color schemas
        leaf_scheme = '#9fffa1' if self.score == 0.00 else '#c4ff9f' if self.score <=0.1 else '#dcff9f' if self.score < 0.3 else '#fff800' if self.score < 0.7 else '#ffcb30'
        node_scheme = '#11f9ff' if '1' in str(self.depth)[-1] or '6' in str(self.depth)[-1] else '#00ecff' if '2' in str(self.depth)[-1] or '7' in str(self.depth)[-1] else '#00dfff' if '3' in str(self.depth)[-1] or '8' in str(self.depth)[-1] else '#00d2ff' if '4' in str(self.depth)[-1] or '9' in str(self.depth)[-1] else '#ffbf00'
        #'lightgoldenrod1', 'lightcyan'

        rootshape, rootfillclr, rootclr, rootfontclr = 'octagon', 'lightskyblue', '#338be3', 'midnightblue'
        nodeshape, nodefillclr, nodeclr, nodefontclr = 'box', '#89ffff', '#00d2ff', 'steelblue4'
        leafshape, leaffillclr, leafclr, leaffontclr = 'oval', leaf_scheme,'#3eb265', 'saddlebrown'
        nodebgclr = "#89ffff"
        if previous is None:
            dot.attr('node', shape=rootshape, style='rounded, filled', fillcolor=rootfillclr, color=rootclr, fontcolor=rootfontclr)

            dot.node('Root', f'Root Node at depth {self.depth} ({X_header} {Feature_header} {operator} {self.threshold})\n Total: {self.n_samples}   counts = {self.label_bins}\n score = {self.score:0.6f}\n split_score = {root_splitscore:0.6f}')
            #self.left._draw('Root', dot, Left=True), self.right._draw('Root', dot, Right=True) if self.split_score

            if self.split_score:
                self.left._draw('Root', dot, Left=True)
                self.right._draw('Root', dot, Right=True)

        elif self.is_leaf_node():
            dot.attr('node', shape=leafshape, style='filled, rounded', fillcolor=leaffillclr, color=leafclr, fontcolor=leaffontclr)

            dot.node(f'{side}Leaf {self.depth} {self.value}' + previous, f'{side} Leaf at depth {self.depth}\n Total: {self.n_samples}   counts = {self.label_bins}\n score = {self.score:0.6f}\n value = {self.value}')

            dot.edge(previous, f'{side}Leaf {self.depth} {self.value}' + previous, label='Yes' if Left else 'No', fontname='Rubik', color='lightblue2')
        else:
            dot.attr('node', shape=nodeshape, style='rounded, filled', fillcolor=nodefillclr, bgcolor=nodebgclr, color=nodeclr, fontcolor=nodefontclr, label_scheme='2')
            #dot.node(f'Left {self.depth} {self.feature}' + previous, f'Left Node at depth {self.depth} (X {self.feature} <= {self.threshold})')
            #dot.node(f'right {self.depth} {self.feature}' + previous, f'Right Node at depth {self.depth} (X {self.feature} > {self.threshold})')
            #self.left.draw(f'Left {self.depth} {self.feature}' + previous, dot)
            #self.right.draw(f'right {self.depth} {self.feature}' + previous, dot)
            #dot.edge(previous, f'Left {self.depth} {self.feature}' + previous)
            #dot.edge(previous, f'right {self.depth} {self.feature}' + previous)

            dot.node(f'{side} Node {self.depth} {self.feature}' + previous, f'{side} Node at depth {self.depth} ({X_header} {Feature_header} {operator} {self.threshold})\n Total: {self.n_samples}   counts = {self.label_bins}\n score = {self.score:0.6f}\n split_score = {self.split_score:06f}')

            self.left._draw(f'{side} Node {self.depth} {self.feature}' + previous, dot, Left=True)
            self.right._draw(f'{side} Node {self.depth} {self.feature}' + previous, dot, Right=True)

            dot.edge(previous, f'{side} Node {self.depth} {self.feature}' + previous, label='Yes' if Left else 'No', label_scheme='2', fontname='Rubik', color='lightblue2')
        dot.attr(overlap='false')

        return dot

    def code(self):
        Prev = 'X'
        s = self._coding(Prev)
        #result = f'df= source.values\nfor X in df:\n{s}\n    Y.append(y)'
        result = f'for X in df:\n{s}'
        return result

    def _coding(self, prev=None, indent=1, Left=False, Right=False):
        space = "    " * indent
        operator = "==" if self.cat == 'Cat' else "<="
        if self.is_leaf_node():
            #s = f'y = {self.value if type(self.value) == str or type(self.value) == np.str_ else self.value}'
            # define classify within branch as node do not have values
            classify = '"'+self.value+'"'if type(self.value) == str or type(self.value) == np.str_ else self.value
            s = f'return {classify}'
        else:
            # define threshold within branch as leaves do not have thresholds
            threshold = '"'+self.threshold+'"' if (self.cat == 'Cat') and (type(self.value) == str or type(self.value) == np.str_) else self.threshold
            #left_data = f'L{prev}{self.depth}{self.feature}'
            #right_data = f'R{prev}{self.depth}{self.feature}'
            left = self.left._coding(prev, indent=indent+1, Left=True)
            right = self.right._coding(prev, indent=indent+1, Right=True)

            s = f'if ({prev}[{self.feature}] {operator} {threshold}):\n{left}\n{space}else:\n{right}'
        return space+s

    def __str__(self):
        #return self.formatted()
        return ' KaelanCode DecisionTree '

    def __repr__(self):
        #return str(self)
        return self.formatted()
