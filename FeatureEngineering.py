'''
Available methods are the followings:
[1] TimebasedFunction
[2] TimebaseFeatures (class)

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 30-09-2023

'''
import pandas as pd, numpy as np, os
from itertools import permutations, product

class ValidateParams:
    
    '''Validate parameters'''
    
    def Interval(self, Param, Value, dtype=int, 
                 left=None, right=None, closed="both"):

        '''
        Validate numerical input.

        Parameters
        ----------
        Param : str
            Parameter's name

        Value : float or int
            Parameter's value

        dtype : {int, float}, default=int
            The type of input.

        left : float or int or None, default=None
            The left bound of the interval. None means left bound is -∞.

        right : float, int or None, default=None
            The right bound of the interval. None means right bound is +∞.

        closed : {"left", "right", "both", "neither"}
            Whether the interval is open or closed. Possible choices are:
            - "left": the interval is closed on the left and open on the 
              right. It is equivalent to the interval [ left, right ).
            - "right": the interval is closed on the right and open on the 
              left. It is equivalent to the interval ( left, right ].
            - "both": the interval is closed.
              It is equivalent to the interval [ left, right ].
            - "neither": the interval is open.
              It is equivalent to the interval ( left, right ).

        Returns
        -------
        Value : float or int
            Parameter's value
            
        '''
        Options = {"left"    : (np.greater_equal, np.less), # a<=x<b
                   "right"   : (np.greater, np.less_equal), # a<x<=b
                   "both"    : (np.greater_equal, np.less_equal), # a<=x<=b
                   "neither" : (np.greater, np.less)} # a<x<b

        f0, f1 = Options[closed]
        c0 = "[" if f0.__name__.find("eq")>-1 else "(" 
        c1 = "]" if f1.__name__.find("eq")>-1 else ")"
        v0 = "-∞" if left is None else str(dtype(left))
        v1 = "+∞" if right is None else str(dtype(right))
        if left  is None: left  = -np.inf
        if right is None: right = +np.inf
        interval = ", ".join([c0+v0, v1+c1])
        tuples = (Param, dtype.__name__, interval, Value)
        err_msg = "%s must be %s or in %s, got %s " % tuples    

        if isinstance(Value, dtype):
            if not (f0(Value, left) & f1(Value, right)):
                raise ValueError(err_msg)
        else: raise ValueError(err_msg)
        return Value

    def StrOptions(self, Param, Value, options, dtype=str):

        '''
        Validate string or boolean inputs.

        Parameters
        ----------
        Param : str
            Parameter's name
            
        Value : float or int
            Parameter's value

        options : set of str
            The set of valid strings.

        dtype : {str, bool}, default=str
            The type of input.
        
        Returns
        -------
        Value : float or int
            Parameter's value
            
        '''
        if Value not in options:
            err_msg = f'{Param} ({dtype.__name__}) must be either '
            for n,s in enumerate(options):
                if n<len(options)-1: err_msg += f'"{s}", '
                else: err_msg += f' or "{s}" , got %s'
            raise ValueError(err_msg % Value)
        return Value
    
    def check_range(self, param0, param1):
        
        '''
        Validate number range.
        
        Parameters
        ----------
        param0 : tuple(str, float)
            A lower bound parameter e.g. ("name", -100.)
            
        param1 : tuple(str, float)
            An upper bound parameter e.g. ("name", 100.)
        '''
        if param0[1] >= param1[1]:
            raise ValueError(f"`{param0[0]}` ({param0[1]}) must be less"
                             f" than `{param1[0]}` ({param1[1]}).")
            
    def check_class(self, obj_name, obj, classinfo=(int)):
        
        '''
        Validate object.

        Parameters
        ----------
        obj_name : str
            Object's name.
            
        obj : object
            Object.

        classinfo : tuple, default=(int)
            Tuple of type objects.
            
        Returns
        -------
        obj : object
        
        '''
        if not isinstance(obj, classinfo):
            info = ", ".join([c.__name__ for c in classinfo])
            raise ValueError(f"`{obj_name}` must be ({info}). "
                             f"Got {type(obj)} instead.")
        else: return obj

def TimebasedFunction(n=(1,1), agg="mean", operand="divide", n_chars=10):
    
    '''
    Performs following operations:
        - divides x into 2 periods i.e. n1, and n2, where n1 + n2 
          is less than or equal to len(x).
        - aggregates x for each period using `agg` function.
        - performs the operation (`operand`) between two aggregated 
          outputs.
    
    Parameters
    ----------
    n : (int, int), default=(1,1)
        Numbers of periods (`n1`,`n2`). 

    agg : str or function, default="mean"
        Function to use for aggregating the data. If a function, it 
        must compute the aggregation of the flattened array only. The 
        accepted string function names follows numpy operation 
        functions. 
 
    operand: str or function, default="divide"
        If str, it follows numpy operation functions i.e. "subtract", 
        "add", "divide", and "multiply". If a function, it must 
        accept 2 parameters i.e. `x1` and `x2` as inputs.
        
    n_chars : int, default=10
        Number of characters to be kept as part of variable name i.e. 
        "{`agg`}_n{`n1`}_{`operand`}_n{`n2`}". This applies only to 
        `agg` and `operand`. 
        
    Returns
    -------
    compute : aggregation function
    
    '''
    # Validate parameters
    valid = ValidateParams()
    if callable(operand)==False:
        options = ["subtract", "add", "divide", "multiply"]
        operand = valid.StrOptions("operand", operand, options)
    n_chars = valid.Interval("n_chars", n_chars, dtype=int, 
                             left=1, right=None, closed="left")
    
    def compute(x):
        
        '''Aggregating data'''
        x  = np.array(x).flatten().copy()
        x1, x2 = x[:int(n[0])], x[int(n[0]):int(sum(n))]
        if len(x2)==0: return np.nan
        
        # Aggregate data
        if callable(agg): x1, x2 = agg(x1), agg(x2)
        else: x1, x2 = (getattr(np, agg)(x1), 
                        getattr(np, agg)(x2))
        
        if callable(operand):
            return operand(x1=x1, x2=x2)
        elif (operand=="divide") & (x2==0): 
            return np.nan
        else: return getattr(np, operand)(x1, x2)
    
    def find_name(fnc):
        name = fnc.__name__ if callable(fnc) else fnc
        return name[:1].upper() + name[1:].lower()
        
    # Function name
    names = (find_name(agg)[:n_chars], 
             find_name(operand)[:n_chars])
    name_ = f"{names[0]}_n{n[0]}_{names[1]}_n{n[1]}"
    compute.__name__ = name_
    return compute

class TimebaseFeatures(ValidateParams):
    
    '''
    Performs following operations:
        - divides x into 2 periods i.e. n1, and n2, where n1 + n2 
          is less than or equal to len(x).
        - aggregates x for each period using `agg` function.
        - performs the operation (`operand`) between two aggregated 
          outputs.
    
    Parameters
    ----------
    start : int, default=1
        Number of the most recent period.
    
    stop : int, default=1
        Number of the latter period.
    
    attr : str, default="less_equal"
        Sum of permuted pair must satisfies `attr` and `n_period` 
        condition e.g. start + stop <= 2. The accepted string function 
        names follows numpy operation functions.
    
    n_period : int, default=2
        A threshold of sum of permuted pair.
    
    agg : str or function, default="mean"
        Function to use for aggregating the data. If a function, it 
        must compute the aggregation of the flattened array only. The 
        accepted string function names follows numpy operation 
        functions.
 
    operand: str or function, default="divide"
        If str, it follows numpy operation functions i.e. "subtract", 
        "add", "divide", and "multiply". If a function, it must 
        accept 2 parameters i.e. `x1` and `x2` as inputs.
        
    n_chars : int, default=10
        Number of characters to be kept as part of variable name i.e. 
        "{`agg`}_n{`n1`}_{`operand`}_n{`n2`}". This applies only to 
        `agg` and `operand`. 
        
    '''
    def __init__(self, start=1, stop=1, attr="less_equal", n_periods=2, 
                 agg="mean", operand="divide", n_chars=10):
        
        # Validate parameters
        kwds = dict(left=1, closed="left")
        self.start = self.Interval("start", start, **kwds)
        self.stop = self.Interval("stop", stop, **kwds)
        self.n_chars = self.Interval("n_chars", n_chars, **kwds)
        self.n_periods = self.Interval("n_periods", n_periods, 
                                        **{**kwds,**{"left":2}})
        self.attr = getattr(np, attr)
        
        # Permutate set of periods
        periods = list(permutations(np.arange(self.start, self.stop+1),2))
        self.periods = [n for n in periods if self.attr(sum(n), self.n_periods)]
        
        # Create list of `agg` and `operand`
        self.agg = agg if isinstance(agg, list) else [agg]
        self.operand = operand if isinstance(operand, list) else [operand]
        
        # Create time-related functions
        args = product(self.periods, self.agg, 
                       self.operand, [self.n_chars])
        self.funcs = [TimebasedFunction(*a) for a in list(args)]
        
    def fit(self, X, by, columns=None):
        
        '''
        Fit model.
        
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            `X` must be sorted by date-time. if order is descending, `n1` 
            represents the most recent period while `n2` is the latter.
        
        by : list of labels
            Used to determine the groups for the group-by. 
            
        columns : list of labels, default=None
            List of column labels to be aggregated. If None, it defaults 
            to columns, whose dtype is either int or float.
            
        Attributes
        ----------
        func : dict
            Key represents column name, and value contains list of 
            aggregation functions e.g. {'feature_01': ["mean", "sum"]}.
        
        Returns
        -------
        self
        
        '''
        # Validate parameters
        X = self.check_class("X", X, (pd.DataFrame))
        self.by = self.check_class("by", by, (list))
        
        # Columns to be aggregated
        if columns is None:
            cols = X.columns[(X.dtypes==float) | (X.dtypes==int)]
        else: cols = self.check_class("columns", columns, list)
        self.columns = set(cols).difference(self.by)
        
        # Aggregations per column
        self.func = dict([(c, self.funcs) for c in self.columns])

        return self
    
    def transform(self, X, join=False):
        
        '''
        Transform X into aggregated X.
        
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            `X` must be sorted by date-time. if order is descending, `n1` 
            represents the most recent period while `n2` is the latter.
            
        join : bool, default=False
            If True, it joins MultiIndex columns with "_" e.g. 
            from MultiIndex(["feature_01","mean"]) to "feature_01_mean".
            
        Returns
        -------
        grouped_X : DataFrame of shape (n_groups, n_newfeatures)
            An aggregated X, where n_groups is number of unique keys, 
            and n_newfeatures is n_features * n_funcs.
            
        '''
        # Validate parameters
        X = self.check_class("X", X, pd.DataFrame)
        join = self.StrOptions("join", join, [True, False], bool)
        
        # Aggreate columns
        grouped_X = X.groupby(self.by).agg(self.func)
        if join: grouped_X.columns = ["_".join(n) for n in 
                                      grouped_X.columns]
        
        return grouped_X
    
    def fit_transform(self, X, by, columns=None, join=False):
        
        '''
        Transform X into aggregated X.
        
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            `X` must be sorted by date-time. if order is descending, `n1` 
            represents the most recent period while `n2` is the latter.
            
        by : list of labels
            Used to determine the groups for the group-by. 
            
        columns : list of labels, default=None
            List of column labels to be aggregated. If None, it defaults 
            to columns, whose dtype is either int or float.
            
        join : bool, default=False
            If True, it joins MultiIndex columns with "_" e.g. 
            from MultiIndex(["feature_01","mean"]) to "feature_01_mean".
            
        Returns
        -------
        grouped_X : DataFrame of shape (n_groups, n_newfeatures)
            An aggregated X, where n_groups is number of unique keys, 
            and n_newfeatures is n_features * n_funcs.
            
        '''
        self.fit(X, by, columns)
        return self.transform(X, by, join)