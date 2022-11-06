import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import f1_score#, balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#This class maps values in a column, numeric or categorical.
class MappingTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
    
    #now check to see if all keys are contained in column
    column_set = set(X[self.mapping_column])
    keys_not_found = set(self.mapping_dict.keys()) - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  set(self.mapping_dict.keys())
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result


 
#This class will rename one or more columns.
class RenamingTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    #now check to see if some column names in the mapping dict are not in the dataframe
    keys = set(self.mapping_dict.keys())
    cols = set(X.columns)

    diff = keys - cols
    
    assert len(diff) == 0, f'{self.__class__.__name__}. Unmatched column(s) "{diff}"'

    X_ = X.copy()
    X_.rename(columns=self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result



class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):  
    assert isinstance(target_column, str), f'{self.__class__.__name__} constructor expected string but got {type(target_column)} instead.'
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    target_column = set([self.target_column])
    cols = set(X.columns)
    diff = target_column - cols

    assert len(diff) == 0 , f'{self.__class__.__name__}. Unmatched column(s) "{diff}"'
    X_ = X.copy()
    X_ = pd.get_dummies(X_,
                  prefix=self.target_column,    #your choice
                  prefix_sep='_',     #your choice
                  columns=[self.target_column],
                  dummy_na=self.dummy_na,    #will try to impute later so leave NaNs in place
                  drop_first=self.drop_first    #really should be True but could screw us up later
                  )
    return X_
    
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
 

  
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'{self.__class__.__name__} action {action} not in ["keep", "drop"]'
    assert isinstance(column_list, list), f'{self.__class__.__name__} constructor expected list but got {type(column_list)} instead.'
    self.column_list = column_list
    self.action = action

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    column_list = set(self.column_list)
    cols = set(X.columns)
    diff = column_list - cols

    if len(diff) > 0:
      if self.action == 'drop':
        print(f"\nWarning: {self.__class__.__name__} does not contain these columns to drop: {diff}.\n")
      assert self.action != 'keep', f'{self.__class__.__name__} does not contain these columns to keep: {diff}'

    X_ = X.copy()
    if self.action == 'keep':
      diff = cols - column_list 
      X_.drop(diff, errors='ignore', axis=1, inplace=True)
    else:
      X_.drop(self.column_list, errors='ignore', axis=1, inplace=True)
    return X_
    
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
  
 
class PearsonTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, threshold):
    assert isinstance(threshold, float), f'{self.__class__.__name__} constructor expected float but got {type(threshold)} instead.'
    self.threshold = threshold

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    df_corr = X.corr(method='pearson')

    first_masked_df = df_corr.mask(df_corr.abs() > self.threshold, True)
    masked_df = first_masked_df.mask(first_masked_df <= self.threshold, False)

    upper_mask = np.triu(masked_df, k=1)
    upper_mask[np.where(upper_mask==0)]=False

    correlated_columns = [i[1] for i in enumerate(masked_df.columns) if True in upper_mask[:, i[0]]]

    new_df = transformed_df.drop(columns=correlated_columns)

    return new_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

 

class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_name):
    assert isinstance(column_name, str), f'{self.__class__.__name__} constructor expected float but got {type(column_name)} instead.'
    self.column_name = column_name

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.column_name in X.columns.to_list(), f'unknown column {self.column_name}'
    assert all([isinstance(v, (int, float)) for v in X[self.column_name].to_list()])

    #your code below
    mu = X[self.column_name].mean()
    sig = X[self.column_name].std()
    s3max = mu + 3 * sig
    s3min = mu - 3 * sig

    new_df1 = transformed_df.copy()
    new_df1['Fare'] = transformed_df['Fare'].clip(lower=s3min, upper=s3max)

    return new_df1

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

  
  
class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence):
    assert isinstance(target_column, str), f'{self.__class__.__name__} constructor expected string but got {type(target_column)} instead.'
    assert isinstance(fence, str), f'{self.__class__.__name__} constructor expected string but got {type(fence)} instead.'
    self.target_column = target_column
    self.fence = fence

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
    assert all([isinstance(v, (int, float)) for v in X[self.target_column].to_list()])

    new_df1 = X.copy()

    #fig, ax = plt.subplots(1,1, figsize=(3,9))
    #X.boxplot(self.column_name, vert=True, ax=ax, grid=True)  #normal boxplot

    if(self.fence == 'outer'):
      #now add on outer fences
      q1 = X[self.target_column].quantile(0.25)
      q3 = X[self.target_column].quantile(0.75)
      iqr = q3-q1
      outer_low = q1-3*iqr
      outer_high = q3+3*iqr
      #ax.scatter(1, outer_low, c='red', label='outer_low', marker="D", linewidths=5)
      #ax.text(1.1,  outer_low, "Outer fence")
      #ax.scatter(1, outer_high, c='red', label='outer_high', marker="D", linewidths=5)
      #ax.text(1.1,  outer_high, "Outer fence")
      #fig.show()

      new_df1[self.target_column] = X[self.target_column].clip(lower=outer_low, upper=outer_high)
    elif(self.fence == 'inner'):
      #now add on outer fences
      q1 = X[self.target_column].quantile(0.25)
      q3 = X[self.target_column].quantile(0.75)
      iqr = q3-q1
      inner_low = q1-1.5*iqr
      inner_high = q3+1.5*iqr
      #ax.scatter(1, inner_low, c='red', label='inner_low', marker="D", linewidths=5)
      #ax.text(1.1,  inner_low, "Inner fence")
      #ax.scatter(1, inner_high, c='red', label='inner_high', marker="D", linewidths=5)
      #ax.text(1.1,  inner_high, "Inner fence")
      #fig.show()

      new_df1[self.target_column] = X[self.target_column].clip(lower=inner_low, upper=inner_high)

    return new_df1

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

  
  
class MinMaxTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass  #takes no arguments

  #fill in rest below
  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    scaler =  MinMaxScaler()
    columnsX = X.columns.to_list()
    numpy_result = scaler.fit_transform(X)
    new_df = pd.DataFrame(numpy_result, columns=columnsX)
    new_df.describe(include='all').T

    return new_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
  
  
class KNNTransformer(BaseEstimator, TransformerMixin):
  def __init__(self,n_neighbors=5, weights="uniform"):
    assert isinstance(weights, str), f'{self.__class__.__name__} constructor expected string but got {type(weights)} instead.'
    assert isinstance(n_neighbors, int), f'{self.__class__.__name__} constructor expected int but got {type(n_neighbors)} instead.'
    self.n_neighbors = n_neighbors
    self.weights = weights

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    columns = list(X.columns)

    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=self.n_neighbors,        #a rough guess
                        weights=self.weights,                 #could alternatively have distance factor in
                        add_indicator=False)                  #do not add extra column for NaN

    imputed_data = imputer.fit_transform(X)  #instantiate

    new_df = pd.DataFrame(imputed_data, columns = columns)
    return new_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
  
  
def find_random_state(features_df, labels, n=200):
  model = KNeighborsClassifier(n_neighbors=5)

  var = []  #collect test_error/train_error where error based on F1 score

  #2 minutes
  for i in range(1, n):
      train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                      random_state=i, stratify=labels)
      model.fit(train_X, train_y)  #train model
      train_pred = model.predict(train_X)           #predict against training set
      test_pred = model.predict(test_X)             #predict against test set
      train_f1 = f1_score(train_y, train_pred)   #F1 on training predictions
      test_f1 = f1_score(test_y, test_pred)      #F1 on test predictions
      f1_ratio = test_f1/train_f1          #take the ratio
      var.append(f1_ratio)

  rs_value = sum(var)/len(var)  #get average ratio value
  idx = np.array(abs(var - rs_value)).argmin()  #find the index of the smallest value
  return idx



titanic_transformer = Pipeline(steps=[
    ('drop', DropColumnsTransformer(['Age', 'Gender', 'Class', 'Joined', 'Married',  'Fare'], 'keep')),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', MappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe', OHETransformer(target_column='Joined')),
    ('age', TukeyTransformer(target_column='Age', fence='outer')), #from chapter 4
    ('fare', TukeyTransformer(target_column='Fare', fence='outer')), #from chapter 4
    ('minmax', MinMaxTransformer()),  #from chapter 5
    ('imputer', KNNTransformer())  #from chapter 6
    ], verbose=True)



customer_transformer = Pipeline(steps=[
    ('id', DropColumnsTransformer(column_list=['ID'])),
    ('os', OHETransformer(target_column='OS')),
    ('isp', OHETransformer(target_column='ISP')),
    ('level', MappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('time spent', TukeyTransformer('Time Spent', 'inner')),
    ('minmax', MinMaxTransformer()),
    ('imputer', KNNTransformer())
    ], verbose=True)



def dataset_setup(full_table, label_column_name:str, the_transformer, rs, ts=.2):
  #your code below
  features = full_table.drop(columns=label_column_name)
  labels = full_table[label_column_name].to_list()

  X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=ts, shuffle=True,
                                                    random_state=rs, stratify=labels)
  
  X_train_transformed = the_transformer.fit_transform(X_train)

  X_test_transformed = the_transformer.fit_transform(X_test)

  x_trained_numpy = X_train_transformed.to_numpy()
  x_test_numpy = X_test_transformed.to_numpy()
  y_train_numpy = np.array(y_train)
  y_test_numpy = np.array(y_test)

  return x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy 



def titanic_setup(titanic_table, transformer=titanic_transformer, rs=40, ts=.2):
  x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy = dataset_setup(titanic_table, 'Survived',
                                                                           transformer,
                                                                           rs, ts)
  return x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy



def customer_setup(customer_table, transformer=customer_transformer, rs=76, ts=.2):
  x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy = dataset_setup(customer_table, 'Rating',
                                                                           transformer,
                                                                           rs, ts)
  return x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy
