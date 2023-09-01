#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Data Cleaning Section
import pandas as pd


# In[2]:


import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,20)


# In[3]:


df1 = pd.read_csv("E:\\Bengaluru_House_Data.csv")
df1.head()


# In[4]:


df1.shape


# In[5]:


df1.groupby('area_type')['area_type'].agg('count')


# In[6]:


df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis = 'columns')
df2.head()


# In[7]:


df2.isnull().sum()


# In[8]:


df3 = df2.dropna()
df3.isnull().sum()
df3.shape


# In[9]:


df3['size'].unique()


# In[10]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.head()


# In[11]:


df3['bhk'].unique()


# In[12]:


df3[df3.bhk>20]


# In[13]:


df3.total_sqft.unique()


# In[14]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[15]:


df3[~df3['total_sqft'].apply(is_float)].head()


# In[16]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return(float(tokens[0])+ float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[17]:


convert_sqft_to_num('2166')


# In[18]:


df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)


# In[19]:


df4.head()


# In[20]:


df4.loc[30]


# In[21]:


(2100+2850)/2
#Data Cleaning section completes here


# In[22]:


#Feature Engineering
df4.head()


# In[23]:


df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()


# In[24]:


len(df5.location.unique())


# In[25]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending = False)
location_stats


# In[26]:


len(location_stats[location_stats<=10])


# In[27]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[28]:


len(df5.location.unique())


# In[29]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[30]:


#Feature Engineering Completes here
df5.head(10)


# In[31]:


#Outlier Removal
df5[df5.total_sqft/df5.bhk<300].head()


# In[32]:


df5.shape


# In[33]:


df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# In[34]:


df6.price_per_sqft.describe()


# In[35]:


def remove_apps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m= np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index= True)
    return df_out
df7 = remove_apps_outliers(df6)
df7.shape


# In[36]:


def plot_scatter_chart(df, location):
    bhk2 = df[(df.location==location) & (df.bhk == 2)]
    bhk3 = df[(df.location==location) & (df.bhk == 3)]

    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft, bhk2.price_per_sqft, color = 'blue', label = '2 BHK ', s = 50)
    plt.scatter(bhk3.total_sqft, bhk3.price_per_sqft, marker = '+', color = 'green', label = '3 BHK', s = 50)
    plt.xlabel = ("Total Square Feet Area ")
    plt.ylabel = ("Price Per Square Feet ")
    plt.title(location)
    plt.legend()

plot_scatter_chart(df7, "Rajaji Nagar")


# In[37]:


def remove_bhk_outlier(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats ={}
    for bhk, bhk_df in location_df.groupby('bhk'):
        bhk_stats[bhk] = {
          'mean': np.mean(bhk_df.price_per_sqft),
          'std': np.std(bhk_df.price_per_sqft),
           'count': bhk_df.shape[0]
      }
    for bhk, bhk_df in location_df.groupby('bhk'):
        stats = bhk_stats.get(bhk-1)
        if stats and stats['count']>5:
            exculde_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis = 'index')


# In[38]:


df8 = remove_bhk_outlier(df7)
df8.shape


# In[39]:


plot_scatter_chart(df7, "Hebbal")


# In[40]:


import matplotlib.pyplot as plt
matplotlib.rcParams["figure.figsize"] = (20, 20)
plt.hist(df8.price_per_sqft,rwidth = 0.8)
plt.xlabel = ("Price Per Square Feet")
plt.ylabel = ("Count")


# In[41]:


df8.bath.unique()


# In[42]:


df8[df8.bath>10]


# In[43]:


df8[df8.bath>df8.bhk+2]


# In[44]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[45]:


df10 = df9.drop(['size', 'price_per_sqft'], axis = 'columns')
df10.head(3)
#Outlier Removal Section Complete


# In[46]:


#Model Secion
dummies = pd.get_dummies(df10.location)
dummies.head(3)


# In[47]:


df11 = pd.concat([df10, dummies.drop('other', axis = 'columns')], axis = 'columns')
df11.head(3)


# In[48]:


df12 = df11.drop('location', axis = 'columns')
df12.head(2)


# In[49]:


df12.shape


# In[50]:


X = df12.drop('price', axis = 'columns')
X.head()


# In[51]:


y = df12.price
y.head()


# In[52]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)


# In[53]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)


# In[54]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits = 5, test_size = 0.2,random_state = 0)
cross_val_score(LinearRegression(), X, y, cv = cv)


# In[55]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso  # Corrected import
from sklearn.tree import DecisionTreeRegressor
def find_best_model_using_gridserch(X,y):
  algos = {
      'linear_regression' : {
          'model': LinearRegression(),
           'params': {
                'fit_intercept': [True, False],
                'n_jobs': [1, -1],
                'positive': [True, False]
            }
      },
      'lasso' : {
          'model': Lasso(),
          'params': {
              'alpha': [1, 2],
              'selection': ['random', 'cyclic']
          }
      },
      'decision_tree' : {
          'model': DecisionTreeRegressor(),
             'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
      }
  }

  scores = []
  cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
  for algo_name, config in algos.items():
      gs = GridSearchCV(config['model'], config['params'], cv = cv, return_train_score = False)
      gs.fit(X_train,y_train)
      scores.append({
          'model': algo_name,
          'best_score': gs.best_score_,
          'best_params': gs.best_params_
      })
  return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])


# In[56]:


find_best_model_using_gridserch(X_train,y_train)


# In[57]:


X.columns


# In[58]:


def predict_price(location, sqft, bath, bhk):
  loc_index = np.where(X.columns==location)[0][0]


  x = np.zeros(len(X.columns))
  x[0] = sqft
  x[1] = bath
  x[2] = bhk
  if loc_index >= 0:
    x[loc_index] = 1

  return lr_clf.predict([x])[0]


# In[59]:


predict_price('1st Phase JP Nagar', 1000, 2, 2)


# In[60]:


predict_price('1st Phase JP Nagar', 1000, 3, 3)


# In[61]:


predict_price('Indira Nagar', 1000, 2, 2)


# In[62]:


predict_price('Indira Nagar', 1000, 3, 3)


# In[63]:


import pickle
with open('banglore_home_prices_model.pickle', 'wb') as f:
  pickle.dump(lr_clf, f)


# In[64]:


import json
columns = {
    'data columns' : [col.lower() for col in X.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))


# In[ ]:




