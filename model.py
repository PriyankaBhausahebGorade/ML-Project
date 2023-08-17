# Importing the libraries
'''import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')

dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X = dataset.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('ml_rainfall_data.csv')
df.drop(['day'],axis=1,inplace=True)
df_classi=df.copy()

df_classi['rainfall']=np.where(df_classi['rainfall']>0,1,0)

X=df_classi.drop(['rainfall'],axis=1)
y=df_classi['rainfall']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.preprocessing import StandardScaler
#ml=StandardScaler()
#X_train=ml.fit_transform(X_train)
#X_test=ml.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()

from sklearn.model_selection import GridSearchCV
grid={
    'bootstrap': [True],
    'max_depth': [3,5,6,7,9,10,15],
    'n_estimators': [10,30,50,80,100,150,200,250,300]
}

rrg=GridSearchCV(rf, param_grid = grid, scoring = 'roc_auc', cv=5)

rrg.fit(X_train,y_train)

pred=rrg.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print(accuracy_score(y_test,pred))

import pickle
pickle.dump(rrg,open('new_model_RF.pkl','wb'))

model_cls = pickle.load(open('new_model_RF.pkl', 'rb'))

#c=[[1982,4,95.78,26.44,34.19,7.73,267.56,7.76]]
#c=ml.transform(c)

#rrg.predict(c) ##correct one 14.2322

c=[[1982,4,95.78,26.44,34.19,7.73,267.56,7.76]]
#c=ml.transform(c)

print(rrg.predict(c)) ##correct one 14.2322
print()
print(model_cls.predict(c))