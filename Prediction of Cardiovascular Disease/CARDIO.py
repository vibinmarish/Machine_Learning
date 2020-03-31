import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df=pd.read_csv('D:/cardio.csv',sep=";")         #importing dataset and seperating it by ';'

df2=df.drop(['id'],axis=1)                      #Droping the 'id' dimension

def remove_outlier(df_in, col_name):            #Box Plot analysis
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)] #Removing all the row which has outliers
    return df_out
x0=remove_outlier(df,'age')

x=x0.iloc[:,:11]                                #separating the features till the target class

y=x0.iloc[:,11:]                                #seperating the feature and storing only target class


x1=StandardScaler().fit_transform(x)             #using standardscaler to standardise the features to unit scale and get mean=0 and standard deviation=1

pca=PCA().fit(x1)                                #Doing PCA and fitting the model
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_)) #Plotting the explained variance ratio to find the right n_components(no of dimentions to reduce) value for doing PCA
plt.show()                                      #Showing the plot

pca=PCA(n_components=4)                         #Doing PCA to reduce dimensions to 4
dataset = pca.fit_transform(x1)                  #Fitting and transforming the pca with 'x' 


X_train,X_test,y_train,y_test=train_test_split(dataset,y['cardio'],test_size=0.2) #Splitting the feature set for training and testing 
knn=KNeighborsClassifier(n_neighbors=11)                                          #KNN Classifier

knn.fit(X_train,y_train)                                                          #Training and fitting the model

print(knn.score(X_test,y_test))                                                   #Accuracy of the trained model

y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))







 


 


