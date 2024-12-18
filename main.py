import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

#load data from csv file
df = pd.read_csv("breast-cancer.csv")

#drop missing values
df.dropna(axis=1)

#drop duplicates
df.drop_duplicates(subset=None , keep="first" , inplace=False , ignore_index=False )

#convert diagnosis data  M to 0 and B to 1
df['diagnosis'].replace(to_replace="M" , value = 0 , inplace=True)
df['diagnosis'].replace(to_replace="B" , value = 1 , inplace=True)

#split the data set into independent x and dependant y
X = df.iloc[:,2:32].values   # feature by which could detect cancer
Y = df.iloc[:,1].values   # tell us has cancer or not

# split data set int 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25,random_state =0)


# create a function for the models
def models(X_train, Y_train):

    # KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, Y_train)

    #Decision Tree
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)

    #Naive Bayes
    NB = GaussianNB()
    NB.fit(X_train,Y_train)

    # print the models accuracy on the training data
    print('[0]knn Training Accuracy :', knn.score(X_train, Y_train))
    print('[2]Decision Tree Classifier Training Accuracy :', tree.score(X_train, Y_train))
    print('[3]Naive Bayes Training Accuracy :', NB.score(X_train,Y_train))
    return knn,tree ,NB

# getting all of the models
model = models(X_train, Y_train)

# test model accurecy on tset data on confussion matrix
for i in range (len(model)):
 print('model', i)
 cm = confusion_matrix(Y_test , model[i].predict(X_test))
 TP = cm[0][0]
 TN = cm[1][1]
 FN = cm[1][0]
 FP = cm[0][1]
 print(cm)
 print('testing accuracy :     '+str((TP+TN)/(TP + TN + FN + FP)))

