#Analysis
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier


import pandas

#Pearson correlation, Linear disciminant analysis (PCA, LDA)

def pearson_corr(dataframe, column, correlation):
    r, p = stats.pearsonr(dataframe[column], dataframe[correlation])
    return r

#Linear disciminant analysis
def linear_discriminant(dataframe):
    #Add dummies for nan values
    df_encoded = pd.get_dummies(dataframe, columns=['equipment', 'location'], drop_first=True)

    #Separating target and features
    x = df_encoded.drop(columns='faulty')
    y = df_encoded['faulty']

    #divide train and test data
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.4, random_state=0)

    #train and test model
    clf = lda()
    clf.fit(x_train, y_train)
    clf.score(x_train, y_train)
    y_pred = clf.predict(x_test)
    class_report = classification_report(y_test, y_pred)
    confusing_matrix=confusion_matrix(y_test, y_pred)
    print("_____________Accuracy________________")
    print(accuracy_score(y_test, y_pred))
    print("____________Confusion matrix__________")
    print(confusing_matrix)
    print("____________Class Report______________")
    print(class_report)

    return

def multi_layer_perceptron(dataframe):
    #Add dummies for nan values
    df_encoded = pd.get_dummies(dataframe, columns=['equipment', 'location'], drop_first=True)

    #Separating target and features
    x = df_encoded.drop(columns='faulty')
    y = df_encoded['faulty']

    #create train and test data
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.4, random_state=0)

    #Create MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=11,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=True
    )
    mlp.fit(x_train, y_train)

    #Predict
    y_pred = mlp.predict(x_test)

    #Evaluation
    print("_________MLP Accuracy score_________")
    print(accuracy_score(y_test, y_pred))
    print("_________Confusion Matrix___________")
    print(confusion_matrix(y_test, y_pred))

    return