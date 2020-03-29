import numpy
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.classifier import ClassificationReport

#IMPORTAÇÃO DOS DADOS
data = pd.read_excel("data/dataset.xlsx")
#ACRESCIMO DE COLUNA COM ID
data.insert(0, 'ID', range(0, len(data)))
#SET DE COLUNA COMO INDEX
data=data.set_index(data["ID"])

#DELETE DA COLUNA ID
del data["ID"]
data=data.drop_duplicates(keep=False)
#EXPORTAÇÃO DA TABELA
#data.to_excel (r'data/dataset_mod.xlsx', engine='xlsxwriter', index = True, header=True)

#Contagem de valores positivos e negativos
qtd_result=data["SARS-Cov-2 exam result"].value_counts()
negative=qtd_result[0]
positive=qtd_result[1]
print("O total de Testados é: {}\nA quantidade de pessoas testadas positivas é:{}\nA quantidade de pessoas testadas negativas é:{}\n\n".format(len(data),positive,negative))

####Plotagem de gráficos

# # set the background colour of the plot to white
# sns.set(style="whitegrid", color_codes=True)
# # setting the plot size for all plots
# sns.set(rc={'figure.figsize':(11.7,8.27)})
# # create a countplot
# sns.countplot('Patient age quantile',data=data,hue = 'SARS-Cov-2 exam result')
# # Remove the top and down margin
# sns.despine(offset=10, trim=True)
# plt.show()

#print(data["ctO2 (arterial blood gas analysis)"].dtypes)

le = preprocessing.LabelEncoder()
for x in range(0,111):
    if data[data.columns[x]].dtypes==object:
        data[data.columns[x]] = le.fit_transform(data[data.columns[x]].astype(str))

#print(data["ctO2 (arterial blood gas analysis)"])

target = data["SARS-Cov-2 exam result"]
# select columns other than 'Opportunity Number','Opportunity Result'
cols = [col for col in data.columns if col not in ["Patient ID","SARS-Cov-2 exam result","ID"]]
# dropping the 'Opportunity Number'and 'Opportunity Result' columns
data = data[cols]
#assigning the Oppurtunity Result column as target

data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)

#create an object of the type GaussianNB
gnb = GaussianNB()
#train the algorithm on training data and predict using the testing data
pred = gnb.fit(data_train, target_train).predict(data_test)
#print(pred.tolist())
#print the accuracy score of the model
print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred, normalize = True))

#
# #create an object of type LinearSVC
# svc_model = LinearSVC(random_state=0)
# #train the algorithm on training data and predict using the testing data
# pred = svc_model.fit(data_train, target_train).predict(data_test)
# #print the accuracy score of the model
# print("LinearSVC accuracy : ",accuracy_score(target_test, pred, normalize = True))

#create object of the lassifier
neigh = KNeighborsClassifier(n_neighbors=3)
#Train the algorithm
neigh.fit(data_train, target_train)
# predict the response
pred = neigh.predict(data_test)
# evaluate accuracy
print ("KNeighbors accuracy score : ",accuracy_score(target_test, pred))

visualizer = ClassificationReport(gnb, classes=['Won','Loss'])
visualizer.fit(data_train, target_train) # Fit the training data to the visualizer
visualizer.score(data_test, target_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data