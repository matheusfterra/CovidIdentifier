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

#Associação de exames/resultado
id_positivos = np.zeros((1, 1))
qtd_positivos=0
id_negativos = np.zeros((1, 1))
qtd_negativos=0

#FOR PARA IDENTIFICAR PACIENTES POSITIVOS E EXPORTAR
for n in range(0,len(data)):
    if data.iloc[n]["SARS-Cov-2 exam result"]=="positive":
        id_positivos[0][qtd_positivos]=n
        qtd_positivos=qtd_positivos+1
        # Incremento o vetor id_positivos com 1 posição a mais
        if qtd_positivos<positive:
            #Caso seja a ultima verificação, não precisa incrementar
            id_positivos = np.resize(id_positivos, (1, len(id_positivos[0]) + 1))
##Criação do cabeçalho com range total de positivos
#colunas_positivos=np.arange(1, positive+1)
#export_positivos = pd.DataFrame(data=id_positivos,columns=colunas_positivos)
#export_positivos.to_excel(r'data/positives.xlsx', engine='xlsxwriter', index = True, header=True)



#FOR PARA IDENTIFICAR PACIENTES NEGATIVOS E EXPORTAR
for n in range(0,len(data)):
    if data.iloc[n]["SARS-Cov-2 exam result"]=="negative":
        id_negativos[0][qtd_negativos]=n
        qtd_negativos=qtd_negativos+1
        # Incremento o vetor id_positivos com 1 posição a mais
        if qtd_negativos<negative:
            #Caso seja a ultima verificação, não precisa incrementar
            id_negativos = np.resize(id_negativos, (1, len(id_negativos[0])+1))
##Criação do cabeçalho com range total de positivos
#colunas_negativos=np.arange(1, negative+1)
#export_negativos = pd.DataFrame(data=id_negativos,columns=colunas_negativos)
#export_negativos.to_excel(r'data/negatives.xlsx', engine='xlsxwriter', index = True, header=True)

#Get dos Titulos das Colunas
colname = np.zeros((1, 105), dtype='U55')
n=0
for x in range(6,111):
    colname[0][n] = data.columns[x]
    n=n+1
#Calculo de Max/Min/Med
for x in range(0,105):
    #print(data[colname[0][x]].where((data["SARS-Cov-2 exam result"]=="positive")))
        #print("nao entrou")
    maximo_positivo=data[colname[0][x]].where((data["SARS-Cov-2 exam result"]=="positive")).dropna().max()
    minimo_positivo = data[colname[0][x]].where((data["SARS-Cov-2 exam result"] == "positive")).dropna().min()
    if type(maximo_positivo)==str and type(minimo_positivo)==str:
        media_positivo="NAN"
    else:
        media_positivo = (maximo_positivo + minimo_positivo) / 2
    #print(colname[0][x],"=",minimo_positivo," | ",media_positivo," | ", maximo_positivo)


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