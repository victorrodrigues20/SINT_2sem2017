from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
import numpy as np
from sklearn import datasets
import numpy.random as random


# ---------------------------------------------------------------------
#
# Iris dataset
def splitWithProportion(self, proportion=0.7):

    '''
    Iremos criar 2 datasets. O primeiro contém apenas uma porção de exemplos, e o segundo conterá
    todos os exemplos
    '''

    # Criaremos uma variável randomica para pegar os índices
    indicies = random.permutation(len(self))
    separator = int(len(self) * proportion)

    leftIndicies = indicies[:separator]
    rightIndicies = indicies[separator:]

    leftDs = ClassificationDataSet(inp=self['input'][leftIndicies].copy(),
                                   target=self['target'][leftIndicies].copy())
    rightDs = ClassificationDataSet(inp=self['input'][rightIndicies].copy(),
                                    target=self['target'][rightIndicies].copy())
    return leftDs, rightDs


# Carregar Dataset Iris
irisData = datasets.load_iris()
# Armazenar características
dataFeatures = irisData.data
# Armazenar resultados
dataTargets = irisData.target

# shape
#print(irisData.DESCR)
#print(irisData.data)
#print(irisData.target)


# Criando dataset de classificação
dataSet = ClassificationDataSet(4, 1, nb_classes=3)

#print(dataFeatures[10])
#print(np.ravel(dataFeatures[10]))
#print(dataTargets[10])

# Adicionando cada exemplo no DataSet
for i in range(len(dataFeatures)):
    dataSet.addSample(np.ravel(dataFeatures[i]), dataTargets[i])

# 70% do DataSet será utilizado para testes
# 30% do DataSet será utilizado para testar os valores
trainingData, testData = splitWithProportion(dataSet, 0.7)

#print(trainingData)
trainingData._convertToOneOfMany()
#print(trainingData)

testData._convertToOneOfMany()

# Construindo Rede Neural
#print(trainingData.indim)
#print(trainingData.outdim)
neuralNetwork = buildNetwork(trainingData.indim, 7, trainingData.outdim, outclass=SoftmaxLayer)

# Treinando Treinador para a Rede Neural
trainer = BackpropTrainer(neuralNetwork, dataset=trainingData, momentum=0.01, learningrate=0.05, verbose=False)

# O treinamento pode ser executado da maneira abaixo:
#trainer.trainEpochs(10000)

for i in range(1, 10000):
    error = trainer.train()

    #if i % 1000 == 0:
    print("Interação ", i, " - Erro é: ", error)

print('\n\n')
registro = 0
for input in dataFeatures:
    print("Registro ", registro, " - Saída: ", neuralNetwork.activate(input))
    registro = registro + 1

