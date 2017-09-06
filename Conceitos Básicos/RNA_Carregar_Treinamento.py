from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import pickle

# Carregar Treinamento
fileObject = open('treinamento.txt','rb')
net = pickle.load(fileObject)

print("\nSOLUCOES E RESULTADOS\n")
print(net.activate([0,0]))
print(net.activate([1,0]))
print(net.activate([0,1]))
print(net.activate([1,1]))