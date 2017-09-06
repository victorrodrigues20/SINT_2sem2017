from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import pickle

net = buildNetwork(2, 4, 1, bias=True)

dataset = SupervisedDataSet(2,1)

dataset.addSample((0,0),(0))
dataset.addSample((1,0),(0))
dataset.addSample((0,1),(0))
dataset.addSample((1,1),(1))

trainer = BackpropTrainer(net, dataset=dataset, learningrate=0.01, momentum=0.06, verbose=True)

for i in range(1, 10000):
    error = trainer.train()

# Salvar Treinamento
fileObject = open('treinamento.txt', 'wb')
pickle.dump(net, fileObject)
fileObject.close()
