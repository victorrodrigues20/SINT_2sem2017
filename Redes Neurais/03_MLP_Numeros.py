from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

neuralNetwork = buildNetwork(25, 20, 10, bias=True)

dataset = SupervisedDataSet(25,10)

dataset.addSample((0,1,1,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0),(1,0,0,0,0,0,0,0,0,0))
dataset.addSample((0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0),(0,1,0,0,0,0,0,0,0,0))
dataset.addSample((0,1,1,1,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,1,1,1,0),(0,0,1,0,0,0,0,0,0,0))
dataset.addSample((0,1,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,1,1,1,0),(0,0,0,1,0,0,0,0,0,0))
dataset.addSample((0,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,0,0,0,1,0),(0,0,0,0,1,0,0,0,0,0))
dataset.addSample((0,1,1,1,0,0,1,0,0,0,0,1,1,1,0,0,0,0,1,0,0,1,1,1,0),(0,0,0,0,0,1,0,0,0,0))
dataset.addSample((0,1,0,0,0,0,1,0,0,0,0,1,1,1,0,0,1,0,1,0,0,1,1,1,0),(0,0,0,0,0,0,1,0,0,0))
dataset.addSample((0,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0),(0,0,0,0,0,0,0,1,0,0))
dataset.addSample((0,1,1,1,0,0,1,0,1,0,0,1,1,1,0,0,1,0,1,0,0,1,1,1,0),(0,0,0,0,0,0,0,0,1,0))
dataset.addSample((0,1,1,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,0,1,1,1,0),(0,0,0,0,0,0,0,0,0,1))

trainer = BackpropTrainer(neuralNetwork, dataset=dataset, learningrate=0.01, momentum=0.06)

for i in range(1, 10000):
    error = trainer.train()

    if i % 1000 == 0:
        print("Error in iteration ", i, " is: ", error)
    if error < 0.0001:
        break
        '''print(neuralNetwork.activate([0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]))
        print(neuralNetwork.activate([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]))
        print(neuralNetwork.activate([0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0]))
        print(neuralNetwork.activate([0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0]))
        print(neuralNetwork.activate([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]))
        print(neuralNetwork.activate([0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0]))
        print(neuralNetwork.activate([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]))
        print(neuralNetwork.activate([0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]))
        print(neuralNetwork.activate([0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]))
        print(neuralNetwork.activate([0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0]))'''

print("\n\n\nSOLUCOES E RESULTADOS\n")
print(neuralNetwork.activate([0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]))
print(neuralNetwork.activate([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]))
print(neuralNetwork.activate([0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0]))
print(neuralNetwork.activate([0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0]))
print(neuralNetwork.activate([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]))
print(neuralNetwork.activate([0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0]))
print(neuralNetwork.activate([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]))
print(neuralNetwork.activate([0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]))
print(neuralNetwork.activate([0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]))
print(neuralNetwork.activate([0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0]))
