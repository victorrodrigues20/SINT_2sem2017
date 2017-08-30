from pybrain.tools.shortcuts import buildNetwork

# 2 entradas, 3 hidden, 1 saída
network = buildNetwork(2, 3, 1)
#network = buildNetwork(2, 3, 1, hiddenclass=TanhLayer, outclass=SoftmaxLayer)

print(network.activate([1,0]))

print(network['in'])
print(network['hidden0'])
print(network['out'])
print(network['bias'])