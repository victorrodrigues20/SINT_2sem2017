# Instalar biblioteca skfuzzy
# comando do Anaconda: pip install -U scikit-fuzzy

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Criação dos Conjuntos de Entrada
qualidade = ctrl.Antecedent(np.arange(0, 11, 1), 'qualidade')
servico = ctrl.Antecedent(np.arange(0, 11, 1), 'servico')

# Criação do Conjunto de Saída
gorjeta = ctrl.Consequent(np.arange(0, 26, 1), 'gorjeta')

# Função de Predicados automáticos - valores possíveis  .automf(3, 5, or 7)
# ficará com os valores poor (ruim), average (mediano) e good (bom)
qualidade.automf(3)
servico.automf(3)

# Função de população manual
gorjeta['pequena'] = fuzz.trimf(gorjeta.universe, [0, 0, 13])
gorjeta['media'] = fuzz.trimf(gorjeta.universe, [0, 13, 25])
gorjeta['alta'] = fuzz.trimf(gorjeta.universe, [13, 25, 25])

# Ver conjuntos fuzzy
qualidade.view()
servico.view()
gorjeta.view()

#Regras Fuzzy
regra1 = ctrl.Rule(qualidade['poor'] | servico['poor'], gorjeta['pequena'])
regra2 = ctrl.Rule(servico['average'], gorjeta['media'])
regra3 = ctrl.Rule(servico['good'] | qualidade['good'], gorjeta['alta'])

regra1.view()

# Criação do mecanismo Fuzzy
gorjeta_controle = ctrl.ControlSystem([regra1, regra2, regra3])
simulacaoGorjeta = ctrl.ControlSystemSimulation(gorjeta_controle)

simulacaoGorjeta.input['qualidade'] = 6.5
simulacaoGorjeta.input['servico'] = 9.8

# Calcular Simulação
simulacaoGorjeta.compute()


print("A gorjeta sera de ", simulacaoGorjeta.output['gorjeta'])
gorjeta.view(sim=simulacaoGorjeta)






