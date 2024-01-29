import numpy.random as rand
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class Individual:
    # c numero de coeficientes
    # d numero de variables
    def __init__(self, c, d):
        # Generate normal distributed coefficients for each variable plus the intercept
        self.values = [[rand.normal() for _ in range(c + 1)] for _ in range(d)]
        self.fitness = None

    #Evalua MSE
    def evaluate(self, lookupTable):
        self.fitness = 0
        # Para cada fila
        for x in lookupTable.keys():
            image = 0
            # Para cada variable
            for variable in self.values:
                # Para cada coeficiente
                for power, coefficient in enumerate(variable):
                    #valor polynomial
                    image += coefficient * x ** power
            # Compute squared error
            target = lookupTable[x]
            mse = (target - image) ** 2
            self.fitness += mse

    # Mutacion
    def mutate(self, rate):
        self.values = [[rand.uniform(c - rate, c + rate) for c in variable]
                       for variable in self.values]
                
class Population:
    costos = []
    def __init__(self, c, d, size=100):
        # Crea los individuos
        self.individuals = [Individual(c, d) for _ in range(size)]
        # alamacena los mejores individuos
        self.best = [Individual(c, d)]
        # Rate de Mutacion
        self.rate = 0.1

        plt.ion()

    def sort(self):
        self.individuals = sorted(self.individuals, key=lambda indi: indi.fitness)
                    
    def evaluate(self, lookupTable):
        for indi in self.individuals:
            indi.evaluate(lookupTable)

    def enhance(self, lookupTable, generation):
        newIndividuals = []
        # Toma los 10 mejores individuis
        for individual in self.individuals[:10]:
            newIndividuals.append(deepcopy(individual))
            # Crea 4 individuos mutados
            for _ in range(4):
                newIndividual = deepcopy(individual)
                newIndividual.mutate(self.rate)
                newIndividuals.append(newIndividual)
        # Remplaza con los nuevos individuos
        self.individuals = newIndividuals
        self.evaluate(lookupTable)
        self.sort()
        #Almacena los mejores individuos
        self.best.append(self.individuals[0])
        # Si la población no sufrio cambio alguno se incrementa la mutacion
        if self.best[-1].fitness == self.best[-2].fitness:
            self.rate += 0.01
        else:
            self.rate = 0.1
        self.costos.append(self.individuals[0].fitness)

    def plot_ga_convergence(self, costs):
        x = range(len(costs))
        plt.title("Convergencia GA")
        plt.xlabel('# Generación')
        plt.ylabel('error')
        plt.text(x[len(x) // 2], costs[0], 'costo minimo: {} '.format(costs[-1]), ha='center', va='center')
        plt.plot(x, costs, '-')


    def graficas(self, x, y, generation):
        X = np.linspace(min(x), max(x))
        Y = [sum(c * x ** p
          for p, c in enumerate(variable))
          for variable in self.best[-1].values
          for x in X]
        plt.clf()
        plt.subplot(121)
        self.plot_ga_convergence(self.costos)
        plt.subplot(122)
        plt.plot(X, Y, c='blue')
        plt.scatter(x, y, c='red', s=100)
        plt.title('Generacion : ' + str(generation) + ' / ' +
                  'Error : ' + str(self.best[-1].fitness))
        plt.pause(30)

