import pandas as pd
from lib.classes import Population
import matplotlib.pyplot as plt
import numpy as np
import time
from pylab import savefig

plt.style.use('ggplot')

#La funcion Polinomial utilizada fue del tipo Ax^3 + Bx^2 + Cx + D 
#Se generaron los valores para x^3+2x^2+3x+0 
df = pd.read_csv('data/valores_polinomiales.csv')
function = 'polinomial'
lookupTable = {}
for i, record in df.iterrows():
    key = record['X']
    lookupTable[key] = record[function]

generations = 200
terminos = 4
variables = 1

polynomials = Population(terminos, variables)
polynomials.evaluate(lookupTable)
polynomials.sort()

for g in range(generations):
    polynomials.enhance(lookupTable,g)

polynomials.graficas(df['X'], df[function], g)
