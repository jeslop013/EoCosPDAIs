# -*- coding: utf-8 -*-
"""Complexity72hrs.ipynb"""

import random
import numpy as np
import matplotlib.pyplot as plt

# # General Payoff values   T>R>P>S
# T = 5  # Temptation
# R = 3  # Reward
# P = 1  # Punishment
# S = 0  # Sucker's payoff

# Donation Game Payoff values   b>c

b = 1.1
c = 1

T,R,P,S = b, b-c ,0 ,-c


# Other parameters
alpha = 0.1      #fraction of AI
beta_H = 0.1 # Intesity of selections human-human interactions (how much the difference in payoff is valued in the choice)
beta_AI = 0.1 # " "  " human-AI interactions (how much the difference in payoff is valued in the choice when interacting with an AI)

# Functions

def SAM_payoff_difference(x,alpha,T,R,P,S):
  return (R-T)*((1-alpha)*x+alpha) + (S-P)*(1-alpha)*(1-x)

def SAM_evolution_eq(x,alpha,beta_H,beta_AI):
  return (1-x) * (x*(1-alpha)*np.tanh(beta_H* SAM_payoff_difference(x,alpha,T,R,P,S )*0.5) + alpha/ (1+np.exp(-beta_AI* SAM_payoff_difference(x,alpha,T,R,P,S ))))

def DISCR_payoff_difference(x,alpha,T,R,P,S):
  return (R-T)*((1-alpha)*x) + (S-P)*(1-alpha)*(1-x) + alpha*(R-P)

def DISCR_evolution_eq(x,alpha,beta_H,beta_AI):
  return (1-x) * x * (1-alpha) * np.tanh(beta_H* DISCR_payoff_difference(x,alpha,T,R,P,S )*0.5)


## Changing beta

x = np.linspace(0,1,100)

parameter_array = [0.1,0.2,0.5]
colors_SAM = ['orange','red','darkred']
colors_DISCR = ['cyan','blue','darkblue']

b = 6
c = 2
alpha = 0.2

for j in range(len(parameter_array)):
  beta = parameter_array[j]
  beta_AI,beta_H = beta,beta
  T,R,P,S = b, b-c ,0 ,-c
  print(alpha*b-c)
  plt.plot(x,SAM_evolution_eq(x,alpha,beta_H,beta_AI),label=r'$h(x)_{SAM,don},\beta$ = %.1f ' %beta,color=colors_SAM[j])

for j in range(len(parameter_array)):
  beta = parameter_array[j]
  beta_AI,beta_H = beta,beta
  T,R,P,S = b, b-c ,0 ,-c
  print(alpha*b-c)
  plt.plot(x,DISCR_evolution_eq(x,alpha,beta_H,beta_AI),linestyle='dotted',label=r'$h(x)_{DIS,don},\beta$ = %.1f ' %beta,color=colors_SAM[j])

plt.hlines(0,0,1,color="black",linewidths=0.5)
plt.xlabel(r'$x$')
plt.legend(fontsize='small',ncol=2)

plt.savefig('moving_beta.png',dpi=400)
plt.show()

## Changing alpha

x = np.linspace(0,1,100)

parameter_array = [0.1,0.2,0.5]
colors_SAM = ['cyan','blue','darkblue']

b = 6
c = 2

beta = 0.5
beta_AI,beta_H = beta, beta

for j in range(len(parameter_array)):
  alpha = parameter_array[j]
  T,R,P,S = b, b-c ,0 ,-c
  print(alpha*b-c)
  plt.plot(x,SAM_evolution_eq(x,alpha,beta_H,beta_AI),label=r'$h(x)_{SAM,don},\alpha$ = %.1f ' %alpha,color=colors_SAM[j])


for j in range(len(parameter_array)):
  alpha = parameter_array[j]
  T,R,P,S = b, b-c ,0 ,-c
  print(alpha*b-c)
  plt.plot(x,DISCR_evolution_eq(x,alpha,beta_H,beta_AI),linestyle='dotted',label=r'$h(x)_{DIS,don},\alpha$ = %.1f ' %alpha,color=colors_SAM[j])

plt.hlines(0,0,1,color="black",linewidths=0.5)
plt.xlabel(r'$x$')
plt.legend(fontsize='small',ncol=2)


plt.savefig('moving_alpha.png',dpi=400)
plt.show()

alpha = 0.1
alpha/(1-alpha) / (1+np.exp(beta_AI*c))

alpha = 0.5

C1 = np.tanh(beta*c*0.5)
C2 = 1/(1+np.exp(beta*c))
print(0.5*(1-alpha/(1-alpha)*C2/C1))

alpha = 0.2
b= 4
c= 2
print(alpha*b-c)

T,R,P,S = b, b-c ,0 ,-c
print(DISCR_payoff_difference(x,alpha,T,R,P,S))
print(SAM_payoff_difference(x,alpha,T,R,P,S))