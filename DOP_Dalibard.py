#######################################################################################################################################
########This function calculates the entanglement of the system specified by the variables declared below #############################
#######################################################################################################################################

import numpy as np
from numpy.linalg import *
import cmath
from sympy.physics.quantum import Dagger
from DOP_Heff import *
import matplotlib.pyplot as plt

im = cmath.sqrt(-1)

J=20.0			#Eigenvalue of J^2 operator, enter value as a float
p1=1.0			#Moment of Inertia of first top
p2=1.0			#Moment of Inertia of first top
K1=15.0			#Spring constant of Top 1
K2=10.0			#Spring constant of Top 2
epsilon=5.0		#Coupling strength of the two tops

Sv_array=[]		#To store the entanglement values for different eigenstates
Hamilt = Heff(J,p1,p2,K1,K2,epsilon)	#Calculates the Effective Hamiltonian
eigenvalues,eigenvectors=eigh(Hamilt)	#Calculates eigenvectors and eigenvalues of Hamiltonian using routines in numpy.linalg
eigenvectors=np.array(eigenvectors)
eigenvectors_tp=np.transpose(np.array(eigenvectors))
N=int(2*J+1)
N_sq=N*N

#To calculate the reduced density matrix of the system (C_Dagger * C)
for vector in eigenvectors_tp:
	C=np.zeros((N,N),dtype=np.complex64)
	dummy_index=0

	for i in range(N_sq):
		C[dummy_index][i%N]=vector[i]
		if i%N == N-1:
			dummy_index=dummy_index+1

	C=np.matrix(C)
	DensityMatrix1 = C*Dagger(C)
	Lambda=eigvalsh(DensityMatrix1)
	Sv=0.0
	for x in Lambda:
		if x > 0:
			Sv=Sv-x*np.log(x)
	
	Sv_array.append(Sv)

#plt.title('J=20, epsilon=5.0')
plt.axis([0, N_sq, -0.5, 4])
plt.xlabel('Eigenvalues')
plt.ylabel('Entanglement')
plt.plot(Sv_array,'bo')
plt.show()
