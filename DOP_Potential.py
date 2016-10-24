#######################################################################################################################
########This function calculates the Fourier coefficients of the Coupled Kicked Rotor (Dirac Detla) Potential function.
#######################################################################################################################

import numpy as np
from sympy.physics.quantum import TensorProduct
def Potential(K1,K2,J,epsilon):
############ LIST OF CONSTANTS USED#################
#	K1		#Spring constant of Top 1
#	K2		#Spring constant of Top2
#	J		#Eignevalue of J^2 operator
#	epsilon		#Coupling strength of the two tops
	N=int(2*J+1)		#Dimension of the Hamiltonian of one top; Number of spin degenerate states permissible
	N2=N*N		#Dimension of the Tensor Product space
####################################################

	Jz=[[0 for x in range(N)] for y in range(N)]		#To calculate the representation of Jz
	Identity=[[0 for x in range(N)] for y in range(N)]	#To calculate the Identity

#To calculate the Jz (Angular Momentum along z axis)
	for x in range(N):
		m=-x+J
		Jz[x][x]=m
		Identity[x][x]=1

	Identity=np.matrix(Identity)
	Jz=np.matrix(Jz)
	Jz_sq=Jz*Jz
	J1=(Jz+0.5*Identity)
	J_sq=J1*J1
	Identity=np.matrix(Identity)
	V=(K1/(2*J))*TensorProduct(Jz_sq,Identity)+(K2/(2*J))*TensorProduct(Identity,J_sq)+(epsilon/J)*TensorProduct(Jz,J1)#To calculate the Fourier Coefficient of the Potential
	return V
