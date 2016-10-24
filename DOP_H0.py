#######################################################################################################################
########This function calculates the Kinetic Energy Operator of the Coupled Kicked Rotor (Dirac Detla) Potential function.
#######################################################################################################################

import numpy as np
import cmath
from sympy.physics.quantum import TensorProduct as tp

def KineticEnergy(p1,p2,J): 
#############LIST OF CONSTANTS################
#	p1		#Moment of Inertia of first top
#	p2		#Moment of Inertia of second top
#	J		#Eigenvalue of J^2 operator
	im = cmath.sqrt(-1)
 ###############################################
	N = int(2*J + 1)				#Dimension of the Hilbert space
	m_arr = np.linspace(J,-J,num=N)			#List of spin eigenvalues
	
	Jy= [[0 for x in range(N)] for x in range(N)]	#To calculate the representation of Jy
	H0 = [[0 for x in range(N)] for x in range(N)]	#Kinetic Energy Operator
	identity = np.identity(N)
 
#To calculate the representation of Jy using ladder operators
	for index in range(N-1):
	    Jy[index+1][index] = -np.sqrt((J+m_arr[index])*(J-m_arr[index]+1))
	    Jy[index][index+1] = np.sqrt((J-m_arr[index+1])*(J+m_arr[index+1]+1))

	Jy = np.matrix(Jy)
	identity = np.matrix(identity)
	H0 = (-0.5*im)*(p1*tp(Jy,identity)+p2*tp(identity,Jy))    #To calculate the Kinetic Energy Operator of the Coupled Kicked Rotor
 	return np.matrix(H0)
 
