###################################################################################################################################################
########This function calculates the Effective Hamiltonian of the Coupled Kicked Rotor (Dirac Detla) Potential function using Dalibard's expansion.
###################################################################################################################################################
import numpy as np
from numpy.linalg import *
import cmath
from sympy.physics.quantum import Dagger
from DOP_Potential import *
from DOP_H0 import *

im = cmath.sqrt(-1)

#To define the Commutator of pair of Operators
def Comm(A,B):
	A=np.matrix(A)
	B=np.matrix(B)
	C=A*B - B*A
	return C

def Heff(J,p1,p2,K1,K2,epsilon):
#	J=10.0		#Eigenvalue of J^2 operator
#	p1=1		#Moment of Inertia of first top
#	p2=1		#Moment of Inertia of first top
#	K1=10		#Spring constant of Top 1
#	K2=15		#Spring constant of Top 2
#	epsilon=0.0	#Coupling strength of the two tops

	Omega=2*np.pi			#Frequency of the Potential (=2*PI/T (T=1) )
	H0=KineticEnergy(p1,p2,J)	#Kinetic Energy Operator
	V=Potential(K1,K2,J,epsilon)	#Fourier Coefficients of the Potential Operator
	Hprime= Comm(Comm(V,H0),V)
	Hprime=Hprime+Dagger(Hprime)
	Hamilt=H0+V+1/(2*Omega*Omega)*((np.pi)**2)/6*Hprime #H_eff for a Coupled Kicked Rotor System approximated upto an accuracy of (1/W*W)
	return Hamilt

