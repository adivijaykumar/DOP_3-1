import numpy as np
from numpy.linalg import *
import scipy.linalg as sp
import cmath
from sympy.physics.quantum import Dagger, TensorProduct
from DOP_Heff import *
import matplotlib.pyplot as plt

im = cmath.sqrt(-1)

J = 0.5
p1=1
p2=1
K1=1
K2=1
epsilon = 0.5
N = int(2*J+1)

m_arr = np.linspace(J,-J,num=N)	
Jy= [[0 for x in range(N)] for x in range(N)]	#To calculate the representation of Jy
identity = np.identity(N)
 
#To calculate the representation of Jy using ladder operators
for index in range(N-1):
    Jy[index+1][index] = -np.sqrt((J+m_arr[index])*(J-m_arr[index]+1))
    Jy[index][index+1] = np.sqrt((J-m_arr[index+1])*(J+m_arr[index+1]+1))
Jy = np.matrix(Jy)

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


Uf=sp.expm(-im*0.5*np.pi*Jy)
Uk_1=sp.expm(-im*K1*Jz_sq/(2*J))
Uk_2=sp.expm(-im*K2*J_sq/(2*J))
U_12=sp.expm(-im*epsilon*Jz*J1)	#J_sq could be Jz*(Jz+0.5*Identity)

UT=TensorProduct(U_12,(TensorProduct(Uk_1*Uf,Uk_2*Uf)))	#Verify this!!!!!!!!!!!!

QuasiEnergy=sp.logm(UT)*im

eigenvalues,eigenvectors=eig(QuasiEnergy)