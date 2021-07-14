import numpy as np
from func_body import evolve

#domain size
nx = 256
ny = 256

#Number of steps
steps = 5000

#Parameters
aniso = 6.0
theta0 = 0.3
kappa = 2.2


#Pick initial configuration
#More starting conditions can be added to the list
#1 = Circular Nucleus at center
#2 = Left Wall
#3 = 4 Circular Nuclei at quadrant centers
#4 = Square Nucleus at center
#5 = Elliptical Nucleus at center

#Set seed
seed = 1

#Phase and Temperature arrays
phi           = np.zeros((nx,ny), dtype = np.double)
tempr         = np.zeros((nx,ny), dtype = np.double)

#Circular nucleus
if seed == 1:
    for i in range(nx):
        for j in range(ny):
            if (i-nx/2 -1)**2 + (j-ny/2 -1)**2 <= 9:
                phi[i,j] = 1.0
                tempr[i,j] = 1.0
#Left wall
elif seed == 2:
    for i in range(nx):
            phi[i,0:2] = 1.0
            tempr[i,0:2] = 1.0 

#4 nuclei
elif seed == 3:
    for i in range(nx):
        for j in range(ny):
            
            if (i-nx/4 -1)**2 + (j-ny/4 -1)**2 <= 4:
                phi[i,j] = 1.0
                tempr[i,j] = 1.0
            
            if (i-3*nx/4 -1)**2 + (j-ny/4 -1)**2 <= 4:
                phi[i,j] = 1.0
                tempr[i,j] = 1.0
            
            if (i-3*nx/4 -1)**2 + (j-3*ny/4 -1)**2 <= 4:
                phi[i,j] = 1.0
                tempr[i,j] = 1.0
            
            if (i-nx/4 -1)**2 + (j-3*ny/4 -1)**2 <= 4:
                phi[i,j] = 1.0
                tempr[i,j] = 1.0

#Square nucleus
elif seed == 4:
        phi[30:34,30:34]   = 1.0
        tempr[30:34,30:34] = 1.0

#Elliptical nucleus
elif seed == 5:
    for i in range(nx):
        for j in range(ny):
            if ((i-nx/2 -1)**2)/4 + ((j-ny/2 -1)**2)/9 <= 1:
                phi[i,j] = 1.0
                tempr[i,j] = 1.0   

#Arrays
lap_phi       = np.zeros((nx,ny), dtype = np.double)
lap_tempr     = np.zeros((nx,ny), dtype = np.double)
phidx         = np.zeros((nx,ny), dtype = np.double)
phidy         = np.zeros((nx,ny), dtype = np.double)
epsilon       = np.zeros((nx,ny), dtype = np.double)
epsilon_deriv = np.zeros((nx,ny), dtype = np.double)
con_noise     = np.zeros((nx,ny), dtype = np.double)
A             = np.zeros((50,nx,ny), dtype = np.double)
B             = np.zeros((50,nx,ny), dtype = np.double)
rand          = np.random.random((nx*ny*steps)) - 0.5    
				
#Evolve
p_f,t_f = evolve(theta0,kappa,nx,ny,steps,aniso,phi,tempr,lap_phi,lap_tempr,phidx,phidy,epsilon,epsilon_deriv,con_noise,rand,A,B)

#Cleanup
del con_noise, epsilon_deriv,epsilon, phidy, phidx, lap_tempr, lap_phi, phi, tempr
		
#Save
filename1 = 'Phase'
filename2 = 'Temp'
np.save(filename1,p_f)
np.save(filename2,t_f)
