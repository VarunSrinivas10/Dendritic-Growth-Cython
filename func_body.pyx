import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport atan
from libc.math cimport cos
from libc.math cimport sin

# data type
ctypedef np.float64_t DT  
  
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#turn off array bounds check
@cython.boundscheck(False)

#turn off negative indices (u[-1,-1])  
@cython.wraparound(False) 

#Keep cdivision True  
@cython.cdivision(True)

#Function evolve 
cpdef evolve(double theta0,
             double kappa,
             int nx,
             int ny,
             int steps,
             double aniso,
             np.ndarray[DT, ndim=2, mode='c'] phi,
             np.ndarray[DT, ndim=2, mode='c'] tempr,
             np.ndarray[DT, ndim=2, mode='c'] lap_phi,
             np.ndarray[DT, ndim=2, mode='c'] lap_tempr,
             np.ndarray[DT, ndim=2, mode='c'] phidx,
             np.ndarray[DT, ndim=2, mode='c'] phidy,
             np.ndarray[DT, ndim=2, mode='c'] epsilon,
             np.ndarray[DT, ndim=2, mode='c'] epsilon_deriv,
             np.ndarray[DT, ndim=2, mode='c'] con_noise,
             np.ndarray[DT, ndim=1, mode='c'] random_no,
             np.ndarray[DT, ndim=3, mode='c'] A,
             np.ndarray[DT, ndim=3, mode='c'] B
             ):
    
                cdef:
                    int i 
                    int j 
                    int jp
                    int jm
                    int ip
                    int im
                    int step = 0
                    int nxny = nx*ny
                    int rand_count = 0
                    int row_num = 0
                    double hne
                    double hnw
                    double hnn
                    double hns
                    double hnc
                    double theta
                    double term1
                    double term2
                    double m
                    double phiold
                    double random_num
                    
                    #Spatial discretisation dx and dy
                    double dx = 0.05
                    double dy = 0.05
                    
                    #Temporal discretisation dt
                    double dtime = 0.0005
                    
                    #Other simulation related constants
                    double tau = 0.001
                    double epsilonb = 0.01
                    double mu = 1.0
                    double delta = 0.03
                    double alpha = 0.9
                    double gamma = 10.0
                    double teq = 1.0
                    double amp_noise = 0.01
                    double pi = 3.141592659
                
                
                #Start steps loop
                while step < steps:     
                    for i in range(nx):
                        for j in range(ny):
            
                            jp = j + 1
                            jm = j - 1
                            
                            ip = i + 1
                            im = i - 1
                            
                            if im == -1:
                                im = nx-1
                            if ip == nx:
                                ip = 0
                            
                            if jm == -1:
                                jm = ny-1
                            if jp == ny:
                                jp = 0
                            
                            #Laplacians
                                
                            hne = phi[ip,j]
                            hnw = phi[im,j]
                            hns = phi[i,jm]
                            hnn = phi[i,jp]
                            hnc = phi[i,j]
                            
                            lap_phi[i,j] =  (hnw + hne + hns + hnn -4.0*hnc)/(dx*dy)
                            
                            hne = tempr[ip,j]
                            hnw = tempr[im,j]
                            hns = tempr[i,jm]
                            hnn = tempr[i,jp]
                            hnc = tempr[i,j]
                            
                            lap_tempr[i,j] = (hnw + hne + hns + hnn -4.0*hnc)/(dx*dy)
                            
                            #Gradients
                            phidx[i,j] = (phi[ip,j] - phi[im,j])/dx
                            phidy[i,j] = (phi[i,jp] - phi[i,jm])/dy
                            
                            if phidx[i,j] == 0:
                                if phidy[i,j] < 0:
                                    theta = -0.5*pi
                                elif phidy[i,j] > 0:
                                    theta = 0.5*pi
                                else:
                                    theta = 0;
                            
                            else:
                                theta = atan(phidy[i,j]/phidx[i,j])
                                    
                            
                            #Epsilon and derivatives               
                            epsilon[i,j] = epsilonb*(1.0+delta*(cos(aniso*(theta-theta0))))
                            epsilon_deriv[i,j] = -epsilonb*aniso*delta*(sin(aniso*(theta-theta0)))
                            
                            #Noise
                            random_num = random_no[rand_count]
                            con_noise[i,j] = amp_noise*phi[i,j]*(1-phi[i,j])*random_num
                            rand_count+=1
                            
                           
                    for i in range(nx):
                        for j in range(ny):
                            
                            jp = j + 1
                            jm = j - 1
                            
                            ip = i + 1
                            im = i - 1
                            
                            if im == -1:
                                im = nx-1
                            if ip == nx:
                                ip = 0
                            
                            if jm == -1:
                                jm = ny-1
                            if jp == ny:
                                jp = 0
                            
                            phiold = phi[i,j]
                            
                            #First term
                            term1 =  (epsilon[i,jp]*epsilon_deriv[i,jp]*phidx[i,jp] - epsilon[i,jm]*epsilon_deriv[i,jm]*phidx[i,jm])/dy
            
                            #Second term
                            term2 = -(epsilon[ip,j]*epsilon_deriv[ip,j]*phidy[ip,j]-epsilon[im,j]*epsilon_deriv[im,j]*phidy[im,j])/dx
                            
                            #Factor m
                            m = (alpha/pi) * atan(gamma*(teq-tempr[i,j]))
            
                            #Time integration
                            phi[i,j] = phi[i,j] + (dtime/tau)*(term1 + term2 + ((epsilon[i,j])**2)*lap_phi[i,j] + phiold*(1.0-phiold)*(phiold-0.5+m) + con_noise[i,j])
                            
                            #Condition
                            if phi[i,j]<0.0:
                                phi[i,j] = 0
                                            
                            #Temp evolution
                            tempr[i,j] = tempr[i,j] +dtime*lap_tempr[i,j] + kappa*(phi[i,j]-phiold) 
                            
                            #Condition
                            if tempr[i,j]<0.0:
                                tempr[i,j] = 0
                                    
                    
                    if step%100 == 0:
                        A[row_num]  = phi.copy()
                        B[row_num]  = tempr.copy()
                        row_num = row_num + 1
                    
                    step = step + 1
                          
                return A,B
               
