import math
import scipy as sp
from scipy import integrate
import numpy as np
import pandas
import matplotlib.pyplot as plt 
import matplotlib.tri as tri
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def CB(Ni,Nj,Nazi,Np,R,J):
    # i = radial 
    # j = sector 
    # rho = distance from center
    # theta = angle 
    # R : radius of sphere
    # Nj : number of sectors
    # Ni : number of radial sections
    # (r,z) vertices
    # J = number of corners in a cell
    
    #print("CB-PWLD")
    #print("Given")
    #print("Q = ",Q," , sigma_t = ",Sig_t," , sigma_s = ",Sig_s," , psi_inc = ",psi_inc)
    #print("N_r = ",Ni," , N_z = ",Nj," , N_p = ",Np," , N_azi = ",Nazi,"\n")
    rho = np.zeros(Ni)
    rho_half = np.zeros(Ni+1)
    theta = np.zeros(Nj)
    theta_half = np.zeros(Nj+1)
    r = np.zeros((Ni,Nj,J))
    z = np.zeros((Ni,Nj,J))
    r_k = np.zeros((Ni,Nj))
    z_k = np.zeros((Ni,Nj))
    
    #determining the vertices
    for j in range(0,Nj):
        for i in range(0,Ni):
            rho_half[i+1] = ((i+1)/Ni)*R
            rho_half[0] = (10**(-6))*rho_half[i+1]
            rho[i] = (rho_half[i]+rho_half[i+1])/2
            theta_half[j] = ((j/Nj)*np.pi)-(np.pi/2)
            theta_half[j+1] = (((j+1)/Nj)*np.pi)-(np.pi/2)
            theta[j] = (theta_half[j]+theta_half[j+1])/2
            r[i,j,0] = rho_half[i]*np.cos(theta_half[j])
            r[i,j,1] = rho_half[i+1]*np.cos(theta_half[j])
            r[i,j,2] = rho_half[i+1]*np.cos(theta_half[j+1])
            r[i,j,3] = rho_half[i]*np.cos(theta_half[j+1])
            z[i,j,0] = rho_half[i]*np.sin(theta_half[j])
            z[i,j,1] = rho_half[i+1]*np.sin(theta_half[j])
            z[i,j,2] = rho_half[i+1]*np.sin(theta_half[j+1])
            z[i,j,3] = rho_half[i]*np.sin(theta_half[j+1])
            r_k[i,j] = (1/4)*(r[i,j,0]+r[i,j,1]+r[i,j,2]+r[i,j,3])
            z_k[i,j] = (1/4)*(z[i,j,0]+z[i,j,1]+z[i,j,2]+z[i,j,3])
            
    #
    r_c_plusHalf = np.zeros((Ni,Nj,J))
    r_c_minusHalf = np.zeros((Ni,Nj,J))
    l_c_plus = np.zeros((Ni,Nj,J))
    l_c_minus = np.zeros((Ni,Nj,J))
    l_c_plusHalf = np.zeros((Ni,Nj,J))
    l_c_minusHalf = np.zeros((Ni,Nj,J))
    area_plusHalf = np.zeros((Ni,Nj,J))
    area_minusHalf = np.zeros((Ni,Nj,J))
    
    for j in range(0,Nj):
        for i in range(0,Ni):
            for c in range(0,J):
                if c==0:
                    r_c_plusHalf[i,j,c] = (r[i,j,c]+r[i,j,c+1])/2
                    r_c_minusHalf[i,j,c] = (r[i,j,c]+r[i,j,J-1])/2
                    l_c_plus[i,j,c] = np.sqrt((r[i,j,c+1]-r[i,j,c])**2+(z[i,j,c+1]-z[i,j,c])**2)/2
                    l_c_minus[i,j,c] = np.sqrt((r[i,j,J-1]-r[i,j,c])**2+(z[i,j,J-1]-z[i,j,c])**2)/2
                    l_c_plusHalf[i,j,c] = np.sqrt((r_k[i,j]-((r[i,j,c+1]+r[i,j,c])/2))**2
                                +(z_k[i,j]-((z[i,j,c+1]+z[i,j,c])/2))**2)
                    l_c_minusHalf[i,j,c] = np.sqrt((r_k[i,j]-((r[i,j,J-1]+r[i,j,c])/2))**2
                                +(z_k[i,j]-((z[i,j,J-1]+z[i,j,c])/2))**2)
                    area_plusHalf[i,j,c] = ((1/2)*(z_k[i,j]*(r[i,j,c+1]-r[i,j,c])+ 
                                 z[i,j,c+1]*(r[i,j,c]-r_k[i,j])+z[i,j,c]*(r_k[i,j]-r[i,j,c+1])))
                    area_minusHalf[i,j,c] = (-(1/2)*(z_k[i,j]*(r[i,j,J-1]-r[i,j,c])+ 
                                 z[i,j,J-1]*(r[i,j,c]-r_k[i,j])+z[i,j,c]*(r_k[i,j]-r[i,j,J-1])))
                    
                elif c==J-1:
                    r_c_plusHalf[i,j,c] = (r[i,j,c]+r[i,j,0])/2
                    r_c_minusHalf[i,j,c] = (r[i,j,c]+r[i,j,c-1])/2
                    l_c_plus[i,j,c] = np.sqrt((r[i,j,0]-r[i,j,c])**2+(z[i,j,0]-z[i,j,c])**2)/2
                    l_c_minus[i,j,c] = np.sqrt((r[i,j,c-1]-r[i,j,c])**2+(z[i,j,c-1]-z[i,j,c])**2)/2
                    l_c_plusHalf[i,j,c] = np.sqrt((r_k[i,j]-((r[i,j,0]+r[i,j,c])/2))**2
                                +(z_k[i,j]-((z[i,j,0]+z[i,j,c])/2))**2)
                    l_c_minusHalf[i,j,c] = np.sqrt((r_k[i,j]-((r[i,j,c-1]+r[i,j,c])/2))**2
                                +(z_k[i,j]-((z[i,j,c-1]+z[i,j,c])/2))**2)
                    area_plusHalf[i,j,c] = ((1/2)*(z_k[i,j]*(r[i,j,0]-r[i,j,c])+ 
                                 z[i,j,0]*(r[i,j,c]-r_k[i,j])+z[i,j,c]*(r_k[i,j]-r[i,j,0])))
                    area_minusHalf[i,j,c] = (-(1/2)*(z_k[i,j]*(r[i,j,c-1]-r[i,j,c])+ 
                                 z[i,j,c-1]*(r[i,j,c]-r_k[i,j])+z[i,j,c]*(r_k[i,j]-r[i,j,c-1])))
                else:
                    r_c_plusHalf[i,j,c] = (r[i,j,c]+r[i,j,c+1])/2
                    r_c_minusHalf[i,j,c] = (r[i,j,c]+r[i,j,c-1])/2
                    l_c_plus[i,j,c] = np.sqrt((r[i,j,c+1]-r[i,j,c])**2+(z[i,j,c+1]-z[i,j,c])**2)/2
                    l_c_minus[i,j,c] = np.sqrt((r[i,j,c-1]-r[i,j,c])**2+(z[i,j,c-1]-z[i,j,c])**2)/2
                    l_c_plusHalf[i,j,c] = np.sqrt((r_k[i,j]-((r[i,j,c+1]+r[i,j,c])/2))**2
                                +(z_k[i,j]-((z[i,j,c+1]+z[i,j,c])/2))**2)
                    l_c_minusHalf[i,j,c] = np.sqrt((r_k[i,j]-((r[i,j,c-1]+r[i,j,c])/2))**2
                                +(z_k[i,j]-((z[i,j,c-1]+z[i,j,c])/2))**2)
                    area_plusHalf[i,j,c] = ((1/2)*(z_k[i,j]*(r[i,j,c+1]-r[i,j,c])+ 
                                 z[i,j,c+1]*(r[i,j,c]-r_k[i,j])+z[i,j,c]*(r_k[i,j]-r[i,j,c+1])))
                    area_minusHalf[i,j,c] = (-(1/2)*(z_k[i,j]*(r[i,j,c-1]-r[i,j,c])+ 
                                 z[i,j,c-1]*(r[i,j,c]-r_k[i,j])+z[i,j,c]*(r_k[i,j]-r[i,j,c-1])))
                    
    #matrices    
    Lr_cplus = np.zeros((Ni,Nj,J,J))
    Lr_cminus = np.zeros((Ni,Nj,J,J))
    Lz_cplus = np.zeros((Ni,Nj,J,J))
    Lz_cminus = np.zeros((Ni,Nj,J,J))
    T_mat = np.zeros((Ni,Nj,J,J))
    R_mat = np.zeros((Ni,Nj,J,J))
    Kr_mat = np.zeros((Ni,Nj,J,J))
    Kz_mat = np.zeros((Ni,Nj,J,J))                   
    
    for j in range(0,Nj):
        for i in range(0,Ni):
            Lr_cplus[i,j][0,0] = ((1/12)*np.sin(theta_half[j])*l_c_plus[i,j,0]*(5*r[i,j,0]+
                 4*r_c_plusHalf[i,j,0]))
            Lz_cplus[i,j][0,0] = (-(1/12)*np.cos(theta_half[j])*l_c_plus[i,j,0]*(5*r[i,j,0]+
                 4*r_c_plusHalf[i,j,0]))
            Lr_cminus[i,j][0,0] = (-(1/12)*np.cos(theta[j])*l_c_minus[i,j,0]*(5*r[i,j,0]+
                 4*r_c_minusHalf[i,j,0]))
            Lz_cminus[i,j][0,0] = (-(1/12)*np.sin(theta[j])*l_c_minus[i,j,0]*(5*r[i,j,0]+
                 4*r_c_minusHalf[i,j,0]))
            
            Lr_cplus[i,j][1,1] = ((1/12)*np.cos(theta[j])*l_c_plus[i,j,1]*(5*r[i,j,1]+
                 4*r_c_plusHalf[i,j,1]))
            Lz_cplus[i,j][1,1] = ((1/12)*np.sin(theta[j])*l_c_plus[i,j,1]*(5*r[i,j,1]+
                 4*r_c_plusHalf[i,j,1]))
            Lr_cminus[i,j][1,1] = ((1/12)*np.sin(theta_half[j])*l_c_minus[i,j,1]*(5*r[i,j,1]+
                 4*r_c_minusHalf[i,j,1]))
            Lz_cminus[i,j][1,1] = (-(1/12)*np.cos(theta_half[j])*l_c_minus[i,j,1]*(5*r[i,j,1]+
                 4*r_c_minusHalf[i,j,1]))
            
            Lr_cplus[i,j][2,2] = (-(1/12)*np.sin(theta_half[j+1])*l_c_plus[i,j,2]*(5*r[i,j,2]+
                 4*r_c_plusHalf[i,j,2]))
            Lz_cplus[i,j][2,2] = ((1/12)*np.cos(theta_half[j+1])*l_c_plus[i,j,2]*(5*r[i,j,2]+
                 4*r_c_plusHalf[i,j,2]))
            Lr_cminus[i,j][2,2] = ((1/12)*np.cos(theta[j])*l_c_minus[i,j,2]*(5*r[i,j,2]+
                 4*r_c_minusHalf[i,j,2]))
            Lz_cminus[i,j][2,2] = ((1/12)*np.sin(theta[j])*l_c_minus[i,j,2]*(5*r[i,j,2]+
                 4*r_c_minusHalf[i,j,2]))
            
            Lr_cplus[i,j][3,3] = (-(1/12)*np.cos(theta[j])*l_c_plus[i,j,3]*(5*r[i,j,3]+
                 4*r_c_plusHalf[i,j,3]))
            Lz_cplus[i,j][3,3] = (-(1/12)*np.sin(theta[j])*l_c_plus[i,j,3]*(5*r[i,j,3]+
                 4*r_c_plusHalf[i,j,3]))
            Lr_cminus[i,j][3,3] = (-(1/12)*np.sin(theta_half[j+1])*l_c_minus[i,j,3]*(5*r[i,j,3]+
                 4*r_c_minusHalf[i,j,3]))
            Lz_cminus[i,j][3,3] = ((1/12)*np.cos(theta_half[j+1])*l_c_minus[i,j,3]*(5*r[i,j,3]+
                 4*r_c_minusHalf[i,j,3]))
            
            Lr_cplus[i,j][0,1] = ((1/12)*np.sin(theta_half[j])*l_c_plus[i,j,0]*(r[i,j,0]+2*r_c_plusHalf[i,j,0]))
            Lz_cplus[i,j][0,1] = (-(1/12)*np.cos(theta_half[j])*l_c_plus[i,j,0]*(r[i,j,0]+2*r_c_plusHalf[i,j,0]))
            
            Lr_cplus[i,j][1,2] = ((1/12)*np.cos(theta[j])*l_c_plus[i,j,1]*(r[i,j,1]+2*r_c_plusHalf[i,j,1]))
            Lz_cplus[i,j][1,2] = ((1/12)*np.sin(theta[j])*l_c_plus[i,j,1]*(r[i,j,1]+2*r_c_plusHalf[i,j,1]))
            
            Lr_cplus[i,j][2,3] = -(1/12)*np.sin(theta_half[j+1])*l_c_plus[i,j,2]*(r[i,j,2]+2*r_c_plusHalf[i,j,2])
            Lz_cplus[i,j][2,3] = (1/12)*np.cos(theta_half[j+1])*l_c_plus[i,j,2]*(r[i,j,2]+2*r_c_plusHalf[i,j,2])
            
            Lr_cplus[i,j][3,0] = -(1/12)*np.cos(theta[j])*l_c_plus[i,j,3]*(r[i,j,3]+2*r_c_plusHalf[i,j,3])
            Lz_cplus[i,j][3,0] = (-(1/12)*np.sin(theta[j])*l_c_plus[i,j,3]*(r[i,j,3]+2*r_c_plusHalf[i,j,3]))
            
            Lr_cminus[i,j][0,3] = -(1/12)*np.cos(theta[j])*l_c_minus[i,j,0]*(r[i,j,0]+2*r_c_minusHalf[i,j,0])
            Lz_cminus[i,j][0,3] = (-(1/12)*np.sin(theta[j])*l_c_minus[i,j,0]*(r[i,j,0]+2*r_c_minusHalf[i,j,0]))
            
            Lr_cminus[i,j][1,0] = (1/12)*np.sin(theta_half[j])*l_c_minus[i,j,1]*(r[i,j,1]+2*r_c_minusHalf[i,j,1])
            Lz_cminus[i,j][1,0] = (-(1/12)*np.cos(theta_half[j])*l_c_minus[i,j,1]*(r[i,j,1]+2*r_c_minusHalf[i,j,1]))
            
            Lr_cminus[i,j][2,1] = (1/12)*np.cos(theta[j])*l_c_minus[i,j,2]*(r[i,j,2]+2*r_c_minusHalf[i,j,2])
            Lz_cminus[i,j][2,1] = ((1/12)*np.sin(theta[j])*l_c_minus[i,j,2]*(r[i,j,2]+2*r_c_minusHalf[i,j,2]))
            
            Lr_cminus[i,j][3,2] = -(1/12)*np.sin(theta_half[j+1])*l_c_minus[i,j,3]*(r[i,j,3]+2*r_c_minusHalf[i,j,3])
            Lz_cminus[i,j][3,2] = ((1/12)*np.cos(theta_half[j+1])*l_c_minus[i,j,3]*(r[i,j,3]+2*r_c_minusHalf[i,j,3]))
            
            
            Kr_mat[i,j][0,0] = ((1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,0]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,0])
                -(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,0]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,0]))
            Kr_mat[i,j][1,1] = (-(1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,1]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,1])
                -(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,1]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,1]))
            Kr_mat[i,j][2,2] = (-(1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,2]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,2])
                +(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,2]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,2]))
            Kr_mat[i,j][3,3] = ((1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,3]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,3])
                +(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,3]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,3]))
            Kr_mat[i,j][0,1] = ((1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,0]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,0])
                -(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,0]*(2*r_k[i,j]+r_c_minusHalf[i,j,0]))
            Kr_mat[i,j][1,2] = (-(1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,1]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,1])
                -(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,1]*(2*r_k[i,j]+r_c_minusHalf[i,j,1]))
            Kr_mat[i,j][2,3] = (-(1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,2]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,2])
                +(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,2]*(2*r_k[i,j]+r_c_minusHalf[i,j,2]))
            Kr_mat[i,j][3,0] = ((1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,3]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,3])
                +(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,3]*(2*r_k[i,j]+r_c_minusHalf[i,j,3]))
            Kr_mat[i,j][0,2] = ((1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,0]*(2*r_k[i,j]+r_c_plusHalf[i,j,0])
                -(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,0]*(2*r_k[i,j]+r_c_minusHalf[i,j,0]))
            Kr_mat[i,j][1,3] = (-(1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,1]*(2*r_k[i,j]+r_c_plusHalf[i,j,1])
                -(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,1]*(2*r_k[i,j]+r_c_minusHalf[i,j,1]))
            Kr_mat[i,j][2,0] = (-(1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,2]*(2*r_k[i,j]+r_c_plusHalf[i,j,2])
                +(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,2]*(2*r_k[i,j]+r_c_minusHalf[i,j,2]))
            Kr_mat[i,j][3,1] = ((1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,3]*(2*r_k[i,j]+r_c_plusHalf[i,j,3])
                +(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,3]*(2*r_k[i,j]+r_c_minusHalf[i,j,3]))
            Kr_mat[i,j][0,3] = ((1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,0]*(2*r_k[i,j]+r_c_plusHalf[i,j,0])
                -(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,0]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,0]))
            Kr_mat[i,j][1,0] = (-(1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,1]*(2*r_k[i,j]+r_c_plusHalf[i,j,1])
                -(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,1]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,1]))
            Kr_mat[i,j][2,1] = (-(1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,2]*(2*r_k[i,j]+r_c_plusHalf[i,j,2])
                +(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,2]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,2]))
            Kr_mat[i,j][3,2] = ((1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,3]*(2*r_k[i,j]+r_c_plusHalf[i,j,3])
                +(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,3]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,3]))
            
            Kz_mat[i,j][0,0] = ((1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,0]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,0])
                +(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,0]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,0]))
            Kz_mat[i,j][1,1] = ((1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,1]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,1])
                -(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,1]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,1]))
            Kz_mat[i,j][2,2] = (-(1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,2]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,2])
                -(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,2]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,2]))
            Kz_mat[i,j][3,3] = (-(1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,3]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,3])
                +(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,3]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,3]))
            Kz_mat[i,j][0,1] = ((1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,0]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,0])
                +(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,0]*(2*r_k[i,j]+r_c_minusHalf[i,j,0]))
            Kz_mat[i,j][1,2] = ((1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,1]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,1])
                -(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,1]*(2*r_k[i,j]+r_c_minusHalf[i,j,1]))
            Kz_mat[i,j][2,3] = (-(1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,2]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,2])
                -(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,2]*(2*r_k[i,j]+r_c_minusHalf[i,j,2]))
            Kz_mat[i,j][3,0] = (-(1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,3]*(4*r_k[i,j]+5*r_c_plusHalf[i,j,3])
                +(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,3]*(2*r_k[i,j]+r_c_minusHalf[i,j,3]))
            Kz_mat[i,j][0,2] = ((1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,0]*(2*r_k[i,j]+r_c_plusHalf[i,j,0])
                +(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,0]*(2*r_k[i,j]+r_c_minusHalf[i,j,0]))
            Kz_mat[i,j][1,3] = ((1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,1]*(2*r_k[i,j]+r_c_plusHalf[i,j,1])
                -(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,1]*(2*r_k[i,j]+r_c_minusHalf[i,j,1]))
            Kz_mat[i,j][2,0] = (-(1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,2]*(2*r_k[i,j]+r_c_plusHalf[i,j,2])
                -(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,2]*(2*r_k[i,j]+r_c_minusHalf[i,j,2]))
            Kz_mat[i,j][3,1] = (-(1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,3]*(2*r_k[i,j]+r_c_plusHalf[i,j,3])
                +(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,3]*(2*r_k[i,j]+r_c_minusHalf[i,j,3]))
            Kz_mat[i,j][0,3] = ((1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,0]*(2*r_k[i,j]+r_c_plusHalf[i,j,0])
                +(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,0]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,0]))
            Kz_mat[i,j][1,0] = ((1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,1]*(2*r_k[i,j]+r_c_plusHalf[i,j,1])
                -(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,1]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,1]))
            Kz_mat[i,j][2,1] = (-(1/24)*np.sin(theta[j])*l_c_plusHalf[i,j,2]*(2*r_k[i,j]+r_c_plusHalf[i,j,2])
                -(1/24)*np.cos(theta[j])*l_c_minusHalf[i,j,2]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,2]))
            Kz_mat[i,j][3,2] = (-(1/24)*np.cos(theta[j])*l_c_plusHalf[i,j,3]*(2*r_k[i,j]+r_c_plusHalf[i,j,3])
                +(1/24)*np.sin(theta[j])*l_c_minusHalf[i,j,3]*(4*r_k[i,j]+5*r_c_minusHalf[i,j,3]))
            
            R_mat[i,j][0,0] = (7/24)*(area_plusHalf[i,j,0]+area_minusHalf[i,j,0])
            R_mat[i,j][1,1] = (7/24)*(area_plusHalf[i,j,1]+area_minusHalf[i,j,1])
            R_mat[i,j][2,2] = (7/24)*(area_plusHalf[i,j,2]+area_minusHalf[i,j,2])
            R_mat[i,j][3,3] = (7/24)*(area_plusHalf[i,j,3]+area_minusHalf[i,j,3])
            R_mat[i,j][0,1] = (1/24)*((3)*area_plusHalf[i,j,0]+area_minusHalf[i,j,0])
            R_mat[i,j][1,2] = (1/24)*((3)*area_plusHalf[i,j,1]+area_minusHalf[i,j,1])
            R_mat[i,j][2,3] = (1/24)*((3)*area_plusHalf[i,j,2]+area_minusHalf[i,j,2])
            R_mat[i,j][3,0] = (1/24)*((3)*area_plusHalf[i,j,3]+area_minusHalf[i,j,3])
            R_mat[i,j][0,2] = (1/24)*(area_plusHalf[i,j,0]+area_minusHalf[i,j,0])
            R_mat[i,j][1,3] = (1/24)*(area_plusHalf[i,j,1]+area_minusHalf[i,j,1])
            R_mat[i,j][2,0] = (1/24)*(area_plusHalf[i,j,2]+area_minusHalf[i,j,2])
            R_mat[i,j][3,1] = (1/24)*(area_plusHalf[i,j,3]+area_minusHalf[i,j,3])
            R_mat[i,j][0,3] = (1/24)*(area_plusHalf[i,j,0]+(3)*area_minusHalf[i,j,0])
            R_mat[i,j][1,0] = (1/24)*(area_plusHalf[i,j,1]+(3)*area_minusHalf[i,j,1])
            R_mat[i,j][2,1] = (1/24)*(area_plusHalf[i,j,2]+(3)*area_minusHalf[i,j,2])
            R_mat[i,j][3,2] = (1/24)*(area_plusHalf[i,j,3]+(3)*area_minusHalf[i,j,3])
            
            T_mat[i,j][0,0] = ((area_plusHalf[i,j,0]/96)*(11*r[i,j,0]+8*r_k[i,j]+9*r_c_plusHalf[i,j,0])
                +(area_minusHalf[i,j,0]/96)*(11*r[i,j,0]+8*r_k[i,j]+9*r_c_minusHalf[i,j,0]))
            T_mat[i,j][1,1] = ((area_plusHalf[i,j,1]/96)*(11*r[i,j,1]+8*r_k[i,j]+9*r_c_plusHalf[i,j,1])
                +(area_minusHalf[i,j,1]/96)*(11*r[i,j,1]+8*r_k[i,j]+9*r_c_minusHalf[i,j,1]))
            T_mat[i,j][2,2] = ((area_plusHalf[i,j,2]/96)*(11*r[i,j,2]+8*r_k[i,j]+9*r_c_plusHalf[i,j,2])
                +(area_minusHalf[i,j,2]/96)*(11*r[i,j,2]+8*r_k[i,j]+9*r_c_minusHalf[i,j,2]))
            T_mat[i,j][3,3] = ((area_plusHalf[i,j,3]/96)*(11*r[i,j,3]+8*r_k[i,j]+9*r_c_plusHalf[i,j,3])
                +(area_minusHalf[i,j,3]/96)*(11*r[i,j,3]+8*r_k[i,j]+9*r_c_minusHalf[i,j,3]))
            T_mat[i,j][0,1] = ((area_plusHalf[i,j,0]/96)*(3*r[i,j,0]+4*r_k[i,j]+5*r_c_plusHalf[i,j,0])
                + (area_minusHalf[i,j,0]/96)*(1*r[i,j,0]+2*r_k[i,j]+1*r_c_minusHalf[i,j,0]))
            T_mat[i,j][1,2] = ((area_plusHalf[i,j,1]/96)*(3*r[i,j,1]+4*r_k[i,j]+5*r_c_plusHalf[i,j,1])
                +(area_minusHalf[i,j,1]/96)*(1*r[i,j,1]+2*r_k[i,j]+1*r_c_minusHalf[i,j,1]))
            T_mat[i,j][2,3] = ((area_plusHalf[i,j,2]/96)*(3*r[i,j,2]+4*r_k[i,j]+5*r_c_plusHalf[i,j,2])
                +(area_minusHalf[i,j,2]/96)*(1*r[i,j,2]+2*r_k[i,j]+1*r_c_minusHalf[i,j,2]))
            T_mat[i,j][3,0] = ((area_plusHalf[i,j,3]/96)*(3*r[i,j,3]+4*r_k[i,j]+5*r_c_plusHalf[i,j,3])
                +(area_minusHalf[i,j,3]/96)*(1*r[i,j,3]+2*r_k[i,j]+1*r_c_minusHalf[i,j,3]))
            T_mat[i,j][0,3] = ((area_plusHalf[i,j,0]/96)*(1*r[i,j,0]+2*r_k[i,j]+1*r_c_plusHalf[i,j,0])
                +(area_minusHalf[i,j,0]/96)*(3*r[i,j,0]+4*r_k[i,j]+5*r_c_minusHalf[i,j,0]))
            T_mat[i,j][1,0] = ((area_plusHalf[i,j,1]/96)*(1*r[i,j,1]+2*r_k[i,j]+1*r_c_plusHalf[i,j,1])
                +(area_minusHalf[i,j,1]/96)*(3*r[i,j,1]+4*r_k[i,j]+5*r_c_minusHalf[i,j,1]))
            T_mat[i,j][2,1] = ((area_plusHalf[i,j,2]/96)*(1*r[i,j,2]+2*r_k[i,j]+1*r_c_plusHalf[i,j,2])
                +(area_minusHalf[i,j,2]/96)*(3*r[i,j,2]+4*r_k[i,j]+5*r_c_minusHalf[i,j,2]))
            T_mat[i,j][3,2] = ((area_plusHalf[i,j,3]/96)*(1*r[i,j,3]+2*r_k[i,j]+1*r_c_plusHalf[i,j,3])
                +(area_minusHalf[i,j,3]/96)*(3*r[i,j,3]+4*r_k[i,j]+5*r_c_minusHalf[i,j,3]))
            T_mat[i,j][0,2] = ((area_plusHalf[i,j,0]/96)*(r[i,j,0]+2*r_k[i,j]+r_c_plusHalf[i,j,0])
                +(area_minusHalf[i,j,0]/96)*(r[i,j,0]+2*r_k[i,j]+r_c_minusHalf[i,j,0]))
            T_mat[i,j][1,3] = ((area_plusHalf[i,j,1]/96)*(r[i,j,1]+2*r_k[i,j]+r_c_plusHalf[i,j,1])
                +(area_minusHalf[i,j,1]/96)*(r[i,j,1]+2*r_k[i,j]+r_c_minusHalf[i,j,1]))
            T_mat[i,j][2,0] = ((area_plusHalf[i,j,2]/96)*(r[i,j,2]+2*r_k[i,j]+r_c_plusHalf[i,j,2])
                +(area_minusHalf[i,j,2]/96)*(r[i,j,2]+2*r_k[i,j]+r_c_minusHalf[i,j,2]))
            T_mat[i,j][3,1] = ((area_plusHalf[i,j,3]/96)*(r[i,j,3]+2*r_k[i,j]+r_c_plusHalf[i,j,3])
                +(area_minusHalf[i,j,3]/96)*(r[i,j,3]+2*r_k[i,j]+r_c_minusHalf[i,j,3]))

    return T_mat,Kr_mat,Kz_mat,R_mat,Lr_cminus,Lr_cplus,Lz_cminus,Lz_cplus,theta,theta_half,r,z,r_k,z_k

def G(Ni,Nj,Nazi,Np,R,J):
    # i = radial 
    # j = sector 
    # rho = distance from center
    # theta = angle 
    # R : radius of sphere
    # Nj : number of sectors
    # Ni : number of radial sections
    # (r,z) vertices
    # J = number of corners in a cell
    
    #print("G-PWLD")
    #print("Given")
    #print("Q = ",Q," , sigma_t = ",Sig_t," , sigma_s = ",Sig_s," , psi_inc = ",psi_inc)
    #print("N_r = ",Ni," , N_z = ",Nj," , N_p = ",Np," , N_azi = ",Nazi,"\n")
    rho = np.zeros(Ni)
    rho_half = np.zeros(Ni+1)
    theta = np.zeros(Nj)
    theta_half = np.zeros(Nj+1)
    r = np.zeros((Ni,Nj,J))
    z = np.zeros((Ni,Nj,J))
    r_k = np.zeros((Ni,Nj))
    z_k = np.zeros((Ni,Nj))
    
    #determining the vertices
    for j in range(0,Nj):
        for i in range(0,Ni):
            #rho_half[i] = (i/Ni)*R
            rho_half[i+1] = ((i+1)/Ni)*R
            rho_half[0] = (10**(-6))*rho_half[i+1]
            rho[i] = (rho_half[i]+rho_half[i+1])/2
            theta_half[j] = ((j/Nj)*np.pi)-(np.pi/2)
            theta_half[j+1] = (((j+1)/Nj)*np.pi)-(np.pi/2)
            theta[j] = (theta_half[j]+theta_half[j+1])/2
            r[i,j,0] = rho_half[i]*np.cos(theta_half[j])
            r[i,j,1] = rho_half[i+1]*np.cos(theta_half[j])
            r[i,j,2] = rho_half[i+1]*np.cos(theta_half[j+1])
            r[i,j,3] = rho_half[i]*np.cos(theta_half[j+1])
            z[i,j,0] = rho_half[i]*np.sin(theta_half[j])
            z[i,j,1] = rho_half[i+1]*np.sin(theta_half[j])
            z[i,j,2] = rho_half[i+1]*np.sin(theta_half[j+1])
            z[i,j,3] = rho_half[i]*np.sin(theta_half[j+1])
            r_k[i,j] = (1/4)*(r[i,j,0]+r[i,j,1]+r[i,j,2]+r[i,j,3])
            z_k[i,j] = (1/4)*(z[i,j,0]+z[i,j,1]+z[i,j,2]+z[i,j,3])
            
    #
    r_c_plusHalf = np.zeros((Ni,Nj,J))
    r_c_minusHalf = np.zeros((Ni,Nj,J))
    l_c_plus = np.zeros((Ni,Nj,J))
    l_c_minus = np.zeros((Ni,Nj,J))
    l_c_plusHalf = np.zeros((Ni,Nj,J))
    l_c_minusHalf = np.zeros((Ni,Nj,J))
    area_plusHalf = np.zeros((Ni,Nj,J))
    area_minusHalf = np.zeros((Ni,Nj,J))
    l_k_c = np.zeros((Ni,Nj,J))
    
    for j in range(0,Nj):
        for i in range(0,Ni):
            for c in range(0,J):
                if c==0:
                    r_c_plusHalf[i,j,c] = (r[i,j,c]+r[i,j,c+1])/2
                    r_c_minusHalf[i,j,c] = (r[i,j,c]+r[i,j,J-1])/2
                    l_c_plus[i,j,c] = np.sqrt((r[i,j,c+1]-r[i,j,c])**2+(z[i,j,c+1]-z[i,j,c])**2)/2
                    l_c_minus[i,j,c] = np.sqrt((r[i,j,J-1]-r[i,j,c])**2+(z[i,j,J-1]-z[i,j,c])**2)/2
                    l_c_plusHalf[i,j,c] = np.sqrt((r_k[i,j]-((r[i,j,c+1]+r[i,j,c])/2))**2
                                +(z_k[i,j]-((z[i,j,c+1]+z[i,j,c])/2))**2)
                    l_c_minusHalf[i,j,c] = np.sqrt((r_k[i,j]-((r[i,j,J-1]+r[i,j,c])/2))**2
                                +(z_k[i,j]-((z[i,j,J-1]+z[i,j,c])/2))**2)
                    area_plusHalf[i,j,c] = ((1/2)*(z_k[i,j]*(r[i,j,c+1]-r[i,j,c])+ 
                                 z[i,j,c+1]*(r[i,j,c]-r_k[i,j])+z[i,j,c]*(r_k[i,j]-r[i,j,c+1])))
                    area_minusHalf[i,j,c] = (-(1/2)*(z_k[i,j]*(r[i,j,J-1]-r[i,j,c])+ 
                                 z[i,j,J-1]*(r[i,j,c]-r_k[i,j])+z[i,j,c]*(r_k[i,j]-r[i,j,J-1])))
                    l_k_c[i,j,c] = np.sqrt((r_k[i,j]-r[i,j,c])**2+(z_k[i,j]-z[i,j,c])**2)
                    
                elif c==J-1:
                    r_c_plusHalf[i,j,c] = (r[i,j,c]+r[i,j,0])/2
                    r_c_minusHalf[i,j,c] = (r[i,j,c]+r[i,j,c-1])/2
                    l_c_plus[i,j,c] = np.sqrt((r[i,j,0]-r[i,j,c])**2+(z[i,j,0]-z[i,j,c])**2)/2
                    l_c_minus[i,j,c] = np.sqrt((r[i,j,c-1]-r[i,j,c])**2+(z[i,j,c-1]-z[i,j,c])**2)/2
                    l_c_plusHalf[i,j,c] = np.sqrt((r_k[i,j]-((r[i,j,0]+r[i,j,c])/2))**2
                                +(z_k[i,j]-((z[i,j,0]+z[i,j,c])/2))**2)
                    l_c_minusHalf[i,j,c] = np.sqrt((r_k[i,j]-((r[i,j,c-1]+r[i,j,c])/2))**2
                                +(z_k[i,j]-((z[i,j,c-1]+z[i,j,c])/2))**2)
                    area_plusHalf[i,j,c] = ((1/2)*(z_k[i,j]*(r[i,j,0]-r[i,j,c])+ 
                                 z[i,j,0]*(r[i,j,c]-r_k[i,j])+z[i,j,c]*(r_k[i,j]-r[i,j,0])))
                    area_minusHalf[i,j,c] = (-(1/2)*(z_k[i,j]*(r[i,j,c-1]-r[i,j,c])+ 
                                 z[i,j,c-1]*(r[i,j,c]-r_k[i,j])+z[i,j,c]*(r_k[i,j]-r[i,j,c-1])))
                    l_k_c[i,j,c] = np.sqrt((r_k[i,j]-r[i,j,c])**2+(z_k[i,j]-z[i,j,c])**2)
                else:
                    r_c_plusHalf[i,j,c] = (r[i,j,c]+r[i,j,c+1])/2
                    r_c_minusHalf[i,j,c] = (r[i,j,c]+r[i,j,c-1])/2
                    l_c_plus[i,j,c] = np.sqrt((r[i,j,c+1]-r[i,j,c])**2+(z[i,j,c+1]-z[i,j,c])**2)/2
                    l_c_minus[i,j,c] = np.sqrt((r[i,j,c-1]-r[i,j,c])**2+(z[i,j,c-1]-z[i,j,c])**2)/2
                    l_c_plusHalf[i,j,c] = np.sqrt((r_k[i,j]-((r[i,j,c+1]+r[i,j,c])/2))**2
                                +(z_k[i,j]-((z[i,j,c+1]+z[i,j,c])/2))**2)
                    l_c_minusHalf[i,j,c] = np.sqrt((r_k[i,j]-((r[i,j,c-1]+r[i,j,c])/2))**2
                                +(z_k[i,j]-((z[i,j,c-1]+z[i,j,c])/2))**2)
                    area_plusHalf[i,j,c] = ((1/2)*(z_k[i,j]*(r[i,j,c+1]-r[i,j,c])+ 
                                 z[i,j,c+1]*(r[i,j,c]-r_k[i,j])+z[i,j,c]*(r_k[i,j]-r[i,j,c+1])))
                    area_minusHalf[i,j,c] = (-(1/2)*(z_k[i,j]*(r[i,j,c-1]-r[i,j,c])+ 
                                 z[i,j,c-1]*(r[i,j,c]-r_k[i,j])+z[i,j,c]*(r_k[i,j]-r[i,j,c-1])))
                    l_k_c[i,j,c] = np.sqrt((r_k[i,j]-r[i,j,c])**2+(z_k[i,j]-z[i,j,c])**2)
    
    A_s = area_plusHalf
    del_L_plus = 2*l_c_plus
    del_L_minus = 2*l_c_minus
    l_e_r = np.zeros((Ni,Nj,J))
    l_e_z = np.zeros((Ni,Nj,J))
    dw_dr = np.zeros((Ni,Nj,J,J))
    dw_dz = np.zeros((Ni,Nj,J,J))
    for i in range(Ni):
        for j in range(Nj):
            l_e_r[i,j,0] = -l_c_minus[i,j,0]*(-np.cos(theta[j]))-l_c_minusHalf[i,j,0]*(-np.sin(theta[j]))
            l_e_r[i,j,1] = -l_c_minus[i,j,1]*(np.sin(theta_half[j]))-l_c_minusHalf[i,j,1]*(-np.cos(theta[j]))
            l_e_r[i,j,2] = -l_c_minus[i,j,2]*(np.cos(theta[j]))-l_c_minusHalf[i,j,2]*(np.sin(theta[j]))
            l_e_r[i,j,3] = -l_c_minus[i,j,3]*(-np.sin(theta_half[j+1]))-l_c_minusHalf[i,j,3]*(np.cos(theta[j]))
            
            l_e_z[i,j,0] = -l_c_minus[i,j,0]*(-np.sin(theta[j]))-l_c_minusHalf[i,j,0]*(np.cos(theta[j]))
            l_e_z[i,j,1] = -l_c_minus[i,j,1]*(-np.cos(theta_half[j]))-l_c_minusHalf[i,j,1]*(-np.sin(theta[j]))
            l_e_z[i,j,2] = -l_c_minus[i,j,2]*(np.sin(theta[j]))-l_c_minusHalf[i,j,2]*(-np.cos(theta[j]))
            l_e_z[i,j,3] = -l_c_minus[i,j,3]*(np.cos(theta_half[j+1]))-l_c_minusHalf[i,j,3]*(np.sin(theta[j]))
            
            dw_dr[i,j][0,0] = (-0.5*l_e_r[i,j,1]-0.125*del_L_plus[i,j,0]*np.sin(theta_half[j]))
            dw_dr[i,j][0,1] = (-0.125*del_L_plus[i,j,1]*np.cos(theta[j]))
            dw_dr[i,j][0,2] = (-0.125*del_L_plus[i,j,2]*(-np.sin(theta_half[j+1])))
            dw_dr[i,j][0,3] = (0.5*l_e_r[i,j,3]-0.125*del_L_minus[i,j,0]*(-np.cos(theta[j])))
            dw_dr[i,j][1,1] = (-0.5*l_e_r[i,j,2]-0.125*del_L_plus[i,j,1]*np.cos(theta[j]))
            dw_dr[i,j][1,2] = (-0.125*del_L_plus[i,j,2]*(-np.sin(theta_half[j+1])))
            dw_dr[i,j][1,3] = (-0.125*del_L_plus[i,j,3]*(-np.cos(theta[j])))
            dw_dr[i,j][1,0] = (0.5*l_e_r[i,j,0]-0.125*del_L_minus[i,j,1]*np.sin(theta_half[j]))
            dw_dr[i,j][2,2] = (-0.5*l_e_r[i,j,3]-0.125*del_L_plus[i,j,2]*(-np.sin(theta_half[j+1])))
            dw_dr[i,j][2,3] = (-0.125*del_L_plus[i,j,3]*(-np.cos(theta[j])))
            dw_dr[i,j][2,0] = (-0.125*del_L_plus[i,j,0]*(np.sin(theta_half[j])))
            dw_dr[i,j][2,1] = (0.5*l_e_r[i,j,1]-0.125*del_L_minus[i,j,2]*np.cos(theta[j]))
            dw_dr[i,j][3,3] = (-0.5*l_e_r[i,j,0]-0.125*del_L_plus[i,j,3]*(-np.cos(theta[j])))
            dw_dr[i,j][3,0] = (-0.125*del_L_plus[i,j,0]*(np.sin(theta_half[j])))
            dw_dr[i,j][3,1] = (-0.125*del_L_plus[i,j,1]*np.cos(theta[j]))
            dw_dr[i,j][3,2] = (0.5*l_e_r[i,j,2]-0.125*del_L_minus[i,j,3]*(-np.sin(theta_half[j+1])))
            
            dw_dz[i,j][0,0] = (-0.5*l_e_z[i,j,1]-0.125*del_L_plus[i,j,0]*(-np.cos(theta_half[j])))
            dw_dz[i,j][0,1] = (-0.125*del_L_plus[i,j,1]*np.sin(theta[j]))
            dw_dz[i,j][0,2] = (-0.125*del_L_plus[i,j,2]*np.cos(theta_half[j+1]))
            dw_dz[i,j][0,3] = (0.5*l_e_z[i,j,3]-0.125*del_L_minus[i,j,0]*(-np.sin(theta[j])))
            dw_dz[i,j][1,1] = (-0.5*l_e_z[i,j,2]-0.125*del_L_plus[i,j,1]*np.sin(theta[j]))
            dw_dz[i,j][1,2] = (-0.125*del_L_plus[i,j,2]*np.cos(theta_half[j+1]))
            dw_dz[i,j][1,3] = (-0.125*del_L_plus[i,j,3]*(-np.sin(theta[j])))
            dw_dz[i,j][1,0] = (0.5*l_e_z[i,j,0]-0.125*del_L_minus[i,j,1]*(-np.cos(theta_half[j])))
            dw_dz[i,j][2,2] = (-0.5*l_e_z[i,j,3]-0.125*del_L_plus[i,j,2]*np.cos(theta_half[j+1]))
            dw_dz[i,j][2,3] = (-0.125*del_L_plus[i,j,3]*(-np.sin(theta[j])))
            dw_dz[i,j][2,0] = (-0.125*del_L_plus[i,j,0]*(-np.cos(theta_half[j])))
            dw_dz[i,j][2,1] = (0.5*l_e_z[i,j,1]-0.125*del_L_minus[i,j,2]*np.sin(theta[j]))
            dw_dz[i,j][3,3] = (-0.5*l_e_z[i,j,0]-0.125*del_L_plus[i,j,3]*(-np.sin(theta[j])))
            dw_dz[i,j][3,0] = (-0.125*del_L_plus[i,j,0]*(-np.cos(theta_half[j])))
            dw_dz[i,j][3,1] = (-0.125*del_L_plus[i,j,1]*np.sin(theta[j]))
            dw_dz[i,j][3,2] = (0.5*l_e_z[i,j,2]-0.125*del_L_minus[i,j,3]*np.cos(theta_half[j+1]))
            
            
    #matrices    
    Lr_cplus = np.zeros((Ni,Nj,J,J))
    Lr_cminus = np.zeros((Ni,Nj,J,J))
    Lz_cplus = np.zeros((Ni,Nj,J,J))
    Lz_cminus = np.zeros((Ni,Nj,J,J))
    T_mat = np.zeros((Ni,Nj,J,J))
    R_mat = np.zeros((Ni,Nj,J,J))
    Kr_mat = np.zeros((Ni,Nj,J,J))
    Kz_mat = np.zeros((Ni,Nj,J,J))                   
    
    for j in range(0,Nj):
        for i in range(0,Ni):
            Lr_cplus[i,j][0,0] = ((1/12)*np.sin(theta_half[j])*del_L_plus[i,j,0]*(3*r[i,j,0]+r[i,j,1]))
            Lr_cplus[i,j][0,1] = ((1/12)*np.sin(theta_half[j])*del_L_plus[i,j,0]*(r[i,j,0]+r[i,j,1]))
            Lr_cplus[i,j][1,1] = ((1/12)*np.cos(theta[j])*del_L_plus[i,j,1]*(3*r[i,j,1]+r[i,j,2]))
            Lr_cplus[i,j][1,2] = ((1/12)*np.cos(theta[j])*del_L_plus[i,j,1]*(r[i,j,1]+r[i,j,2]))
            Lr_cplus[i,j][2,2] = (-(1/12)*np.sin(theta_half[j+1])*del_L_plus[i,j,2]*(3*r[i,j,2]+r[i,j,3]))
            Lr_cplus[i,j][2,3] = (-(1/12)*np.sin(theta_half[j+1])*del_L_plus[i,j,2]*(r[i,j,2]+r[i,j,3]))
            Lr_cplus[i,j][3,3] = (-(1/12)*np.cos(theta[j])*del_L_plus[i,j,3]*(3*r[i,j,3]+r[i,j,0]))
            Lr_cplus[i,j][3,0] = (-(1/12)*np.cos(theta[j])*del_L_plus[i,j,3]*(r[i,j,3]+r[i,j,0]))
            
            Lr_cminus[i,j][0,0] = (-(1/12)*np.cos(theta[j])*del_L_minus[i,j,0]*(3*r[i,j,0]+r[i,j,3]))
            Lr_cminus[i,j][0,3] = (-(1/12)*np.cos(theta[j])*del_L_minus[i,j,0]*(r[i,j,0]+r[i,j,3]))
            Lr_cminus[i,j][1,1] = ((1/12)*np.sin(theta_half[j])*del_L_minus[i,j,1]*(3*r[i,j,1]+r[i,j,0]))
            Lr_cminus[i,j][1,0] = ((1/12)*np.sin(theta_half[j])*del_L_minus[i,j,1]*(r[i,j,1]+r[i,j,0]))
            Lr_cminus[i,j][2,2] = ((1/12)*np.cos(theta[j])*del_L_minus[i,j,2]*(3*r[i,j,2]+r[i,j,1]))
            Lr_cminus[i,j][2,1] = ((1/12)*np.cos(theta[j])*del_L_minus[i,j,2]*(r[i,j,2]+r[i,j,1]))
            Lr_cminus[i,j][3,3] = (-(1/12)*np.sin(theta_half[j+1])*del_L_minus[i,j,3]*(3*r[i,j,3]+r[i,j,2]))
            Lr_cminus[i,j][3,2] = (-(1/12)*np.sin(theta_half[j+1])*del_L_minus[i,j,3]*(r[i,j,3]+r[i,j,2]))
            
            Lz_cplus[i,j][0,0] = (-(1/12)*np.cos(theta_half[j])*del_L_plus[i,j,0]*(3*r[i,j,0]+r[i,j,1]))
            Lz_cplus[i,j][0,1] = (-(1/12)*np.cos(theta_half[j])*del_L_plus[i,j,0]*(r[i,j,0]+r[i,j,1]))
            Lz_cplus[i,j][1,1] = ((1/12)*np.sin(theta[j])*del_L_plus[i,j,1]*(3*r[i,j,1]+r[i,j,2]))
            Lz_cplus[i,j][1,2] = ((1/12)*np.sin(theta[j])*del_L_plus[i,j,1]*(r[i,j,1]+r[i,j,2]))
            Lz_cplus[i,j][2,2] = ((1/12)*np.cos(theta_half[j+1])*del_L_plus[i,j,2]*(3*r[i,j,2]+r[i,j,3]))
            Lz_cplus[i,j][2,3] = ((1/12)*np.cos(theta_half[j+1])*del_L_plus[i,j,2]*(r[i,j,2]+r[i,j,3]))
            Lz_cplus[i,j][3,3] = (-(1/12)*np.sin(theta[j])*del_L_plus[i,j,3]*(3*r[i,j,3]+r[i,j,0]))
            Lz_cplus[i,j][3,0] = (-(1/12)*np.sin(theta[j])*del_L_plus[i,j,3]*(r[i,j,3]+r[i,j,0]))
            
            Lz_cminus[i,j][0,0] = (-(1/12)*np.sin(theta[j])*del_L_minus[i,j,0]*(3*r[i,j,0]+r[i,j,3]))
            Lz_cminus[i,j][0,3] = (-(1/12)*np.sin(theta[j])*del_L_minus[i,j,0]*(r[i,j,0]+r[i,j,3]))
            Lz_cminus[i,j][1,1] = (-(1/12)*np.cos(theta_half[j])*del_L_minus[i,j,1]*(3*r[i,j,1]+r[i,j,0]))
            Lz_cminus[i,j][1,0] = (-(1/12)*np.cos(theta_half[j])*del_L_minus[i,j,1]*(r[i,j,1]+r[i,j,0]))
            Lz_cminus[i,j][2,2] = ((1/12)*np.sin(theta[j])*del_L_minus[i,j,2]*(3*r[i,j,2]+r[i,j,1]))
            Lz_cminus[i,j][2,1] = ((1/12)*np.sin(theta[j])*del_L_minus[i,j,2]*(r[i,j,2]+r[i,j,1]))
            Lz_cminus[i,j][3,3] = ((1/12)*np.cos(theta_half[j+1])*del_L_minus[i,j,3]*(3*r[i,j,3]+r[i,j,2]))
            Lz_cminus[i,j][3,2] = ((1/12)*np.cos(theta_half[j+1])*del_L_minus[i,j,3]*(r[i,j,3]+r[i,j,2]))
            
            T_mat[i,j][0,0] = ((1/160)*A_s[i,j,0]*(19*r[i,j,0]+9*r_k[i,j]+7*r[i,j,1])
                                + (1/480)*A_s[i,j,1]*(1*r[i,j,1]+3*r_k[i,j]+1*r[i,j,2])
                                + (1/480)*A_s[i,j,2]*(1*r[i,j,2]+3*r_k[i,j]+1*r[i,j,3])
                                + (1/160)*A_s[i,j,3]*(19*r[i,j,0]+9*r_k[i,j]+7*r[i,j,3]))
            
            T_mat[i,j][1,1] = ((1/160)*A_s[i,j,1]*(19*r[i,j,1]+9*r_k[i,j]+7*r[i,j,2])
                                + (1/480)*A_s[i,j,2]*(1*r[i,j,2]+3*r_k[i,j]+1*r[i,j,3])
                                + (1/480)*A_s[i,j,3]*(1*r[i,j,3]+3*r_k[i,j]+1*r[i,j,0])
                                + (1/160)*A_s[i,j,0]*(19*r[i,j,1]+9*r_k[i,j]+7*r[i,j,0]))
            
            T_mat[i,j][2,2] = ((1/160)*A_s[i,j,2]*(19*r[i,j,2]+9*r_k[i,j]+7*r[i,j,3])
                                + (1/480)*A_s[i,j,0]*(1*r[i,j,0]+3*r_k[i,j]+1*r[i,j,1])
                                + (1/480)*A_s[i,j,3]*(1*r[i,j,3]+3*r_k[i,j]+1*r[i,j,0])
                                + (1/160)*A_s[i,j,1]*(19*r[i,j,2]+9*r_k[i,j]+7*r[i,j,1]))
            
            T_mat[i,j][3,3] = ((1/160)*A_s[i,j,3]*(19*r[i,j,3]+9*r_k[i,j]+7*r[i,j,0])
                                + (1/480)*A_s[i,j,1]*(1*r[i,j,1]+3*r_k[i,j]+1*r[i,j,2])
                                + (1/480)*A_s[i,j,0]*(1*r[i,j,0]+3*r_k[i,j]+1*r[i,j,1])
                                + (1/160)*A_s[i,j,2]*(19*r[i,j,3]+9*r_k[i,j]+7*r[i,j,2]))
            
            T_mat[i,j][0,1] = ((1/480)*A_s[i,j,0]*(23*r[i,j,0]+19*r_k[i,j]+23*r[i,j,1])
                                + (1/480)*A_s[i,j,1]*(5*r[i,j,1]+7*r_k[i,j]+3*r[i,j,2])
                                + (1/480)*A_s[i,j,2]*(1*r[i,j,2]+3*r_k[i,j]+1*r[i,j,3])
                                + (1/480)*A_s[i,j,3]*(5*r[i,j,0]+7*r_k[i,j]+3*r[i,j,3]))
            
            T_mat[i,j][0,2] = ((1/480)*A_s[i,j,0]*(5*r[i,j,0]+7*r_k[i,j]+3*r[i,j,1])
                                + (1/480)*A_s[i,j,1]*(3*r[i,j,1]+7*r_k[i,j]+5*r[i,j,2])
                                + (1/480)*A_s[i,j,2]*(5*r[i,j,2]+7*r_k[i,j]+3*r[i,j,3])
                                + (1/480)*A_s[i,j,3]*(5*r[i,j,0]+7*r_k[i,j]+3*r[i,j,3]))
            
            T_mat[i,j][0,3] = ((1/480)*A_s[i,j,0]*(5*r[i,j,0]+7*r_k[i,j]+3*r[i,j,1])
                                + (1/480)*A_s[i,j,1]*(1*r[i,j,1]+3*r_k[i,j]+1*r[i,j,2])
                                + (1/480)*A_s[i,j,2]*(3*r[i,j,2]+7*r_k[i,j]+5*r[i,j,3])
                                + (1/480)*A_s[i,j,3]*(23*r[i,j,0]+19*r_k[i,j]+23*r[i,j,3]))
            
            T_mat[i,j][1,2] = ((1/480)*A_s[i,j,1]*(23*r[i,j,1]+19*r_k[i,j]+23*r[i,j,2])
                                + (1/480)*A_s[i,j,2]*(5*r[i,j,2]+7*r_k[i,j]+3*r[i,j,3])
                                + (1/480)*A_s[i,j,3]*(1*r[i,j,3]+3*r_k[i,j]+1*r[i,j,0])
                                + (1/480)*A_s[i,j,0]*(5*r[i,j,1]+7*r_k[i,j]+3*r[i,j,0]))
            
            T_mat[i,j][1,3] = ((1/480)*A_s[i,j,1]*(5*r[i,j,1]+7*r_k[i,j]+3*r[i,j,2])
                                + (1/480)*A_s[i,j,2]*(3*r[i,j,2]+7*r_k[i,j]+5*r[i,j,3])
                                + (1/480)*A_s[i,j,3]*(5*r[i,j,3]+7*r_k[i,j]+3*r[i,j,0])
                                + (1/480)*A_s[i,j,0]*(5*r[i,j,1]+7*r_k[i,j]+3*r[i,j,0]))
            
            T_mat[i,j][1,0] = ((1/480)*A_s[i,j,1]*(5*r[i,j,1]+7*r_k[i,j]+3*r[i,j,2])
                                + (1/480)*A_s[i,j,2]*(1*r[i,j,2]+3*r_k[i,j]+1*r[i,j,3])
                                + (1/480)*A_s[i,j,3]*(3*r[i,j,3]+7*r_k[i,j]+5*r[i,j,0])
                                + (1/480)*A_s[i,j,0]*(23*r[i,j,1]+19*r_k[i,j]+23*r[i,j,0]))
            
            T_mat[i,j][2,3] = ((1/480)*A_s[i,j,2]*(23*r[i,j,2]+19*r_k[i,j]+23*r[i,j,3])
                                + (1/480)*A_s[i,j,3]*(5*r[i,j,3]+7*r_k[i,j]+3*r[i,j,0])
                                + (1/480)*A_s[i,j,0]*(1*r[i,j,0]+3*r_k[i,j]+1*r[i,j,1])
                                + (1/480)*A_s[i,j,1]*(5*r[i,j,2]+7*r_k[i,j]+3*r[i,j,1]))
            
            T_mat[i,j][2,0] = ((1/480)*A_s[i,j,2]*(5*r[i,j,2]+7*r_k[i,j]+3*r[i,j,3])
                                + (1/480)*A_s[i,j,3]*(3*r[i,j,3]+7*r_k[i,j]+5*r[i,j,0])
                                + (1/480)*A_s[i,j,0]*(5*r[i,j,0]+7*r_k[i,j]+3*r[i,j,1])
                                + (1/480)*A_s[i,j,1]*(5*r[i,j,2]+7*r_k[i,j]+3*r[i,j,1]))
            
            T_mat[i,j][2,1] = ((1/480)*A_s[i,j,2]*(5*r[i,j,2]+7*r_k[i,j]+3*r[i,j,3])
                                + (1/480)*A_s[i,j,3]*(1*r[i,j,3]+3*r_k[i,j]+1*r[i,j,0])
                                + (1/480)*A_s[i,j,0]*(3*r[i,j,0]+7*r_k[i,j]+5*r[i,j,1])
                                + (1/480)*A_s[i,j,1]*(23*r[i,j,2]+19*r_k[i,j]+23*r[i,j,1]))
            
            T_mat[i,j][3,0] = ((1/480)*A_s[i,j,3]*(23*r[i,j,3]+19*r_k[i,j]+23*r[i,j,0])
                                + (1/480)*A_s[i,j,0]*(5*r[i,j,0]+7*r_k[i,j]+3*r[i,j,1])
                                + (1/480)*A_s[i,j,1]*(1*r[i,j,1]+3*r_k[i,j]+1*r[i,j,2])
                                + (1/480)*A_s[i,j,2]*(5*r[i,j,3]+7*r_k[i,j]+3*r[i,j,2]))
            
            T_mat[i,j][3,1] = ((1/480)*A_s[i,j,3]*(5*r[i,j,3]+7*r_k[i,j]+3*r[i,j,0])
                                + (1/480)*A_s[i,j,0]*(3*r[i,j,0]+7*r_k[i,j]+5*r[i,j,1])
                                + (1/480)*A_s[i,j,1]*(5*r[i,j,1]+7*r_k[i,j]+3*r[i,j,2])
                                + (1/480)*A_s[i,j,2]*(5*r[i,j,3]+7*r_k[i,j]+3*r[i,j,2]))
            
            T_mat[i,j][3,2] = ((1/480)*A_s[i,j,3]*(5*r[i,j,3]+7*r_k[i,j]+3*r[i,j,0])
                                + (1/480)*A_s[i,j,0]*(1*r[i,j,0]+3*r_k[i,j]+1*r[i,j,1])
                                + (1/480)*A_s[i,j,1]*(3*r[i,j,1]+7*r_k[i,j]+5*r[i,j,2])
                                + (1/480)*A_s[i,j,2]*(23*r[i,j,3]+19*r_k[i,j]+23*r[i,j,2]))
            
            R_mat[i,j][0,0] = (1/96)*(21*A_s[i,j,0]+1*A_s[i,j,1]+1*A_s[i,j,2]+21*A_s[i,j,3])
            R_mat[i,j][0,1] = (1/96)*(13*A_s[i,j,0]+3*A_s[i,j,1]+1*A_s[i,j,2]+3*A_s[i,j,3])
            R_mat[i,j][0,2] = (1/96)*(3*A_s[i,j,0]+3*A_s[i,j,1]+3*A_s[i,j,2]+3*A_s[i,j,3])
            R_mat[i,j][0,3] = (1/96)*(3*A_s[i,j,0]+1*A_s[i,j,1]+3*A_s[i,j,2]+13*A_s[i,j,3])        
            R_mat[i,j][1,1] = (1/96)*(21*A_s[i,j,1]+1*A_s[i,j,2]+1*A_s[i,j,3]+21*A_s[i,j,0])
            R_mat[i,j][1,2] = (1/96)*(13*A_s[i,j,1]+3*A_s[i,j,2]+1*A_s[i,j,3]+3*A_s[i,j,0])
            R_mat[i,j][1,3] = (1/96)*(3*A_s[i,j,1]+3*A_s[i,j,2]+3*A_s[i,j,3]+3*A_s[i,j,0])
            R_mat[i,j][1,0] = (1/96)*(3*A_s[i,j,1]+1*A_s[i,j,2]+3*A_s[i,j,3]+13*A_s[i,j,0])            
            R_mat[i,j][2,2] = (1/96)*(21*A_s[i,j,2]+1*A_s[i,j,3]+1*A_s[i,j,0]+21*A_s[i,j,1])
            R_mat[i,j][2,3] = (1/96)*(13*A_s[i,j,2]+3*A_s[i,j,3]+1*A_s[i,j,0]+3*A_s[i,j,1])
            R_mat[i,j][2,0] = (1/96)*(3*A_s[i,j,2]+3*A_s[i,j,3]+3*A_s[i,j,0]+3*A_s[i,j,1])
            R_mat[i,j][2,1] = (1/96)*(3*A_s[i,j,2]+1*A_s[i,j,3]+3*A_s[i,j,0]+13*A_s[i,j,1])            
            R_mat[i,j][3,3] = (1/96)*(21*A_s[i,j,3]+1*A_s[i,j,0]+1*A_s[i,j,1]+21*A_s[i,j,2])
            R_mat[i,j][3,0] = (1/96)*(13*A_s[i,j,3]+3*A_s[i,j,0]+1*A_s[i,j,1]+3*A_s[i,j,2])
            R_mat[i,j][3,1] = (1/96)*(3*A_s[i,j,3]+3*A_s[i,j,0]+3*A_s[i,j,1]+3*A_s[i,j,2])
            R_mat[i,j][3,2] = (1/96)*(3*A_s[i,j,3]+1*A_s[i,j,0]+3*A_s[i,j,1]+13*A_s[i,j,2])
            
            Kr_mat[i,j][0,0] = ((1/48)*dw_dr[i,j,0,0]*(9*r[i,j,0]+6*r_k[i,j]+5*r[i,j,1])
                                + (1/48)*dw_dr[i,j,0,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2])
                                + (1/48)*dw_dr[i,j,0,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dr[i,j,0,3]*(5*r[i,j,3]+6*r_k[i,j]+9*r[i,j,0]))
            
            Kr_mat[i,j][0,1] = ((1/48)*dw_dr[i,j,0,0]*(5*r[i,j,0]+6*r_k[i,j]+9*r[i,j,1])
                                + (1/48)*dw_dr[i,j,0,1]*(9*r[i,j,1]+6*r_k[i,j]+5*r[i,j,2])
                                + (1/48)*dw_dr[i,j,0,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dr[i,j,0,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0]))
            
            Kr_mat[i,j][0,2] = ((1/48)*dw_dr[i,j,0,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1])
                                + (1/48)*dw_dr[i,j,0,1]*(5*r[i,j,1]+6*r_k[i,j]+9*r[i,j,2])
                                + (1/48)*dw_dr[i,j,0,2]*(9*r[i,j,2]+6*r_k[i,j]+5*r[i,j,3])
                                + (1/48)*dw_dr[i,j,0,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0]))
            
            Kr_mat[i,j][0,3] = ((1/48)*dw_dr[i,j,0,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1])
                                + (1/48)*dw_dr[i,j,0,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2])
                                + (1/48)*dw_dr[i,j,0,2]*(5*r[i,j,2]+6*r_k[i,j]+9*r[i,j,3])
                                + (1/48)*dw_dr[i,j,0,3]*(9*r[i,j,3]+6*r_k[i,j]+5*r[i,j,0]))
            
            Kr_mat[i,j][1,1] = ((1/48)*dw_dr[i,j,1,1]*(9*r[i,j,1]+6*r_k[i,j]+5*r[i,j,2])
                                + (1/48)*dw_dr[i,j,1,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dr[i,j,1,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0])
                                + (1/48)*dw_dr[i,j,1,0]*(5*r[i,j,0]+6*r_k[i,j]+9*r[i,j,1]))
            
            Kr_mat[i,j][1,2] = ((1/48)*dw_dr[i,j,1,1]*(5*r[i,j,1]+6*r_k[i,j]+9*r[i,j,2])
                                + (1/48)*dw_dr[i,j,1,2]*(9*r[i,j,2]+6*r_k[i,j]+5*r[i,j,3])
                                + (1/48)*dw_dr[i,j,1,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0])
                                + (1/48)*dw_dr[i,j,1,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1]))
            
            Kr_mat[i,j][1,3] = ((1/48)*dw_dr[i,j,1,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2])
                                + (1/48)*dw_dr[i,j,1,2]*(5*r[i,j,2]+6*r_k[i,j]+9*r[i,j,3])
                                + (1/48)*dw_dr[i,j,1,3]*(9*r[i,j,3]+6*r_k[i,j]+5*r[i,j,0])
                                + (1/48)*dw_dr[i,j,1,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1]))
            
            Kr_mat[i,j][1,0] = ((1/48)*dw_dr[i,j,1,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2])
                                + (1/48)*dw_dr[i,j,1,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dr[i,j,1,3]*(5*r[i,j,3]+6*r_k[i,j]+9*r[i,j,0])
                                + (1/48)*dw_dr[i,j,1,0]*(9*r[i,j,0]+6*r_k[i,j]+5*r[i,j,1]))
            
            Kr_mat[i,j][2,2] = ((1/48)*dw_dr[i,j,2,2]*(9*r[i,j,2]+6*r_k[i,j]+5*r[i,j,3])
                                + (1/48)*dw_dr[i,j,2,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0])
                                + (1/48)*dw_dr[i,j,2,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1])
                                + (1/48)*dw_dr[i,j,2,1]*(5*r[i,j,1]+6*r_k[i,j]+9*r[i,j,2]))
            
            Kr_mat[i,j][2,3] = ((1/48)*dw_dr[i,j,2,2]*(5*r[i,j,2]+6*r_k[i,j]+9*r[i,j,3])
                                + (1/48)*dw_dr[i,j,2,3]*(9*r[i,j,3]+6*r_k[i,j]+5*r[i,j,0])
                                + (1/48)*dw_dr[i,j,2,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1])
                                + (1/48)*dw_dr[i,j,2,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2]))
            
            Kr_mat[i,j][2,0] = ((1/48)*dw_dr[i,j,2,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dr[i,j,2,3]*(5*r[i,j,3]+6*r_k[i,j]+9*r[i,j,0])
                                + (1/48)*dw_dr[i,j,2,0]*(9*r[i,j,0]+6*r_k[i,j]+5*r[i,j,1])
                                + (1/48)*dw_dr[i,j,2,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2]))
            
            Kr_mat[i,j][2,1] = ((1/48)*dw_dr[i,j,2,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dr[i,j,2,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0])
                                + (1/48)*dw_dr[i,j,2,0]*(5*r[i,j,0]+6*r_k[i,j]+9*r[i,j,1])
                                + (1/48)*dw_dr[i,j,2,1]*(9*r[i,j,1]+6*r_k[i,j]+5*r[i,j,2]))
            
            Kr_mat[i,j][3,3] = ((1/48)*dw_dr[i,j,3,3]*(9*r[i,j,3]+6*r_k[i,j]+5*r[i,j,0])
                                + (1/48)*dw_dr[i,j,3,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2])
                                + (1/48)*dw_dr[i,j,3,2]*(5*r[i,j,2]+6*r_k[i,j]+9*r[i,j,3])
                                + (1/48)*dw_dr[i,j,3,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1]))
            
            Kr_mat[i,j][3,0] = ((1/48)*dw_dr[i,j,3,3]*(5*r[i,j,3]+6*r_k[i,j]+9*r[i,j,0])
                                + (1/48)*dw_dr[i,j,3,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2])
                                + (1/48)*dw_dr[i,j,3,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dr[i,j,3,0]*(9*r[i,j,0]+6*r_k[i,j]+5*r[i,j,1]))
            
            Kr_mat[i,j][3,1] = ((1/48)*dw_dr[i,j,3,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0])
                                + (1/48)*dw_dr[i,j,3,1]*(9*r[i,j,1]+6*r_k[i,j]+5*r[i,j,2])
                                + (1/48)*dw_dr[i,j,3,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dr[i,j,3,0]*(5*r[i,j,0]+6*r_k[i,j]+9*r[i,j,1]))
            
            Kr_mat[i,j][3,2] = ((1/48)*dw_dr[i,j,3,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0])
                                + (1/48)*dw_dr[i,j,3,1]*(5*r[i,j,1]+6*r_k[i,j]+9*r[i,j,2])
                                + (1/48)*dw_dr[i,j,3,2]*(9*r[i,j,2]+6*r_k[i,j]+5*r[i,j,3])
                                + (1/48)*dw_dr[i,j,3,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1]))
            
            Kz_mat[i,j][0,0] = ((1/48)*dw_dz[i,j,0,0]*(9*r[i,j,0]+6*r_k[i,j]+5*r[i,j,1])
                                + (1/48)*dw_dz[i,j,0,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2])
                                + (1/48)*dw_dz[i,j,0,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dz[i,j,0,3]*(5*r[i,j,3]+6*r_k[i,j]+9*r[i,j,0]))
                                
            Kz_mat[i,j][0,1] = ((1/48)*dw_dz[i,j,0,0]*(5*r[i,j,0]+6*r_k[i,j]+9*r[i,j,1])
                                + (1/48)*dw_dz[i,j,0,1]*(9*r[i,j,1]+6*r_k[i,j]+5*r[i,j,2])
                                + (1/48)*dw_dz[i,j,0,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dz[i,j,0,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0]))
                                
            Kz_mat[i,j][0,2] = ((1/48)*dw_dz[i,j,0,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1])
                                + (1/48)*dw_dz[i,j,0,1]*(5*r[i,j,1]+6*r_k[i,j]+9*r[i,j,2])
                                + (1/48)*dw_dz[i,j,0,2]*(9*r[i,j,2]+6*r_k[i,j]+5*r[i,j,3])
                                + (1/48)*dw_dz[i,j,0,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0]))
                                
            Kz_mat[i,j][0,3] = ((1/48)*dw_dz[i,j,0,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1])
                                + (1/48)*dw_dz[i,j,0,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2])
                                + (1/48)*dw_dz[i,j,0,2]*(5*r[i,j,2]+6*r_k[i,j]+9*r[i,j,3])
                                + (1/48)*dw_dz[i,j,0,3]*(9*r[i,j,3]+6*r_k[i,j]+5*r[i,j,0]))
                                
            Kz_mat[i,j][1,1] = ((1/48)*dw_dz[i,j,1,1]*(9*r[i,j,1]+6*r_k[i,j]+5*r[i,j,2])
                                + (1/48)*dw_dz[i,j,1,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dz[i,j,1,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0])
                                + (1/48)*dw_dz[i,j,1,0]*(5*r[i,j,0]+6*r_k[i,j]+9*r[i,j,1]))
                                
            Kz_mat[i,j][1,2] = ((1/48)*dw_dz[i,j,1,1]*(5*r[i,j,1]+6*r_k[i,j]+9*r[i,j,2])
                                + (1/48)*dw_dz[i,j,1,2]*(9*r[i,j,2]+6*r_k[i,j]+5*r[i,j,3])
                                + (1/48)*dw_dz[i,j,1,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0])
                                + (1/48)*dw_dz[i,j,1,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1]))
            
            Kz_mat[i,j][1,3] = ((1/48)*dw_dz[i,j,1,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2])
                                + (1/48)*dw_dz[i,j,1,2]*(5*r[i,j,2]+6*r_k[i,j]+9*r[i,j,3])
                                + (1/48)*dw_dz[i,j,1,3]*(9*r[i,j,3]+6*r_k[i,j]+5*r[i,j,0])
                                + (1/48)*dw_dz[i,j,1,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1]))
                                
            Kz_mat[i,j][1,0] = ((1/48)*dw_dz[i,j,1,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2])
                                + (1/48)*dw_dz[i,j,1,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dz[i,j,1,3]*(5*r[i,j,3]+6*r_k[i,j]+9*r[i,j,0])
                                + (1/48)*dw_dz[i,j,1,0]*(9*r[i,j,0]+6*r_k[i,j]+5*r[i,j,1]))
                                
            Kz_mat[i,j][2,2] = ((1/48)*dw_dz[i,j,2,2]*(9*r[i,j,2]+6*r_k[i,j]+5*r[i,j,3])
                                + (1/48)*dw_dz[i,j,2,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0])
                                + (1/48)*dw_dz[i,j,2,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1])
                                + (1/48)*dw_dz[i,j,2,1]*(5*r[i,j,1]+6*r_k[i,j]+9*r[i,j,2]))
                                
            Kz_mat[i,j][2,3] = ((1/48)*dw_dz[i,j,2,2]*(5*r[i,j,2]+6*r_k[i,j]+9*r[i,j,3])
                                + (1/48)*dw_dz[i,j,2,3]*(9*r[i,j,3]+6*r_k[i,j]+5*r[i,j,0])
                                + (1/48)*dw_dz[i,j,2,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1])
                                + (1/48)*dw_dz[i,j,2,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2]))
                                
            Kz_mat[i,j][2,0] = ((1/48)*dw_dz[i,j,2,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dz[i,j,2,3]*(5*r[i,j,3]+6*r_k[i,j]+9*r[i,j,0])
                                + (1/48)*dw_dz[i,j,2,0]*(9*r[i,j,0]+6*r_k[i,j]+5*r[i,j,1])
                                + (1/48)*dw_dz[i,j,2,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2]))
                                
            Kz_mat[i,j][2,1] = ((1/48)*dw_dz[i,j,2,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dz[i,j,2,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0])
                                + (1/48)*dw_dz[i,j,2,0]*(5*r[i,j,0]+6*r_k[i,j]+9*r[i,j,1])
                                + (1/48)*dw_dz[i,j,2,1]*(9*r[i,j,1]+6*r_k[i,j]+5*r[i,j,2]))
                                
            Kz_mat[i,j][3,3] = ((1/48)*dw_dz[i,j,3,3]*(9*r[i,j,3]+6*r_k[i,j]+5*r[i,j,0])
                                + (1/48)*dw_dz[i,j,3,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2])
                                + (1/48)*dw_dz[i,j,3,2]*(5*r[i,j,2]+6*r_k[i,j]+9*r[i,j,3])
                                + (1/48)*dw_dz[i,j,3,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1]))
                                
            Kz_mat[i,j][3,0] = ((1/48)*dw_dz[i,j,3,3]*(5*r[i,j,3]+6*r_k[i,j]+9*r[i,j,0])
                                + (1/48)*dw_dz[i,j,3,1]*(1*r[i,j,1]+2*r_k[i,j]+1*r[i,j,2])
                                + (1/48)*dw_dz[i,j,3,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dz[i,j,3,0]*(9*r[i,j,0]+6*r_k[i,j]+5*r[i,j,1]))
                                
            Kz_mat[i,j][3,1] = ((1/48)*dw_dz[i,j,3,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0])
                                + (1/48)*dw_dz[i,j,3,1]*(9*r[i,j,1]+6*r_k[i,j]+5*r[i,j,2])
                                + (1/48)*dw_dz[i,j,3,2]*(1*r[i,j,2]+2*r_k[i,j]+1*r[i,j,3])
                                + (1/48)*dw_dz[i,j,3,0]*(5*r[i,j,0]+6*r_k[i,j]+9*r[i,j,1]))
                                
            Kz_mat[i,j][3,2] = ((1/48)*dw_dz[i,j,3,3]*(1*r[i,j,3]+2*r_k[i,j]+1*r[i,j,0])
                                + (1/48)*dw_dz[i,j,3,1]*(5*r[i,j,1]+6*r_k[i,j]+9*r[i,j,2])
                                + (1/48)*dw_dz[i,j,3,2]*(9*r[i,j,2]+6*r_k[i,j]+5*r[i,j,3])
                                + (1/48)*dw_dz[i,j,3,0]*(1*r[i,j,0]+2*r_k[i,j]+1*r[i,j,1]))
    for i in range(Ni):
        for j in range(Nj):        
            Kr_mat[i,j,:,:] = -1*Kr_mat[i,j,:,:]
            Kz_mat[i,j,:,:] = -1*Kz_mat[i,j,:,:]
            
    return T_mat,Kr_mat,Kz_mat,R_mat,Lr_cminus,Lr_cplus,Lz_cminus,Lz_cplus,theta,theta_half,r,z,r_k,z_k


def Solver(method,Ni,Nj,Nazi,Np,R,J,Sig_s,Sig_t,phi_old,Q,psi_inc,tol,maxit):
    if method == "G":
        T_mat,Kr_mat,Kz_mat,R_mat,Lr_cminus,Lr_cplus,Lz_cminus,Lz_cplus,theta,theta_half,r,z,r_k,z_k = G(Ni,Nj,Nazi,Np,R,J)
    elif method == "CB": 
        T_mat,Kr_mat,Kz_mat,R_mat,Lr_cminus,Lr_cplus,Lz_cminus,Lz_cplus,theta,theta_half,r,z,r_k,z_k = CB(Ni,Nj,Nazi,Np,R,J)
    
    # polar weights w_p and polar points ksi_p
    ksi_p,w_p = np.polynomial.legendre.leggauss(Np)
    # azimuthal weights w_q and angles omega_q
    w_q = np.zeros(Nazi)
    omega_q = np.zeros(Nazi)
    omega_qhalf = np.zeros(Nazi)
    for i in range(0,Nazi):
        w_q[i] = np.pi/(Nazi)
        omega_q[i] = (np.pi/(2*Nazi))*((2*Nazi)+1-(2*(i+1)))
        omega_qhalf[i] = (np.pi/(2*Nazi))*((2*Nazi)+1-(2*(i+1+0.5)))
    # mu_pq and w_pq
    mu_pqhalf = np.zeros([Np,Nazi+1])
    mu_pq = np.zeros([Np,Nazi])
    w_pq = np.zeros([Np,Nazi])
    for i in range(Np):
        for j in range(Nazi):
            mu_pqhalf[i,j+1] = np.sqrt(1-(ksi_p[i]**2))*(np.cos(omega_qhalf[j]))
            mu_pq[i,j] = np.sqrt(1-(ksi_p[i]**2))*(np.cos(omega_q[j]))
            w_pq[i,j] = w_p[i]*w_q[j]
    
    #need = np.ones((Ni,Nj))
    #jlist = []
    nextwedgej = 0
    jstart = 0
    jj = 0
    spokecross = np.zeros(Nj+1)
    LHS = np.ndarray(shape = (Np,Nazi,Ni,Nj,J,J))
    RHS = np.ndarray(shape = (Np,Nazi,Ni,Nj,J))
    psi = np.ndarray(shape = (Np,Nazi,Ni,Nj,J))
    psi_half = np.ndarray(shape = (Np,Nazi+1,Ni,Nj,J))
    psi_inc1 = np.zeros(shape = (Nj,J))
    mu = np.zeros(Np)  
    alpha = np.zeros([Np,Nazi+1])
    tau = np.zeros([Np,Nazi])
    converged= False
    count=0
    phi = np.zeros(shape = (Ni,Nj,J))
    psi_bc = np.zeros(shape = (Nj,J))
    
    for j in range(Nj):
        psi_bc[j,1] = psi_inc
        psi_bc[j,2] = psi_inc
    
    while converged == False:
        phi_old = phi
        psi_inc1[:,:] = psi_bc[:,:]
        for p in range(ksi_p.size):
            mu[p] = -np.sqrt(1-(ksi_p[p]**2))
            need = np.ones((Ni,Nj))
            psi_inc1[:,:] = psi_bc[:,:]
            for j in range(Nj+1):
                if (mu[p]*np.sin(theta_half[j])-ksi_p[p]*np.cos(theta_half[j]))<0:
                    spokecross[j] = 1    #means increase
                else:
                    spokecross[j] = 0    #means decrease
                
            #loop over all cells
            for i in range(Ni):
                for j in range(Nj):
                    if (mu[p]*np.sin(theta_half[j])-ksi_p[p]*np.cos(theta_half[j]))<0:
                        need[i,j] += 1
                    elif (-mu[p]*np.sin(theta_half[j+1])+ksi_p[p]*np.cos(theta_half[j+1]))<0:
                        need[i,j] += 1
                    #correct for center points    
                    if i == 0:                       
                        if (mu[p]*np.cos(theta[j])+ksi_p[p]*np.sin(theta[j]))>0:
                            need[i,j] -= 1
                    #looping over boundary cells
                    if i == Ni-1:                        
                        if (mu[p]*np.cos(theta[j])+ksi_p[p]*np.sin(theta[j]))<0:
                            need[i,j] -= 1
                    #start sweeping
                    if need[i,j] == 0:
                        nextwedgej = j 
                        jstart = nextwedgej
                        if (mu[p]*np.cos(theta[j])+ksi_p[p]*np.sin(theta[j]))<0:
                            istart = Ni-1
                            iend = -1
                            i_incr = -1
                        else:
                            istart = 0
                            iend = Ni
                            i_incr = 1          
            j = jstart
            jj = jstart
            loop = 0
            while loop <((Ni*Nj)-1):
                j = jj
                psi_inc1[j,:] = psi_inc1[jj,:]
                #order of sweep 
                for i in range(istart,iend,i_incr):
                    RHS[p,0,i,j,:] = (((1/(2*np.pi))*np.dot(T_mat[i,j,:,:],(Sig_s*phi[i,j,:]+Q))))
                    LHS[p,0,i,j,:,:] = (mu[p]*(Kr_mat[i,j,:,:]-R_mat[i,j,:,:])
                                        +ksi_p[p]*Kz_mat[i,j,:,:]+Sig_t*T_mat[i,j,:,:])
                    if (i_incr == -1):
                        RHS[p,0,i,j,1] -= ((mu[p]*Lr_cplus[i,j,1,1]+ksi_p[p]*Lz_cplus[i,j,1,1])*psi_inc1[j,1] 
                                    + (mu[p]*Lr_cplus[i,j,1,2]+ksi_p[p]*Lz_cplus[i,j,1,2])*psi_inc1[j,2])
                        RHS[p,0,i,j,2] -= ((mu[p]*Lr_cminus[i,j,2,2]+ksi_p[p]*Lz_cminus[i,j,2,2])*psi_inc1[j,2] 
                                    + (mu[p]*Lr_cminus[i,j,2,1]+ksi_p[p]*Lz_cminus[i,j,2,1])*psi_inc1[j,1])
                        LHS[p,0,i,j,0,0] += mu[p]*Lr_cminus[i,j,0,0] + ksi_p[p]*Lz_cminus[i,j,0,0]
                        LHS[p,0,i,j,0,3] += mu[p]*Lr_cminus[i,j,0,3] + ksi_p[p]*Lz_cminus[i,j,0,3]
                        LHS[p,0,i,j,3,3] += mu[p]*Lr_cplus[i,j,3,3] + ksi_p[p]*Lz_cplus[i,j,3,3]
                        LHS[p,0,i,j,3,0] += mu[p]*Lr_cplus[i,j,3,0] + ksi_p[p]*Lz_cplus[i,j,3,0]
                    else:
                        #psi_inc1[j,0] = psi_bc[jstart,0]
                        #psi_inc1[j,3] = psi_bc[jstart,3]
                        RHS[p,0,i,j,0] -= ((mu[p]*Lr_cminus[i,j,0,0]+ksi_p[p]*Lz_cminus[i,j,0,0])*psi_inc1[j,0] 
                                + (mu[p]*Lr_cminus[i,j,0,3]+ksi_p[p]*Lz_cminus[i,j,0,3])*psi_inc1[j,3])
                        RHS[p,0,i,j,3] -= ((mu[p]*Lr_cplus[i,j,3,3]+ksi_p[p]*Lz_cplus[i,j,3,3])*psi_inc1[j,3] 
                                + (mu[p]*Lr_cplus[i,j,3,0]+ksi_p[p]*Lz_cplus[i,j,3,0])*psi_inc1[j,0])
                        LHS[p,0,i,j,1,1] += mu[p]*Lr_cplus[i,j,1,1] + ksi_p[p]*Lz_cplus[i,j,1,1]
                        LHS[p,0,i,j,1,2] += mu[p]*Lr_cplus[i,j,1,2] + ksi_p[p]*Lz_cplus[i,j,1,2]
                        LHS[p,0,i,j,2,2] += mu[p]*Lr_cminus[i,j,2,2] + ksi_p[p]*Lz_cminus[i,j,2,2]
                        LHS[p,0,i,j,2,1] += mu[p]*Lr_cminus[i,j,2,1] + ksi_p[p]*Lz_cminus[i,j,2,1]
                    
                    if spokecross[j] == 1:
                        RHS[p,0,i,j,0] -= ((mu[p]*Lr_cplus[i,j,0,0]+ksi_p[p]*Lz_cplus[i,j,0,0])*psi_inc1[j,0] 
                                + (mu[p]*Lr_cplus[i,j,0,1]+ksi_p[p]*Lz_cplus[i,j,0,1])*psi_inc1[j,1])
                        RHS[p,0,i,j,1] -= ((mu[p]*Lr_cminus[i,j,1,1]+ksi_p[p]*Lz_cminus[i,j,1,1])*psi_inc1[j,1] 
                                + (mu[p]*Lr_cminus[i,j,1,0]+ksi_p[p]*Lz_cminus[i,j,1,0])*psi_inc1[j,0])
                    else: 
                        LHS[p,0,i,j,0,0] += mu[p]*Lr_cplus[i,j,0,0]+ksi_p[p]*Lz_cplus[i,j,0,0]
                        LHS[p,0,i,j,0,1] += mu[p]*Lr_cplus[i,j,0,1]+ksi_p[p]*Lz_cplus[i,j,0,1]
                        LHS[p,0,i,j,1,1] += mu[p]*Lr_cminus[i,j,1,1]+ksi_p[p]*Lz_cminus[i,j,1,1]
                        LHS[p,0,i,j,1,0] += mu[p]*Lr_cminus[i,j,1,0]+ksi_p[p]*Lz_cminus[i,j,1,0]
                    
                    if spokecross[j+1] == 0:
                        RHS[p,0,i,j,2] -= ((mu[p]*Lr_cplus[i,j,2,2]+ksi_p[p]*Lz_cplus[i,j,2,2])*psi_inc1[j,2] 
                                + (mu[p]*Lr_cplus[i,j,2,3]+ksi_p[p]*Lz_cplus[i,j,2,3])*psi_inc1[j,3])
                        RHS[p,0,i,j,3] -= ((mu[p]*Lr_cminus[i,j,3,3]+ksi_p[p]*Lz_cminus[i,j,3,3])*psi_inc1[j,3] 
                                + (mu[p]*Lr_cminus[i,j,3,2]+ksi_p[p]*Lz_cminus[i,j,3,2])*psi_inc1[j,2])
                    else: 
                        LHS[p,0,i,j,2,2] += mu[p]*Lr_cplus[i,j,2,2]+ksi_p[p]*Lz_cplus[i,j,2,2]
                        LHS[p,0,i,j,2,3] += mu[p]*Lr_cplus[i,j,2,3]+ksi_p[p]*Lz_cplus[i,j,2,3]
                        LHS[p,0,i,j,3,3] += mu[p]*Lr_cminus[i,j,3,3]+ksi_p[p]*Lz_cminus[i,j,3,3]
                        LHS[p,0,i,j,3,2] += mu[p]*Lr_cminus[i,j,3,2]+ksi_p[p]*Lz_cminus[i,j,3,2]
                    #solve for psi_half    
                    psi_half[p,0,i,j,:] = np.dot(np.linalg.inv(LHS[p,0,i,j,:,:]),RHS[p,0,i,j,:])
                    loop+=1
                if loop <((Ni*Nj)-1):
                    if j<= jstart:
                        if j > 0:
                            jj = j-1
                        else:
                            jj = jstart+1
                    else:
                        jj = j+1
                    if (mu[p]*np.cos(theta[jj])+ksi_p[p]*np.sin(theta[jj]))<0:
                        istart = Ni-1
                        iend = -1
                        i_incr = -1
                    else:
                        istart = 0
                        iend = Ni
                        i_incr = 1
                    if i_incr == -1:
                        psi_inc1[jj,1] = psi_half[p,0,i,j,0]
                        psi_inc1[jj,2] = psi_half[p,0,i,j,3]
                    else:
                        psi_inc1[jj,0] = psi_half[p,0,i,j,1]
                        psi_inc1[jj,3] = psi_half[p,0,i,j,2]
                        if i == 0:
                            psi_bc[jj,0] = psi_half[p,0,i,j,0]
                            psi_bc[jj,3] = psi_half[p,0,i,j,3]                     
                    if spokecross[jj] == 0:
                        psi_inc1[jj,2] = psi_half[p,0,i,j,1]
                        psi_inc1[jj,3] = psi_half[p,0,i,j,0]
                    elif spokecross[jj+1] == 1:
                        psi_inc1[jj,0] = psi_half[p,0,i,j,3]
                        psi_inc1[jj,1] = psi_half[p,0,i,j,2] 
                else: 
                    break
                 
                    
            alpha[p,0] = 0
            mu_pqhalf[p,0] = mu[p]
                        
            for q in range(Nazi):
                psi_inc1[:,:] = psi_bc[:,:]
                if mu_pq[p,q] < 0:
                    need = np.ones((Ni,Nj))
                    for j in range(Nj+1):
                        if (mu_pq[p,q]*np.sin(theta_half[j])-ksi_p[p]*np.cos(theta_half[j]))<0:
                            spokecross[j] = 1
                        else:
                            spokecross[j] = 0
                    #loop over all cells
                    for i in range(Ni):
                        for j in range(Nj):   
                            if (mu_pq[p,q]*np.sin(theta_half[j])-ksi_p[p]*np.cos(theta_half[j]))<0:
                                need[i,j] += 1
                                #print("for i=",i," and j=",j," 1+ added")
                            elif (-mu_pq[p,q]*np.sin(theta_half[j+1])+ksi_p[p]*np.cos(theta_half[j+1]))<0:
                                need[i,j] += 1
                                #print("for i=",i," and j=",j," 3+ added")
                        #correct for center points    
                            if i == 0:
                                if (mu_pq[p,q]*np.cos(theta[j])+ksi_p[p]*np.sin(theta[j]))>0:
                                    need[i,j] -= 1
                                    #print("for i=",i," and j=",j," 2+ subtracted")
                        #looping over boundary cells
                            elif i == Ni-1:
                                if (mu_pq[p,q]*np.cos(theta[j])+ksi_p[p]*np.sin(theta[j]))<0:
                                    need[i,j] -= 1
                                    #print("for i=",i," and j=",j," 2+ subtracted")
                            #start sweeping
                            if need[i,j] == 0:
                                jstart = j
                                if (mu_pq[p,q]*np.cos(theta[j])+ksi_p[p]*np.sin(theta[j]))<0:
                                    istart = Ni-1
                                    iend = -1
                                    i_incr = -1
                                else:
                                    istart = 0
                                    iend = Ni
                                    i_incr = 1        
                    j = jstart
                    jj = jstart
                    loop = 0
                    while loop <((Ni*Nj)-1):
                        j=jj
                        psi_inc1[j,:] = psi_inc1[jj,:]
                        for i in range(istart,iend,i_incr):
                            alpha[p,q+1] = alpha[p,q]-w_pq[p,q]*mu_pq[p,q]
                            tau[p,q] = (mu_pq[p,q]-mu_pqhalf[p,q])/(mu_pqhalf[p,q+1]-mu_pqhalf[p,q])
                            RHS[p,q,i,j,:] = (((1/(2*np.pi))*np.dot(T_mat[i,j,:,:],(Sig_s*phi[i,j,:]+Q)))
                                                +(np.dot(R_mat[i,j,:,:],(alpha[p,q+1]*((1-tau[p,q])/tau[p,q])
                                                       +alpha[p,q])*psi_half[p,q,i,j,:]/w_pq[p,q])))
                                             
                            LHS[p,q,i,j,:,:] = (mu_pq[p,q]*Kr_mat[i,j,:,:]
                                                +ksi_p[p]*Kz_mat[i,j,:,:]+Sig_t*T_mat[i,j,:,:]
                                                +R_mat[i,j,:,:]*(alpha[p,q+1]/(w_pq[p,q]*tau[p,q])))
    
                            if (i_incr == -1):
                                RHS[p,q,i,j,1] -= ((mu_pq[p,q]*Lr_cplus[i,j,1,1]+ksi_p[p]*Lz_cplus[i,j,1,1])*psi_inc1[j,1] 
                                            + (mu_pq[p,q]*Lr_cplus[i,j,1,2]+ksi_p[p]*Lz_cplus[i,j,1,2])*psi_inc1[j,2])
                                RHS[p,q,i,j,2] -= ((mu_pq[p,q]*Lr_cminus[i,j,2,2]+ksi_p[p]*Lz_cminus[i,j,2,2])*psi_inc1[j,2] 
                                            + (mu_pq[p,q]*Lr_cminus[i,j,2,1]+ksi_p[p]*Lz_cminus[i,j,2,1])*psi_inc1[j,1])
                                LHS[p,q,i,j,0,0] += mu_pq[p,q]*Lr_cminus[i,j,0,0] + ksi_p[p]*Lz_cminus[i,j,0,0]
                                LHS[p,q,i,j,0,3] += mu_pq[p,q]*Lr_cminus[i,j,0,3] + ksi_p[p]*Lz_cminus[i,j,0,3]
                                LHS[p,q,i,j,3,3] += mu_pq[p,q]*Lr_cplus[i,j,3,3] + ksi_p[p]*Lz_cplus[i,j,3,3]
                                LHS[p,q,i,j,3,0] += mu_pq[p,q]*Lr_cplus[i,j,3,0] + ksi_p[p]*Lz_cplus[i,j,3,0]
                            else:
                                #psi_inc1[j,0] = psi_bc[jstart,0]
                                #psi_inc1[j,3] = psi_bc[jstart,3]
                                RHS[p,q,i,j,0] -= ((mu_pq[p,q]*Lr_cminus[i,j,0,0]+ksi_p[p]*Lz_cminus[i,j,0,0])*psi_inc1[j,0] 
                                        + (mu_pq[p,q]*Lr_cminus[i,j,0,3]+ksi_p[p]*Lz_cminus[i,j,0,3])*psi_inc1[j,3])
                                RHS[p,q,i,j,3] -= ((mu_pq[p,q]*Lr_cplus[i,j,3,3]+ksi_p[p]*Lz_cplus[i,j,3,3])*psi_inc1[j,3] 
                                        + (mu_pq[p,q]*Lr_cplus[i,j,3,0]+ksi_p[p]*Lz_cplus[i,j,3,0])*psi_inc1[j,0])
                                LHS[p,q,i,j,1,1] += mu_pq[p,q]*Lr_cplus[i,j,1,1] + ksi_p[p]*Lz_cplus[i,j,1,1]
                                LHS[p,q,i,j,1,2] += mu_pq[p,q]*Lr_cplus[i,j,1,2] + ksi_p[p]*Lz_cplus[i,j,1,2]
                                LHS[p,q,i,j,2,2] += mu_pq[p,q]*Lr_cminus[i,j,2,2] + ksi_p[p]*Lz_cminus[i,j,2,2]
                                LHS[p,q,i,j,2,1] += mu_pq[p,q]*Lr_cminus[i,j,2,1] + ksi_p[p]*Lz_cminus[i,j,2,1]
                            
                            if spokecross[j] == 1:
                                RHS[p,q,i,j,0] -= ((mu_pq[p,q]*Lr_cplus[i,j,0,0]+ksi_p[p]*Lz_cplus[i,j,0,0])*psi_inc1[j,0] 
                                        + (mu_pq[p,q]*Lr_cplus[i,j,0,1]+ksi_p[p]*Lz_cplus[i,j,0,1])*psi_inc1[j,1])
                                RHS[p,q,i,j,1] -= ((mu_pq[p,q]*Lr_cminus[i,j,1,1]+ksi_p[p]*Lz_cminus[i,j,1,1])*psi_inc1[j,1] 
                                        + (mu_pq[p,q]*Lr_cminus[i,j,1,0]+ksi_p[p]*Lz_cminus[i,j,1,0])*psi_inc1[j,0])
                            else: 
                                LHS[p,q,i,j,0,0] += mu_pq[p,q]*Lr_cplus[i,j,0,0]+ksi_p[p]*Lz_cplus[i,j,0,0]
                                LHS[p,q,i,j,0,1] += mu_pq[p,q]*Lr_cplus[i,j,0,1]+ksi_p[p]*Lz_cplus[i,j,0,1]
                                LHS[p,q,i,j,1,1] += mu_pq[p,q]*Lr_cminus[i,j,1,1]+ksi_p[p]*Lz_cminus[i,j,1,1]
                                LHS[p,q,i,j,1,0] += mu_pq[p,q]*Lr_cminus[i,j,1,0]+ksi_p[p]*Lz_cminus[i,j,1,0]
                            
                            if spokecross[j+1] == 0:
                                RHS[p,q,i,j,2] -= ((mu_pq[p,q]*Lr_cplus[i,j,2,2]+ksi_p[p]*Lz_cplus[i,j,2,2])*psi_inc1[j,2] 
                                        + (mu_pq[p,q]*Lr_cplus[i,j,2,3]+ksi_p[p]*Lz_cplus[i,j,2,3])*psi_inc1[j,3])
                                RHS[p,q,i,j,3] -= ((mu_pq[p,q]*Lr_cminus[i,j,3,3]+ksi_p[p]*Lz_cminus[i,j,3,3])*psi_inc1[j,3] 
                                        + (mu_pq[p,q]*Lr_cminus[i,j,3,2]+ksi_p[p]*Lz_cminus[i,j,3,2])*psi_inc1[j,2])
                            else: 
                                LHS[p,q,i,j,2,2] += mu_pq[p,q]*Lr_cplus[i,j,2,2]+ksi_p[p]*Lz_cplus[i,j,2,2]
                                LHS[p,q,i,j,2,3] += mu_pq[p,q]*Lr_cplus[i,j,2,3]+ksi_p[p]*Lz_cplus[i,j,2,3]
                                LHS[p,q,i,j,3,3] += mu_pq[p,q]*Lr_cminus[i,j,3,3]+ksi_p[p]*Lz_cminus[i,j,3,3]
                                LHS[p,q,i,j,3,2] += mu_pq[p,q]*Lr_cminus[i,j,3,2]+ksi_p[p]*Lz_cminus[i,j,3,2]
                            #solve for psi_half
                            psi[p,q,i,j,:] = np.dot(np.linalg.inv(LHS[p,q,i,j,:,:]),RHS[p,q,i,j,:])
                            psi_half[p,q+1,i,j,:] = ((1/tau[p,q])*psi[p,q,i,j,:]-
                                          ((1-tau[p,q])/tau[p,q])*psi_half[p,q,i,j,:])
                            loop+=1
                        if loop <((Ni*Nj)-1):   
                            if j<= jstart:
                                if j > 0:
                                    jj = j-1
                                else:
                                    jj = jstart+1
                            else:
                                jj = j+1
                            if (mu_pq[p,q]*np.cos(theta[jj])+ksi_p[p]*np.sin(theta[jj]))<0:
                                istart = Ni-1
                                iend = -1
                                i_incr = -1
                            else:
                                istart = 0
                                iend = Ni
                                i_incr = 1
                            if i_incr == -1:
                                psi_inc1[jj,1] = psi[p,q,i,j,0]
                                psi_inc1[jj,2] = psi[p,q,i,j,3]
                            else:
                                psi_inc1[jj,0] = psi[p,q,i,j,1]
                                psi_inc1[jj,3] = psi[p,q,i,j,2]
                            if spokecross[jj] == 0:
                                psi_inc1[jj,2] = psi[p,q,i,j,1]
                                psi_inc1[jj,3] = psi[p,q,i,j,0]
                            elif spokecross[jj+1] == 1:
                                psi_inc1[jj,0] = psi[p,q,i,j,3]
                                psi_inc1[jj,1] = psi[p,q,i,j,2]
                        else:
                            break
    
                if mu_pq[p,q] > 0:
                    need = np.ones((Ni,Nj))
                    for j in range(Nj+1):
                        #spokecross
                        if (mu_pq[p,q]*np.sin(theta_half[j])-ksi_p[p]*np.cos(theta_half[j]))<0:
                            spokecross[j] =1
                        else:
                            spokecross[j] =0   
                    #loop over all cells
                    for i in range(Ni):
                        for j in range(Nj):
                            if (mu_pq[p,q]*np.sin(theta_half[j])-ksi_p[p]*np.cos(theta_half[j]))<0:
                                if j != 0:
                                    need[i,j] += 1
                                    #print("for i=",i," and j=",j," 1+ added")
                            elif (-mu_pq[p,q]*np.sin(theta_half[j+1])+ksi_p[p]*np.cos(theta_half[j+1]))<0:
                                if j != Nj-1:
                                    need[i,j] += 1
                                    #print("for i=",i," and j=",j," 3+ added")
                            #correct for center points    
                            if i == 0:
                                if (mu_pq[p,q]*np.cos(theta[j])+ksi_p[p]*np.sin(theta[j]))>0:
                                    need[i,j] -= 1
                                    #print("for i=",i," and j=",j," 2+ subtracted")
                            #looping over boundary cells
                            elif i == Ni-1:
                                if (mu_pq[p,q]*np.cos(theta[j])+ksi_p[p]*np.sin(theta[j]))<0:
                                    need[i,j] -= 1
                                    #print("for i=",i," and j=",j," 2+ subtracted")
                            #start sweeping
                            if need[i,j] == 0:
                                jstart = j
                                if (mu_pq[p,q]*np.cos(theta[jstart])+ksi_p[p]*np.sin(theta[jstart]))<0:
                                    istart = Ni-1
                                    iend = -1
                                    i_incr = -1
                                else:
                                    istart = 0
                                    iend = Ni
                                    i_incr = 1
                    j = jstart
                    jj = jstart
                    loop = 0
                    while loop <((Ni*Nj)-1):
                        j = jj
                        psi_inc1[j,:] = psi_inc1[jj,:]
                        for i in range(istart,iend,i_incr):
                            alpha[p,q+1] = alpha[p,q]-w_pq[p,q]*mu_pq[p,q]
                            tau[p,q] = (mu_pq[p,q]-mu_pqhalf[p,q])/(mu_pqhalf[p,q+1]-mu_pqhalf[p,q])
                            RHS[p,q,i,j,:] = (((1/(2*np.pi))*np.dot(T_mat[i,j,:,:],(Sig_s*phi[i,j,:]+Q)))
                                                +(np.dot(R_mat[i,j,:,:],(alpha[p,q+1]*((1-tau[p,q])/tau[p,q])
                                                       +alpha[p,q])*psi_half[p,q,i,j,:]/w_pq[p,q])))
                                             
                            LHS[p,q,i,j,:,:] = (mu_pq[p,q]*Kr_mat[i,j,:,:]
                                                +ksi_p[p]*Kz_mat[i,j,:,:]+Sig_t*T_mat[i,j,:,:]
                                                +R_mat[i,j,:,:]*(alpha[p,q+1]/(w_pq[p,q]*tau[p,q])))
                            
                            if (i_incr == -1):
                                RHS[p,q,i,j,1] -= ((mu_pq[p,q]*Lr_cplus[i,j,1,1]+ksi_p[p]*Lz_cplus[i,j,1,1])*psi_inc1[j,1] 
                                            + (mu_pq[p,q]*Lr_cplus[i,j,1,2]+ksi_p[p]*Lz_cplus[i,j,1,2])*psi_inc1[j,2])
                                RHS[p,q,i,j,2] -= ((mu_pq[p,q]*Lr_cminus[i,j,2,2]+ksi_p[p]*Lz_cminus[i,j,2,2])*psi_inc1[j,2] 
                                            + (mu_pq[p,q]*Lr_cminus[i,j,2,1]+ksi_p[p]*Lz_cminus[i,j,2,1])*psi_inc1[j,1])
                                LHS[p,q,i,j,0,0] += mu_pq[p,q]*Lr_cminus[i,j,0,0] + ksi_p[p]*Lz_cminus[i,j,0,0]
                                LHS[p,q,i,j,0,3] += mu_pq[p,q]*Lr_cminus[i,j,0,3] + ksi_p[p]*Lz_cminus[i,j,0,3]
                                LHS[p,q,i,j,3,3] += mu_pq[p,q]*Lr_cplus[i,j,3,3] + ksi_p[p]*Lz_cplus[i,j,3,3]
                                LHS[p,q,i,j,3,0] += mu_pq[p,q]*Lr_cplus[i,j,3,0] + ksi_p[p]*Lz_cplus[i,j,3,0]
                            else:
                                #psi_inc1[j,0] = psi_bc[jstart,0]
                                #psi_inc1[j,3] = psi_bc[jstart,3]
                                RHS[p,q,i,j,0] -= ((mu_pq[p,q]*Lr_cminus[i,j,0,0]+ksi_p[p]*Lz_cminus[i,j,0,0])*psi_inc1[j,0] 
                                        + (mu_pq[p,q]*Lr_cminus[i,j,0,3]+ksi_p[p]*Lz_cminus[i,j,0,3])*psi_inc1[j,3])
                                RHS[p,q,i,j,3] -= ((mu_pq[p,q]*Lr_cplus[i,j,3,3]+ksi_p[p]*Lz_cplus[i,j,3,3])*psi_inc1[j,3] 
                                        + (mu_pq[p,q]*Lr_cplus[i,j,3,0]+ksi_p[p]*Lz_cplus[i,j,3,0])*psi_inc1[j,0])
                                LHS[p,q,i,j,1,1] += mu_pq[p,q]*Lr_cplus[i,j,1,1] + ksi_p[p]*Lz_cplus[i,j,1,1]
                                LHS[p,q,i,j,1,2] += mu_pq[p,q]*Lr_cplus[i,j,1,2] + ksi_p[p]*Lz_cplus[i,j,1,2]
                                LHS[p,q,i,j,2,2] += mu_pq[p,q]*Lr_cminus[i,j,2,2] + ksi_p[p]*Lz_cminus[i,j,2,2]
                                LHS[p,q,i,j,2,1] += mu_pq[p,q]*Lr_cminus[i,j,2,1] + ksi_p[p]*Lz_cminus[i,j,2,1]
                            
                            if spokecross[j] == 1:
                                RHS[p,q,i,j,0] -= ((mu_pq[p,q]*Lr_cplus[i,j,0,0]+ksi_p[p]*Lz_cplus[i,j,0,0])*psi_inc1[j,0] 
                                        + (mu_pq[p,q]*Lr_cplus[i,j,0,1]+ksi_p[p]*Lz_cplus[i,j,0,1])*psi_inc1[j,1])
                                RHS[p,q,i,j,1] -= ((mu_pq[p,q]*Lr_cminus[i,j,1,1]+ksi_p[p]*Lz_cminus[i,j,1,1])*psi_inc1[j,1] 
                                        + (mu_pq[p,q]*Lr_cminus[i,j,1,0]+ksi_p[p]*Lz_cminus[i,j,1,0])*psi_inc1[j,0])
                            else:
                                LHS[p,q,i,j,0,0] += mu_pq[p,q]*Lr_cplus[i,j,0,0]+ksi_p[p]*Lz_cplus[i,j,0,0]
                                LHS[p,q,i,j,0,1] += mu_pq[p,q]*Lr_cplus[i,j,0,1]+ksi_p[p]*Lz_cplus[i,j,0,1]
                                LHS[p,q,i,j,1,1] += mu_pq[p,q]*Lr_cminus[i,j,1,1]+ksi_p[p]*Lz_cminus[i,j,1,1]
                                LHS[p,q,i,j,1,0] += mu_pq[p,q]*Lr_cminus[i,j,1,0]+ksi_p[p]*Lz_cminus[i,j,1,0]
                            
                            if spokecross[j+1] == 0:
                                RHS[p,q,i,j,2] -= ((mu_pq[p,q]*Lr_cplus[i,j,2,2]+ksi_p[p]*Lz_cplus[i,j,2,2])*psi_inc1[j,2] 
                                        + (mu_pq[p,q]*Lr_cplus[i,j,2,3]+ksi_p[p]*Lz_cplus[i,j,2,3])*psi_inc1[j,3])
                                RHS[p,q,i,j,3] -= ((mu_pq[p,q]*Lr_cminus[i,j,3,3]+ksi_p[p]*Lz_cminus[i,j,3,3])*psi_inc1[j,3] 
                                        + (mu_pq[p,q]*Lr_cminus[i,j,3,2]+ksi_p[p]*Lz_cminus[i,j,3,2])*psi_inc1[j,2])
                            else:
                                LHS[p,q,i,j,2,2] += mu_pq[p,q]*Lr_cplus[i,j,2,2]+ksi_p[p]*Lz_cplus[i,j,2,2]
                                LHS[p,q,i,j,2,3] += mu_pq[p,q]*Lr_cplus[i,j,2,3]+ksi_p[p]*Lz_cplus[i,j,2,3]
                                LHS[p,q,i,j,3,3] += mu_pq[p,q]*Lr_cminus[i,j,3,3]+ksi_p[p]*Lz_cminus[i,j,3,3]
                                LHS[p,q,i,j,3,2] += mu_pq[p,q]*Lr_cminus[i,j,3,2]+ksi_p[p]*Lz_cminus[i,j,3,2]
                            #solve for psi_half  
                            psi[p,q,i,j,:] = np.dot(np.linalg.inv(LHS[p,q,i,j,:,:]),RHS[p,q,i,j,:])
                            psi_half[p,q+1,i,j,:] = ((1/tau[p,q])*psi[p,q,i,j,:]-
                                          ((1-tau[p,q])/tau[p,q])*psi_half[p,q,i,j,:])
                            loop+=1
                        if loop < (Ni*Nj)-1:
                            if j<= jstart:
                                if j > 0:
                                    jj = j-1
                                else:
                                    jj = jstart+1
                            else:
                                jj = j+1
                            if (mu_pq[p,q]*np.cos(theta[jj])+ksi_p[p]*np.sin(theta[jj]))<0:
                                istart = Ni-1
                                iend = -1
                                i_incr = -1
                            else:
                                istart = 0
                                iend = Ni
                                i_incr = 1
                            if i_incr == -1:
                                psi_inc1[jj,1] = psi[p,q,i,j,0]
                                psi_inc1[jj,2] = psi[p,q,i,j,3]
                            else:
                                psi_inc1[jj,0] = psi[p,q,i,j,1]
                                psi_inc1[jj,3] = psi[p,q,i,j,2]
                                #print(psi[p,q,i,j,:])
                            if spokecross[jj] == 0:
                                psi_inc1[jj,2] = psi[p,q,i,j,1]
                                psi_inc1[jj,3] = psi[p,q,i,j,0]
                            elif spokecross[jj+1] == 1:
                                psi_inc1[jj,0] = psi[p,q,i,j,3]
                                psi_inc1[jj,1] = psi[p,q,i,j,2]
                        else:
                            break
                            
                                
        phi = np.zeros(shape = (Ni,Nj,J))
        converged = True
        for i in range(Ni):
            for j in range(Nj):
                for k in range(J):
                    for s in range(ksi_p.size):
                        for q in range(Nazi):
                            phi[i,j,k] += w_pq[s,q]*psi[s,q,i,j,k]
                    if np.absolute(phi[i,j,k]-phi_old[i,j,k]) > tol*np.absolute(phi[i,j,k]):
                        converged = False
        count +=1
                      
        if converged == True:
            break
        if count > maxit:
            break
    
    return psi,phi