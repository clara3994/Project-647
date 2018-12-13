def PC(q0,sig_a0,sig_s0,order,unc_par,Nq):
    #input parameters for black box code
    method = "CB"
    tol = 10**(-6)
    psi_inc = 3.0
    q0 = 2*np.pi*sig_a0*psi_inc
    Sig_t = sig_s0+sig_a0
    Ni = 3
    Nj = 4
    R = 4
    J = 4
    Nazi = 2
    Np = 2
    maxit = 1000
    phi_old = np.zeros(shape = (Ni,Nj,J))
    T_mat,Kr_mat,Kz_mat,R_mat,Lr_cminus,Lr_cplus,Lz_cminus,Lz_cplus,theta,theta_half,r,z,r_k,z_k = CB(Ni,Nj,Nazi,Np,R,J) 
    N_mc = 5000
    
    #standard deviation
    a1 = a0/10
    sig_a1 = sig_a0/10
    sig_s1 = sig_s0/10
    
    #quadrature points and weights
    ksi1,w = np.polynomial.hermite.hermgauss(Nq)
    ksi = np.sqrt(2)*ksi1
    
    #building Hermite polynomials
    psi = np.zeros((ksi.size,order+1))
    for s in range(ksi.size):
        psi[s,0] = 1
        psi[s,1] = ksi[s]
        for i in range(2,order+1):
            psi[s,i] = ksi[s]*psi[s,i-1]-(i-1)*psi[s,i-2] 
    
    #calculate T using analytic expression
    if unc_par == 'a':
        Abs_Rate = np.zeros(shape=(order+1,Ni,Nj))
        PHI = np.zeros(shape=(order+1,Ni,Nj))
        psiCB = np.zeros(shape=(Nq,Np,Nazi,Ni,Nj,J))
        phi_CB = np.zeros(shape=(Nq,Ni,Nj,J))
        phiCB = np.zeros(shape=(Nq,Ni,Nj))
        for i in range(order+1):
            for s in range(ksi.size): 
                psi_inc = (a0+a1*ksi[s])*q0/Sig_t
                psiCB[s],phi_CB[s] = Solver(method,Ni,Nj,Nazi,Np,R,J,sig_s0,Sig_t,phi_old,q0,psi_inc,tol,maxit)
                for r in range(Ni):
                    for j in range(Nj):
                        phiCB[s,r,j] = (1/4)*np.sum(np.dot(R_mat[r,j,:,:],phi_CB[s,r,j,:]))
                Abs_Rate[i,:,:] += ((1/(np.sqrt(np.pi)*math.factorial(i)))
                                      *w[s]*psi[s,i]*sig_a0*phiCB[s,:,:])
                PHI[i,:,:] += ((1/(np.sqrt(np.pi)*math.factorial(i)))
                                 *w[s]*psi[s,i]*phiCB[s,:,:])
        #moments
        moments_abs = np.zeros(shape=(4,Ni,Nj))
        moments_phi = np.zeros(shape=(4,Ni,Nj))
        for i in range(4):
            moments_abs[i,:,:] += Abs_Rate[i,:,:]*math.factorial(i)
            moments_phi[i,:,:] += PHI[i,:,:]*math.factorial(i)
            
        #MC
        sample = np.random.normal(size=N_mc)
        A_mc = np.zeros(shape=(N_mc,Ni,Nj))
        psiCBm = np.zeros(shape=(N_mc,Np,Nazi,Ni,Nj,J))
        phi_CBm = np.zeros(shape=(N_mc,Ni,Nj,J))
        phiCBm = np.zeros(shape=(N_mc,Ni,Nj))
        Phi_mc = np.zeros(shape=(N_mc,Ni,Nj))
        for s in range(sample.size): 
            psi_inc = (a0+a1*sample[s])*q0/Sig_t
            psiCBm[s],phi_CBm[s] = Solver(method,Ni,Nj,Nazi,Np,R,J,sig_s0,Sig_t,phi_old,q0,psi_inc,tol,maxit)
            for r in range(Ni):
                for j in range(Nj):
                    phiCBm[s,r,j] = (1/4)*np.sum(np.dot(R_mat[r,j,:,:],phi_CBm[s,r,j,:]))
                    Phi_mc[s,r,j] = (1/4)*np.sum(phi_CBm[s,r,j,:])
            A_mc[s,:,:] = sig_a0*phiCBm[s,:,:]
            
            
    elif unc_par == 'sig_a':
        Abs_Rate = np.zeros(shape=(order+1,Ni,Nj))
        PHI = np.zeros(shape=(order+1,Ni,Nj))
        psiCB = np.zeros(shape=(Nq,Np,Nazi,Ni,Nj,J))
        phi_CB = np.zeros(shape=(Nq,Ni,Nj,J))
        phiCB = np.zeros(shape=(Nq,Ni,Nj))
        for i in range(order+1):
            for s in range(ksi.size): 
                sig_a = sig_a0+sig_a1*ksi[s]
                Sig_t = sig_s0+sig_a
                q = 2*np.pi*sig_a*psi_inc
                psiCB[s],phi_CB[s] = Solver(method,Ni,Nj,Nazi,Np,R,J,sig_s0,Sig_t,phi_old,q,psi_inc,tol,maxit)
                for r in range(Ni):
                    for j in range(Nj):
                        phiCB[s,r,j] = (1/4)*np.sum(np.dot(R_mat[r,j,:,:],phi_CB[s,r,j,:]))
                Abs_Rate[i,:,:] += ((1/(np.sqrt(np.pi)*math.factorial(i)))
                            *w[s]*psi[s,i]*sig_a*phiCB[s,:,:])
                PHI[i,:,:] += ((1/(np.sqrt(np.pi)*math.factorial(i)))
                                 *w[s]*psi[s,i]*phiCB[s,:,:])
        #moments
        moments_abs = np.zeros(shape=(4,Ni,Nj))
        moments_phi = np.zeros(shape=(4,Ni,Nj))
        for i in range(4):
            moments_abs[i,:,:] += Abs_Rate[i,:,:]*math.factorial(i)
            moments_phi[i,:,:] += PHI[i,:,:]*math.factorial(i)
            
        #MC
        sample = np.random.normal(size=N_mc)
        A_mc = np.zeros(shape=(N_mc,Ni,Nj))
        psiCBm = np.zeros(shape=(N_mc,Np,Nazi,Ni,Nj,J))
        phi_CBm = np.zeros(shape=(N_mc,Ni,Nj,J))
        phiCBm = np.zeros(shape=(N_mc,Ni,Nj))
        Phi_mc = np.zeros(shape=(N_mc,Ni,Nj))
        for s in range(sample.size): 
            sig_a = sig_a0+sig_a1*sample[s]
            Sig_t = sig_s0+sig_a
            q = 2*np.pi*sig_a*psi_inc
            psiCBm[s],phi_CBm[s] = Solver(method,Ni,Nj,Nazi,Np,R,J,sig_s0,Sig_t,phi_old,q,psi_inc,tol,maxit)
            for r in range(Ni):
                for j in range(Nj):
                    phiCBm[s,r,j] = (1/4)*np.sum(np.dot(R_mat[r,j,:,:],phi_CBm[s,r,j,:]))
                    Phi_mc[s,r,j] = (1/4)*np.sum(phi_CBm[s,r,j,:])
            A_mc[s,:,:] = sig_a0*phiCBm[s,:,:]
            
    elif unc_par == 'sig_s':
        Abs_Rate = np.zeros(shape=(order+1,Ni,Nj))
        PHI = np.zeros(shape=(order+1,Ni,Nj))
        psiCB = np.zeros(shape=(Nq,Np,Nazi,Ni,Nj,J))
        phi_CB = np.zeros(shape=(Nq,Ni,Nj,J))
        phiCB = np.zeros(shape=(Nq,Ni,Nj))
        for i in range(order+1):
            for s in range(ksi.size): 
                sig_s = sig_s0+sig_s1*ksi[s]
                Sig_t = sig_a0+sig_s
                q = 2*np.pi*sig_a0*psi_inc
                psiCB[s],phi_CB[s] = Solver(method,Ni,Nj,Nazi,Np,R,J,sig_s,Sig_t,phi_old,q,psi_inc,tol,maxit)
                for r in range(Ni):
                    for j in range(Nj):
                        phiCB[s,r,j] = (1/4)*np.sum(np.dot(R_mat[r,j,:,:],phi_CB[s,r,j,:]))
                Abs_Rate[i,:,:] += ((1/(np.sqrt(np.pi)*math.factorial(i)))
                            *w[s]*psi[s,i]*sig_a0*phiCB[s,:,:])
                PHI[i,:,:] += ((1/(np.sqrt(np.pi)*math.factorial(i)))
                                 *w[s]*psi[s,i]*phiCB[s,:,:])
        #moments
        moments_abs = np.zeros(shape=(4,Ni,Nj))
        moments_phi = np.zeros(shape=(4,Ni,Nj))
        for i in range(4):
            moments_abs[i,:,:] += Abs_Rate[i,:,:]*math.factorial(i)
            moments_phi[i,:,:] += PHI[i,:,:]*math.factorial(i)
            
        #MC
        sample = np.random.normal(size=N_mc)
        A_mc = np.zeros(shape=(N_mc,Ni,Nj))
        psiCBm = np.zeros(shape=(N_mc,Np,Nazi,Ni,Nj,J))
        phi_CBm = np.zeros(shape=(N_mc,Ni,Nj,J))
        phiCBm = np.zeros(shape=(N_mc,Ni,Nj))
        Phi_mc = np.zeros(shape=(N_mc,Ni,Nj))
        for s in range(sample.size): 
            sig_s = sig_s0+sig_s1*sample[s]
            Sig_t = sig_a0+sig_s
            q = 2*np.pi*sig_a0*psi_inc
            psiCBm[s],phi_CBm[s] = Solver(method,Ni,Nj,Nazi,Np,R,J,sig_s0,Sig_t,phi_old,q,psi_inc,tol,maxit)
            for r in range(Ni):
                for j in range(Nj):
                    phiCBm[s,r,j] = (1/4)*np.sum(np.dot(R_mat[r,j,:,:],phi_CBm[s,r,j,:]))
                    Phi_mc[s,r,j] = (1/4)*np.sum(phi_CBm[s,r,j,:])
            A_mc[s,:,:] = sig_a0*phiCBm[s,:,:]
            
    return Abs_Rate,moments_abs,A_mc,PHI,moments_phi,Phi_mc

sig_s0 = 2.0
sig_a0 = 5.0
Sig_t = sig_s0+sig_a0
q0 = 2*np.pi*sig_a0*psi_inc
a0 = psi_inc*Sig_t/q0
order = 3
unc_par = 'a'
Nq=4


Abs_Rate,moments_abs,A_mc,PHI,moments_phi,Phi_mc = PC(a0,sig_a0,sig_s0,order,unc_par,Nq)
print(Abs_Rate[:,1,1])
print(moments_abs)
plt.hist(A_mc[:,1,1],100)
plt.savefig('absr_a.jpg', dpi=1800)
plt.show()
print(PHI[:,1,1])
print(moments_phi)
plt.hist(Phi_mc[:,1,1],100)
plt.savefig('phi_a.jpg', dpi=1800)
plt.show()


Abs_Ratea,moments_absa,A_mca,PHIa,moments_phia,Phi_mca = PC(a0,sig_a0,sig_s0,order,'sig_a',Nq)
print(Abs_Ratea[:,1,1])
print(moments_absa)
plt.hist(A_mca[:,1,1],100)
plt.savefig('absr_b.jpg', dpi=1800)
plt.show()
print(PHIa[:,1,1])
print(moments_phia)
plt.hist(Phi_mca[:,1,1],100)
plt.savefig('phi_b.jpg', dpi=1800)
plt.show()

Abs_Rates,moments_abss,A_mcs,PHIs,moments_phis,Phi_mcs = PC(a0,sig_a0,sig_s0,order,'sig_s',Nq)
print(Abs_Rates[:,1,1])
print(moments_abss)
plt.hist(A_mcs[:,1,1],100)
plt.savefig('absr_s.jpg', dpi=1800)
plt.show()
print(PHIs[:,1,1])
print(moments_phis)
plt.hist(Phi_mcs[:,1,1],100)
plt.savefig('phi_s.jpg', dpi=1800)
plt.show()