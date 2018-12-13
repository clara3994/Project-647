def PCd(q0,sig_a0,sig_s0,order,unc_par,Nq,method1):
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
    
    if method1 == 'tensor':
        Nq = 16
        ksia,wa = np.polynomial.hermite.hermgauss(Nq)
        ksia = np.sqrt(2)*ksia
        wa = (1/np.sqrt(np.pi))*wa
        ksib,wb = np.polynomial.hermite.hermgauss(Nq)
        ksib = np.sqrt(2)*ksib
        wb = (1/np.sqrt(np.pi))*wb
        ksis,ws = np.polynomial.hermite.hermgauss(Nq)
        ksis = np.sqrt(2)*ksis
        ws = (1/np.sqrt(np.pi))*ws
        
        
    else:
        if order == 3:
            f=open("GQN_d3_l3.asc","r")
        elif order == 4:
            f=open("GQN_d3_l4.asc","r")
        else:
            f=open("GQN_d3_l5.asc","r")
        lines=f.readlines()
        xia=[]
        xib=[]
        xis=[]
        w = []
        for x in lines:
            xia.append(x.split(',')[0])
            xib.append(x.split(',')[1])
            xis.append(x.split(',')[2])
            w.append(x.split(',')[3])
        f.close()
        xia = np.asarray(xia)
        ksia = list(map(float ,xia))
        ksia = np.asarray(ksia)
        xib = np.asarray(xib)
        ksib = list(map(float ,xib))
        ksib = np.asarray(ksib)
        xis = np.asarray(xis)
        ksis = list(map(float ,xis))
        ksis = np.asarray(ksis)
        w = np.asarray(w)
        w = list(map(float ,w))
        wa=w
        wb=w
        ws=w
        
    #building Hermite polynomials
    psia = np.zeros((ksia.size,order+1))
    for s in range(ksia.size):
        psia[s,0] = 1
        psia[s,1] = ksia[s]
        for i in range(2,order+1):
            psia[s,i] = ksia[s]*psia[s,i-1]-(i-1)*psia[s,i-2]   
    
    psib = np.zeros((ksib.size,order+1))
    for s in range(ksib.size):
        psib[s,0] = 1
        psib[s,1] = ksib[s]
        for i in range(2,order+1):
            psib[s,i] = ksib[s]*psib[s,i-1]-(i-1)*psib[s,i-2]   

    psis = np.zeros((ksis.size,order+1))
    for s in range(ksis.size):
        psis[s,0] = 1
        psis[s,1] = ksis[s]
        for i in range(2,order+1):
            psis[s,i] = ksis[s]*psis[s,i-1]-(i-1)*psis[s,i-2]    
    
    psi_0 = np.zeros(order+1)
    psi_0[0] = 1
    psi_0[1] = 0
    for i in range(2,order+1):
        psi_0[i] = 0*psi_0[i-1]-(i-1)*psi_0[i-2] 
    
    T = [] #PC coefficients for abs_rate
    Tm=[]  #PC moments for abs_rate
    PSI = [] # psi_q*psi_k*psi_L*psi_T evaluated at xi=0
    PSI_T = [] # psi_q*psi_k*psi_L*psi_T evaluated for all xi
    Coef=[] #coefficients at xi=0 for abs_rate
    T_p = [] #PC coefficients for scalar flux
    Tm_p =[]  #PC moments for scalar flux
    Coef_p=[] #coefficients at xi=0 for scalar flux
    A =[] #indices of a in tuple
    B=[] #indices of sig_a in tuple
    S=[] #indices of sig_s in tuple
    T_0 = 0 #sum of coefficients*psi at xi=0 for abs_rate
    T_0_p = 0 #sum of coefficients*psi at xi=0 for scalar flux

    for aa in range(order+1):
        for bb in range(order+1):
            for ss in range(order+1):
                if aa+bb+ss <= order:
                    A.append(aa)
                    B.append(bb)
                    S.append(ss)
                    psiCB = np.zeros(shape=(ksia.size,Np,Nazi,Ni,Nj,J))
                    phi_CB = np.zeros(shape=(ksia.size,Ni,Nj,J))
                    phiCB = np.zeros(shape=(ksia.size,Ni,Nj))
                    PHI = np.zeros(shape=(ksia.size,Ni,Nj))
                    sumq=0
                    sum_p=0
                    for n in range(ksia.size):
                        psi_total = psia[n,aa]*psib[n,bb]*psis[n,ss]
                        PSI_T.append(psi_total)
                        sig_s = sig_s0+sig_s1*ksis[n]
                        sig_a = sig_a0+sig_a1*ksib[n]
                        Sig_t = sig_a+sig_s
                        psi_inc = (a0+a1*ksia[n])*q0/Sig_t
                        psiCB[n],phi_CB[n] = Solver(method,Ni,Nj,Nazi,Np,R,J,sig_s,Sig_t,phi_old,q0,psi_inc,tol,maxit)
                        for i in range(Ni):
                            for j in range(Nj):
                                phiCB[n,i,j] = (1/4)*np.sum(np.dot(R_mat[i,j,:,:],phi_CB[n,i,j,:]))
                                PHI[n,i,j] = (1/4)*np.sum(phi_CB[n,i,j,:])
                        sumq += (wa[n]*psi_total*phiCB[n,1,1]*sig_a)
                        sum_p += (wa[n]*psi_total*PHI[n,1,1])
                        sumN = (wb[n]*psi_total*phiCB[n,1,1]*sig_a)
                        sumN_p = (wb[n]*psi_total*PHI[n,1,1])
                        coeffN = ((1/(math.factorial(aa)
                                     *math.factorial(bb)*math.factorial(ss)))*sumN)
                        coeffN_p = ((1/(math.factorial(aa)
                                     *math.factorial(bb)*math.factorial(ss)))*sumN_p)
                        Coef.append(coeffN)
                        Coef_p.append(coeffN_p)
                    psi_0_T = psi_0[aa]*psi_0[bb]*psi_0[ss]
                    PSI.append(psi_0_T)
                    coeff = ((1/(math.factorial(aa)
                                     *math.factorial(bb)*math.factorial(ss)))*sumq)
                    coeff_p = ((1/(math.factorial(aa)
                                     *math.factorial(bb)*math.factorial(ss)))*sum_p)

                    T.append(coeff)  
                    T_p.append(coeff_p)  
                    tm = (coeff*math.factorial(aa)
                                     *math.factorial(bb)*math.factorial(ss))
                    tm_p = (coeff_p*math.factorial(aa)
                                     *math.factorial(bb)*math.factorial(ss))
                    Tm.append(tm)
                    Tm_p.append(tm_p)
                    T_0 += coeff*psi_0_T
                    T_0_p += coeff_p*psi_0_T

                else:
                    pass
                    
    PSI_T = np.asarray(PSI_T)
    PSI_T = np.reshape(PSI_T,(len(A),ksia.size))
        
    #MC calculation
    sample = np.random.normal(size=N_mc)
    A_mc = np.zeros(shape=(N_mc,Ni,Nj))
    psiCBm = np.zeros(shape=(N_mc,Np,Nazi,Ni,Nj,J))
    phi_CBm = np.zeros(shape=(N_mc,Ni,Nj,J))
    phiCBm = np.zeros(shape=(N_mc,Ni,Nj))
    Phi_mc = np.zeros(shape=(N_mc,Ni,Nj))
    for s in range(sample.size): 
        sig_s = sig_s0+sig_s1*sample[s]
        sig_a = sig_a0+sig_a1*sample[s]
        Sig_t = sig_a+sig_s
        psi_inc = (a0+a1*sample[s])*q0/Sig_t
        psiCBm[s],phi_CBm[s] = Solver(method,Ni,Nj,Nazi,Np,R,J,sig_s,Sig_t,phi_old,q0,psi_inc,tol,maxit)
        for r in range(Ni):
            for j in range(Nj):
                phiCBm[s,r,j] = (1/4)*np.sum(np.dot(R_mat[r,j,:,:],phi_CBm[s,r,j,:]))
                Phi_mc[s,r,j] = (1/4)*np.sum(phi_CBm[s,r,j,:])
        A_mc[s,1,1] = sig_a0*phiCBm[s,1,1]
        
    #local sensitivity for rate of absorption
    Sa = 0
    Sb = 0
    Ss = 0
    for i in range(1,len(T)):
        Sa += (a0/a1)*(1/T_0)*T[i]*(A[i])*psi_0[A[i-1]]*psi_0[B[i]]*psi_0[S[i]]
        Sb += (sig_a0/sig_a1)*(1/T_0)*T[i]*(B[i])*psi_0[B[i-1]]*psi_0[S[i]]*psi_0[A[i]]
        Ss += (sig_s0/sig_s1)*(1/T_0)*T[i]*(S[i])*psi_0[S[i-1]]*psi_0[B[i]]*psi_0[A[i]]
        
    #local sensitivity for flux
    Sa_p = 0
    Sb_p = 0
    Ss_p = 0
    for i in range(1,len(T_p)):
        Sa_p += (a0/a1)*(1/T_0_p)*T_p[i]*(A[i])*psi_0[A[i-1]]*psi_0[B[i]]*psi_0[S[i]]
        Sb_p += (sig_a0/sig_a1)*(1/T_0_p)*T_p[i]*(B[i])*psi_0[B[i-1]]*psi_0[S[i]]*psi_0[A[i]]
        Ss_p += (sig_s0/sig_s1)*(1/T_0_p)*T_p[i]*(S[i])*psi_0[S[i-1]]*psi_0[B[i]]*psi_0[A[i]]
                              
    return T,Tm,A_mc,T_p,Tm_p,Phi_mc,Sa,Sb,Ss,Sa_p,Sb_p,Ss_p
                              
sig_s0 = 2.0
sig_a0 = 5.0
Sig_t = sig_s0+sig_a0
q0 = 2*np.pi*sig_a0*psi_inc
a0 = psi_inc*Sig_t/q0
order = 4
unc_par = 'a'
Nq=5
                              
T,Tm,A_mc,T_p,Tm_p,Phi_mc,Sa,Sb,Ss,Sa_p,Sb_p,Ss_p = PCd(q0,sig_a0,sig_s0,order,unc_par,Nq,'tensor')
plt.hist(A_mc[:,1,1],100)
plt.savefig('absr.jpg', dpi=1800)
plt.show()
plt.hist(Phi_mc[:,1,1],100)
plt.savefig('phi.jpg', dpi=1800)
plt.show()
print(Sa,Sb,Ss,Sa_p,Sb_p,Ss_p)

Tsp,Tmsp,A_mcsp,T_psp,Tm_psp,Phi_mcsp,Sa1,Sb1,Ss1,Sa_p1,Sb_p1,Ss_p1 = PCd(q0,sig_a0,sig_s0,order,unc_par,Nq,'sparse')
plt.hist(A_mcsp[:,1,1],100)
plt.savefig('absr_sp.jpg', dpi=1800)
plt.show()
plt.hist(Phi_mcsp[:,1,1],100)
plt.savefig('phi_sp.jpg', dpi=1800)
plt.show()
print(Sa1,Sb1,Ss1,Sa_p1,Sb_p1,Ss_p1)