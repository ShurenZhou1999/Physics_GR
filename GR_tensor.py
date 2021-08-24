'''
Author: Shuren Zhou
This code references the code by zhiqihuang for the course of General Relativity.
The website is http://zhiqihuang.top/gr/lectures.php .
'''

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from IPython.display import Latex
from IPython.display import display, Math

dim = 4

# dimension = 4
def product_inner(a, b):
    return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] )



'''     some tensor defined in the General relativeity theory    '''

# \g^i_j
def Metric_mixed(g_down, g_up):
    g_mixed = sym.MutableDenseNDimArray.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            g_mixed[i, j] = product_inner( g_up[i, :], g_down[:, j] )
    return g_mixed

# \Gamma_{i, j, k}
def Connection_down(g_down, x):
    gamma_down = sym.MutableDenseNDimArray.zeros(dim, dim, dim)
    for i in range(dim):
        for j in range(dim):
            for k in range(j+1):
                gamma_down[i, j, k] = ( sym.diff(g_down[i, j], x[k]) + sym.diff(g_down[i, k], x[j]) - \
                                       sym.diff(g_down[j, k] , x[i]) )/2
                if k != j:
                    gamma_down[i, k, j] = gamma_down[i, j, k]
    return gamma_down

# \Gamma^i_{j, k}
def Connection_mixed(gamma_down, g_up):
    gamma_mixed = sym.MutableDenseNDimArray.zeros(dim, dim, dim)
    for i in range(dim):
        for j in range(dim):
            for k in range(j+1):
                gamma_mixed[i, j, k] = product_inner( g_up[i, :], gamma_down[:, j, k] )
                if k != j:
                    gamma_mixed[i, k, j] = gamma_mixed[i, j, k]
    return gamma_mixed

# \R^i_{j, k, l}
def Riemann_tensor(gamma_mixed, x):
    riemann_tensor = sym.MutableDenseNDimArray.zeros(dim, dim, dim, dim)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(k+1):
                    term3 = product_inner(gamma_mixed[:, j, k], gamma_mixed[i, :, l])
                    term4 = product_inner(gamma_mixed[:, j, l], gamma_mixed[i, :, k])
                    riemann_tensor[i, j, k, l] = sym.diff(gamma_mixed[i, j, k], x[l]) - \
                                    sym.diff(gamma_mixed[i, j, l], x[k]) + term3 - term4
                    if l!=k:
                        riemann_tensor[i, j, l, k] = - riemann_tensor[i, j, k, l]
    return riemann_tensor

# \R_{i, j}
def Ricci_down(riemann_tensor):
    ricci_down = sym.MutableDenseNDimArray.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            ricci_down[i, j] = riemann_tensor[0, i, j, 0] + riemann_tensor[1, i, j, 1] + \
                                    riemann_tensor[2, i, j, 2] + riemann_tensor[3, i, j, 3]
    return ricci_down

# \R^i_j
def Ricci_mixed(ricci_down, g_up):
    ricci_mixed = sym.MutableDenseNDimArray.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            ricci_mixed[i, j] = product_inner( g_up[i, :], ricci_down[:, j] )
    return ricci_mixed

# \R^{i, j}
def Ricci_up(ricci_mixed, g_up):
    ricci_up = sym.MutableDenseNDimArray.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            ricci_up[i, j] = product_inner( g_up[i, :], ricci_mixed[j, :] )
    return ricci_up

# R
def Ricci_scalar(ricci_down, g_up):
    R = 0
    for i in range(dim):
        for j in range(dim):
            R += g_up[i, j]*ricci_down[i, j]
    return R

# Einstein tensor
# G_{i, j}
def Einstein_down(ricci_down, R, g_down):
    einstein_down = sym.MutableDenseNDimArray.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            einstein_down[i, j] = ricci_down[i, j] - g_down[i, j]*R/2
    return einstein_down

# G^i_j
def Einstein_mixed(ricci_mixed, R, g_mixed):
    einstein_mixed = sym.MutableDenseNDimArray.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            einstein_mixed[i, j] = ricci_mixed[i, j] - g_mixed[i, j]*R/2
    return einstein_mixed




'''        the standard Einstein field equations        '''

def Einstein_equations(Gij, Tij):
    G = sym.symbols('G')
    count = 0
    for ii in range(dim):
        for jj in range(ii+1):
            equs = Gij[ii, jj] - 8*sym.pi*G*Tij[ii, jj]
            equs = sym.simplify(equs)
            if equs!= 0:
                print('\n\n Equation '+str(count)+', for index(i,j) = '+str(ii)+' '+str(jj)+':')
                count+=1
                display(Math( sym.latex(equs) + r"\;\;=\;\; 0" ))





def Print_Connection(gamma_mixed, style = 0):
    print('\n')
    display(Math(r'Connection \,\, \Gamma^{i}_{\,jk} \, :'))
    for ii in range(dim):
        for jj in range(dim):
            for kk in range(jj+1):
                if gamma_mixed[ii, jj, kk] != 0:
                    if style == 0:
                        display(Math(  r'\Gamma ^{\,'+str(ii)+ '}_{\,\,' +str(jj)+' ' +str(kk)+'}\:\,\,=\;\,' + str(sym.simplify(gamma_mixed[ii, jj, kk]))  ))
                    else:
                        display(Math(  r'\Gamma ^{\,'+str(ii)+ '}_{\,\,' +str(jj)+' ' +str(kk)+'}\;\;=' ))
                        display(sym.simplify(gamma_mixed[ii, jj, kk]) )

def Print_Ricci_tensor(ricci_mixed):
    print('\n')
    display(Math(r'Ricci \;tensor \,\, R^{i}_{\,j} \, :'))
    for ii in range(dim):
        for jj in range(ii+1):
            if ricci_mixed[ii, jj] != 0:
                #display(Math(  r' R ^'+str(ii)+ '_{\;' +str(jj) +'}\:\,\,=\;\,' + str(sym.simplify(ricci_mixed[ii, jj]))  ))
                display(Math(  r' R ^'+str(ii)+ '_{\;' +str(jj) +'} ' ) )
                display(sym.simplify(ricci_mixed[ii, jj]) )
            
def Print_Ricci_scalar(R):
    print('\n\nRicci scalar, R =  ')            
    display(sym.simplify(R))
    
def Print_Einstein_tensor(einstein_mixed):
    print('\n')
    display(Math(r'Einstein \;tensor \,\, G^{i}_{\,j} \, :'))
    for ii in range(dim):
        for jj in range(ii+1):
            if einstein_mixed[ii, jj] != 0:
                #display(Math(  r' G ^'+str(ii)+ '_{\;' +str(jj) +'}\:\,\,=\;\,' + str(sym.simplify(einstein_mixed[ii, jj]))  ))
                display(Math(  r' G ^'+str(ii)+ '_{\;' +str(jj) +'} ' ) )
                display(sym.simplify(einstein_mixed[ii, jj]) )





class GR_tensor:
    
    def __init__(self, x, g_down):
        
        # coordinate
        self.x = x
        
        # \g_{i, j}
        self.g_down = g_down
        
        # \g^{i, j}
        self.g_up = g_down**(-1)

        # \g^i_j
        self.g_mixed = Metric_mixed(self.g_down, self.g_up)

        # \Gamma_{i, j, k}
        self.gamma_down = Connection_down(self.g_down, x)

        # \Gamma^i_{j, k}
        self.gamma_mixed = Connection_mixed(self.gamma_down, self.g_up)

        # \R^i_{j, k, l}
        self.riemann_tensor = Riemann_tensor(self.gamma_mixed, x)

        # \R_{i, j}
        self.ricci_down = Ricci_down(self.riemann_tensor)

        # \R^i_j
        self.ricci_mixed = Ricci_mixed(self.ricci_down, self.g_up)

        # \R^{i, j}
        self.ricci_up = Ricci_up(self.ricci_mixed, self.g_up)

        # R
        ricci_scalar = Ricci_scalar(self.ricci_down, self.g_up)
        self.ricci_scalar = sym.simplify(ricci_scalar)

        # \G_{i, j}
        self.einstein_down = Einstein_down(self.ricci_down, self.ricci_scalar, self.g_down)

        # \G^{i}_{j}
        self.einstein_mixed = Einstein_mixed(self.ricci_mixed, self.ricci_scalar, self.g_mixed)
        
    def print_Connection(self):
        Print_Connection(self.gamma_mixed)
        
    def print_Ricci_tensor(self):
        Print_Ricci_tensor(self.ricci_mixed)
        
    def print_Ricci_scalar(self):
        Print_Ricci_scalar(self.ricci_scalar)
        
    def print_Einstein_tensor(self):
        Print_Einstein_tensor(self.einstein_mixed)
        
    def print_Einstein_equations(self, Tij):
        Einstein_equations(self.einstein_down, Tij)





t, r, theta, phi = sym.symbols('t, r, theta, phi')
k, G, M = sym.symbols('k, G, M')
GM = G*M
x = [t, r, theta, phi]


'''                          metric                             '''

#########################  FRW metric  ######################### 
a = sym.Function('a' )
g_down1 = sym.diag( 1, -a(x[0])**2/(1-k*r**2), -a(x[0])**2 *r**2, -a(x[0])**2 *r**2 *sym.sin(theta)**2)

################# spherical symmetric metric ###################
g_down2 = sym.diag(1-2*GM/r, 1/(1-2*GM/r), -r**2, -r**2*sym.sin(theta))

##################### Schwarzschild metric #####################
Phi = sym.Function('Phi')
Psi = sym.Function('Psi')
g_down3 = sym.diag( sym.exp(2*Phi(x[0], x[1])), -sym.exp(-2*Psi(x[0], x[1])) , -x[1]**2, -x[1]**2 *sym.sin(x[2]) **2 )

######################   Kerr black hole  ######################
J = sym.symbols('J')
rho_squared = r**2 + J**2 *sym.cos(theta)**2
Delta = r**2 - 2*r*GM + J**2
g_down4 = sym.diag( 1-2*GM*r/rho_squared, -rho_squared/Delta, -rho_squared,  \
                  -(r**2 +J**2 +2*GM*r*J**2*sym.sin(theta)**2/rho_squared)*sym.sin(theta)**2)
g_down4[3, 0] = 2*GM*r*J*sym.sin(theta)**2/rho_squared
g_down4[0, 3] = g_down4[3, 0]

#################  Kerr black hole with charge  ###############
Q = sym.symbols('Q')
Delta += G*Q**2
g_down5 = sym.diag( 1-2*GM*r/rho_squared, -rho_squared/Delta, -rho_squared,  \
                  -(r**2 +J**2 +2*GM*r*J**2*sym.sin(theta)**2/rho_squared)*sym.sin(theta)**2)
g_down5[3, 0] = 2*GM*r*J*sym.sin(theta)**2/rho_squared
g_down5[0, 3] = g_down5[3, 0]


'''                     Energy Momentum tensor                  '''
    
#########################  ideal fluid, in covming coordinate  #########################
rho, p = sym.symbols('rho, p')
Tij = sym.diag(rho, p, p, p)









gr1 = GR_tensor(x, g_down1)
gr1.print_Connection()
gr1.print_Ricci_tensor()
gr1.print_Ricci_scalar()


print('\nFRW metric, Einstein equations : \n')
gr1.print_Einstein_equations(Tij)



