from tIGAr import *

def psi_el(E,A0,A1,material):
    '''
        original form of the material model
        returns the total potential energy
    '''
    def unit(v):
        return v/sqrt(inner(v, v))

    def localCartesianBasis(a0, a1):                
        # Perform Gram--Schmidt orthonormalization to get e0 and e1.
        e0 = unit(a0)
        e1 = unit(a1- e0*inner(a1, e0))
        return e0, e1

    def change2basis(v,g1,g2,g3):
        M = as_matrix(((inner(g1,g1),inner(g2,g1),inner(g3,g1)),
            (inner(g1,g2),inner(g2,g2),inner(g3,g2)),
            (inner(g1,g3),inner(g2,g3),inner(g3,g3))))
        Mc = inv(M)
        b = as_vector([inner(v,g1),inner(v,g2),inner(v,g3)])
        return Mc*b

    # read material model and parameters
    # the last key is used
    for key in material:
        el_model = int(key)
        params = material[key]

    if el_model == -1:
        # for testing purpose
        c0 = 1000.
        c1 = 50.
        c2 = 9 
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        return 0.5*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1)
    if el_model == 0:
        # for testing purpose
        c0 = 1000.
        c1 = 1.
        c2 = 9
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        return 0.5*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1)
    if el_model == 1:
        # Neo-Hookean potential, as an example:
        c0 = 1000
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        return 0.5*c0*(I1 - 3.0)
    if el_model == 2:
        # Exponential potential:
        # parameters taken from Kamensky's paper
        c0 = 2.5
        c1 = 0.05
        c2 = 9.0
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        return 0.5*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1)
    if el_model == 3:
        # Exponential potential:
        # parameters taken from Kiendl's paper, 
        # Isogeometric KL shell formulation for general hyperelastic materials
        c0 = 50. # ~ 0.2MPa
        c1 = 12.5 # ~ 0.05MPa
        c2 = 100
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        return 0.5*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1)

    if el_model == 4:
        #TODO: double check!
        # anisotropic model
        # fiber stiffens in the circumferential direction
        '''
        c0 = 50. # ~ 0.2MPa
        c1 = 12.5 # ~ 0.05MPa
        c2 = 9.
        c3 = 500.
        c4 = 10.
        '''
        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']
        c3 = params['c3']
        c4 = params['c4']
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        ''' 
        f = unit(A0)
        e0,e1 = localCartesianBasis(A0, A1)                
        e2 = unit(cross(e0,e1))
        f_newbasis = change2basis(f,e0,e1,e2)     
        I4 = inner(f_newbasis,C*f_newbasis)
        '''
        I4 = C[0,0]
        return 0.5*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1) + 0.5*c3*(exp(c4*pow(I4 - 1.0,2.0))-1)
        #return 0.5*c3*(exp(c4*pow(I4 - 1.0,2.0))-1) + 0.5*c0*(exp(pow(I1 - 3.0,1.0))-1)

    if el_model == 5:
        # stiffness in the coaptation is softened
        domain_material = material['5']['domain_material']
        c0_coapt = params['c0_coapt']
        c1_coapt = params['c1_coapt']
        c2_coapt = params['c2_coapt']
        c3_coapt = params['c3_coapt']
        c4_coapt = params['c4_coapt']
        c0 = c0_coapt + (params['c0']-c0_coapt)*domain_material
        c1 = c1_coapt + (params['c1']-c1_coapt)*domain_material
        c2 = c2_coapt + (params['c2']-c2_coapt)*domain_material
        c3 = c3_coapt + (params['c3']-c3_coapt)*domain_material
        c4 = c4_coapt + (params['c4']-c4_coapt)*domain_material

        C = 2.0*E + Identity(3)
        I1 = tr(C)
        ''' 
        f = unit(A0)
        e0,e1 = localCartesianBasis(A0, A1)                
        e2 = unit(cross(e0,e1))
        f_newbasis = change2basis(f,e0,e1,e2)     
        I4 = inner(f_newbasis,C*f_newbasis)
        '''
        I4 = C[0,0]
        return 0.5*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1) + 0.5*c3*(exp(c4*pow(I4 - 1.0,2.0))-1)

def psi_el_sub_mat4(E,A0,A1,material):
    '''
        material model for model 4
        anisotropic along the circumferential direction
        returns each component of the strain energy
    '''
    def unit(v):
        return v/sqrt(inner(v, v))

    def localCartesianBasis(a0, a1):                
        # Perform Gram--Schmidt orthonormalization to get e0 and e1.
        e0 = unit(a0)
        e1 = unit(a1- e0*inner(a1, e0))
        return e0, e1

    def change2basis(v,g1,g2,g3):
        M = as_matrix(((inner(g1,g1),inner(g2,g1),inner(g3,g1)),
            (inner(g1,g2),inner(g2,g2),inner(g3,g2)),
            (inner(g1,g3),inner(g2,g3),inner(g3,g3))))
        Mc = inv(M)
        b = as_vector([inner(v,g1),inner(v,g2),inner(v,g3)])
        return Mc*b

    # read material model and parameters
    # the last key is used
    for key in material:
        el_model = int(key)
        params = material[key]

    if el_model == 4:
        #TODO: double check!
        # anisotropic model
        # fiber stiffens in the circumferential direction
        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']
        c3 = params['c3']
        c4 = params['c4']
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        I4 = C[0,0]
        return 0.5*c0*(I1-3.0), 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1), 0.5*c3*(exp(c4*pow(I4 - 1.0,2.0))-1)
    else:
        print('model is not 4, pass')
        pass

def psi_el_T(E,A0,A1,material,stepper):
    '''
        add a time stepper to psi_el for the c0 term
    '''
    def unit(v):
        return v/sqrt(inner(v, v))

    def localCartesianBasis(a0, a1):                
        # Perform Gram--Schmidt orthonormalization to get e0 and e1.
        e0 = unit(a0)
        e1 = unit(a1- e0*inner(a1, e0))
        return e0, e1

    def change2basis(v,g1,g2,g3):
        M = as_matrix(((inner(g1,g1),inner(g2,g1),inner(g3,g1)),
            (inner(g1,g2),inner(g2,g2),inner(g3,g2)),
            (inner(g1,g3),inner(g2,g3),inner(g3,g3))))
        Mc = inv(M)
        b = as_vector([inner(v,g1),inner(v,g2),inner(v,g3)])
        return Mc*b

    # read material model and parameters
    # the last key is used
    for key in material:
        el_model = int(key)
        params = material[key]

    if el_model == -1:
        # for testing purpose
        c0 = 1000.
        c1 = 50.
        c2 = 9 
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        return 0.5*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1)
    if el_model == 0:
        # for testing purpose
        c0 = 1000.
        c1 = 1.
        c2 = 9
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        return 0.5*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1)
    if el_model == 1:
        # Neo-Hookean potential, as an example:
        c0 = 1000
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        return 0.5*c0*(I1 - 3.0)
    if el_model == 2:
        # Exponential potential:
        # parameters taken from Kamensky's paper
        c0 = 2.5
        c1 = 0.05
        c2 = 9.0
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        return 0.5*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1)
    if el_model == 3:
        # Exponential potential:
        # parameters taken from Kiendl's paper, 
        # Isogeometric KL shell formulation for general hyperelastic materials
        c0 = 50. # ~ 0.2MPa
        c1 = 12.5 # ~ 0.05MPa
        c2 = 100
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        return 0.5*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1)

    if el_model == 4:
        #TODO: double check!
        # anisotropic model
        # fiber stiffens in the circumferential direction
        '''
        c0 = 50. # ~ 0.2MPa
        c1 = 12.5 # ~ 0.05MPa
        c2 = 9.
        c3 = 500.
        c4 = 10.
        '''
        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']
        c3 = params['c3']
        c4 = params['c4']
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        ''' 
        f = unit(A0)
        e0,e1 = localCartesianBasis(A0, A1)                
        e2 = unit(cross(e0,e1))
        f_newbasis = change2basis(f,e0,e1,e2)     
        I4 = inner(f_newbasis,C*f_newbasis)
        '''
        I4 = C[0,0]
        # reduce the neoHookean term gradually
        return 0.5*stepper*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1) + 0.5*c3*(exp(c4*pow(I4 - 1.0,2.0))-1)
        #return 0.5*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1) + 0.5*c3*(exp(c4*pow(I4 - 1.0,2.0))-1)
        #return 0.5*c3*(exp(c4*pow(I4 - 1.0,2.0))-1) + 0.5*c0*(exp(pow(I1 - 3.0,1.0))-1)
    if el_model == 5:
        # stiffness in the coaptation is softened
        domain_material = material['5']['domain_material']
        c0_coapt = params['c0_coapt']
        c1_coapt = params['c1_coapt']
        c2_coapt = params['c2_coapt']
        c3_coapt = params['c3_coapt']
        c4_coapt = params['c4_coapt']
        c0 = c0_coapt + (params['c0']-c0_coapt)*domain_material
        c1 = c1_coapt + (params['c1']-c1_coapt)*domain_material
        c2 = c2_coapt + (params['c2']-c2_coapt)*domain_material
        c3 = c3_coapt + (params['c3']-c3_coapt)*domain_material
        c4 = c4_coapt + (params['c4']-c4_coapt)*domain_material

        C = 2.0*E + Identity(3)
        I1 = tr(C)
        ''' 
        f = unit(A0)
        e0,e1 = localCartesianBasis(A0, A1)                
        e2 = unit(cross(e0,e1))
        f_newbasis = change2basis(f,e0,e1,e2)     
        I4 = inner(f_newbasis,C*f_newbasis)
        '''
        I4 = C[0,0]
        return 0.5*stepper*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1) + 0.5*c3*(exp(c4*pow(I4 - 1.0,2.0))-1)

def psi_el_sub_mat4_T(E,A0,A1,material,stepper):
    '''
        mat4, time stepper, returns each component of strain energy
    '''
    def unit(v):
        return v/sqrt(inner(v, v))

    def localCartesianBasis(a0, a1):                
        # Perform Gram--Schmidt orthonormalization to get e0 and e1.
        e0 = unit(a0)
        e1 = unit(a1- e0*inner(a1, e0))
        return e0, e1

    def change2basis(v,g1,g2,g3):
        M = as_matrix(((inner(g1,g1),inner(g2,g1),inner(g3,g1)),
            (inner(g1,g2),inner(g2,g2),inner(g3,g2)),
            (inner(g1,g3),inner(g2,g3),inner(g3,g3))))
        Mc = inv(M)
        b = as_vector([inner(v,g1),inner(v,g2),inner(v,g3)])
        return Mc*b

    # read material model and parameters
    # the last key is used
    for key in material:
        el_model = int(key)
        params = material[key]

    if el_model == 4:
        #TODO: double check!
        # anisotropic model
        # fiber stiffens in the circumferential direction
        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']
        c3 = params['c3']
        c4 = params['c4']
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        I4 = C[0,0]
        return 0.5*stepper*c0*(I1-3.0), 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1), 0.5*c3*(exp(c4*pow(I4 - 1.0,2.0))-1)
    else:
        print('model is not 4, pass')
        pass

def psi_el_sub_mat5(E,A0,A1,material):
    '''
        mat5, anisotropic, softer coaptation region
    '''
    def unit(v):
        return v/sqrt(inner(v, v))

    def localCartesianBasis(a0, a1):                
        # Perform Gram--Schmidt orthonormalization to get e0 and e1.
        e0 = unit(a0)
        e1 = unit(a1- e0*inner(a1, e0))
        return e0, e1

    def change2basis(v,g1,g2,g3):
        M = as_matrix(((inner(g1,g1),inner(g2,g1),inner(g3,g1)),
            (inner(g1,g2),inner(g2,g2),inner(g3,g2)),
            (inner(g1,g3),inner(g2,g3),inner(g3,g3))))
        Mc = inv(M)
        b = as_vector([inner(v,g1),inner(v,g2),inner(v,g3)])
        return Mc*b

    # read material model and parameters
    # the last key is used
    for key in material:
        el_model = int(key)
        params = material[key]

    if el_model == 5:
        # stiffness in the coaptation is softened
        domain_material = material['5']['domain_material']
        c0_coapt = params['c0_coapt']
        c1_coapt = params['c1_coapt']
        c2_coapt = params['c2_coapt']
        c3_coapt = params['c3_coapt']
        c4_coapt = params['c4_coapt']
        c0 = c0_coapt + (params['c0']-c0_coapt)*domain_material
        c1 = c1_coapt + (params['c1']-c1_coapt)*domain_material
        c2 = c2_coapt + (params['c2']-c2_coapt)*domain_material
        c3 = c3_coapt + (params['c3']-c3_coapt)*domain_material
        c4 = c4_coapt + (params['c4']-c4_coapt)*domain_material

        C = 2.0*E + Identity(3)
        I1 = tr(C)
        ''' 
        f = unit(A0)
        e0,e1 = localCartesianBasis(A0, A1)                
        e2 = unit(cross(e0,e1))
        f_newbasis = change2basis(f,e0,e1,e2)     
        I4 = inner(f_newbasis,C*f_newbasis)
        '''
        I4 = C[0,0]
        return 0.5*c0*(I1-3.0), 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1), 0.5*c3*(exp(c4*pow(I4 - 1.0,2.0))-1)
    else:
        print('model is not 5, pass')
        pass

def psi_el_sub_mat5_T(E,A0,A1,material,stepper):
    '''
        mat5, anisotropic, softer coaptation region
        add a time stepper for c0
    '''
    def unit(v):
        return v/sqrt(inner(v, v))

    def localCartesianBasis(a0, a1):                
        # Perform Gram--Schmidt orthonormalization to get e0 and e1.
        e0 = unit(a0)
        e1 = unit(a1- e0*inner(a1, e0))
        return e0, e1

    def change2basis(v,g1,g2,g3):
        M = as_matrix(((inner(g1,g1),inner(g2,g1),inner(g3,g1)),
            (inner(g1,g2),inner(g2,g2),inner(g3,g2)),
            (inner(g1,g3),inner(g2,g3),inner(g3,g3))))
        Mc = inv(M)
        b = as_vector([inner(v,g1),inner(v,g2),inner(v,g3)])
        return Mc*b

    # read material model and parameters
    # the last key is used
    for key in material:
        el_model = int(key)
        params = material[key]

    if el_model == 5:
        # stiffness in the coaptation is softened
        domain_material = material['5']['domain_material']
        c0_coapt = params['c0_coapt']
        c1_coapt = params['c1_coapt']
        c2_coapt = params['c2_coapt']
        c3_coapt = params['c3_coapt']
        c4_coapt = params['c4_coapt']
        c0 = c0_coapt + (params['c0']-c0_coapt)*domain_material
        c1 = c1_coapt + (params['c1']-c1_coapt)*domain_material
        c2 = c2_coapt + (params['c2']-c2_coapt)*domain_material
        c3 = c3_coapt + (params['c3']-c3_coapt)*domain_material
        c4 = c4_coapt + (params['c4']-c4_coapt)*domain_material

        C = 2.0*E + Identity(3)
        I1 = tr(C)
        ''' 
        f = unit(A0)
        e0,e1 = localCartesianBasis(A0, A1)                
        e2 = unit(cross(e0,e1))
        f_newbasis = change2basis(f,e0,e1,e2)     
        I4 = inner(f_newbasis,C*f_newbasis)
        '''
        I4 = C[0,0]
        return 0.5*stepper*c0*(I1-3.0), 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1), 0.5*c3*(exp(c4*pow(I4 - 1.0,2.0))-1)
    else:
        print('model is not 5, pass')
        pass

