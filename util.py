from tIGAr import *
from ufl import atan_2
from ufl.operators import Min
import numpy as np
import inspect
import os
import meshio
from module_geo_design_v11 import make_single_leaflet, Valve
import csv
from scipy.optimize import minimize,basinhopping
from io_igakit import *

sys.stdout = open(os.devnull,'w')
#sys.stdout = sys.__stdout__ 

# Normalize a vector v.
def unit(v):
    return v/sqrt(inner(v, v))

# Geometrical quantities for the shell midsurface in a configuration x.
def midsurfaceGeometry(x, spline):
    # Covariant basis vectors
    dxdxi = spline.parametricGrad(x)

    a0 = as_vector([dxdxi[0, 0], dxdxi[1, 0], dxdxi[2, 0]])
    a1 = as_vector([dxdxi[0, 1], dxdxi[1, 1], dxdxi[2, 1]])
    a2 = unit(cross(a0, a1))
               
    a0_proj = project(a0,spline.V)
    a1_proj = project(a1,spline.V)
    a2_proj = project(a2,spline.V)
    # Midsurface metric tensor
    a = as_matrix(((inner(a0, a0), inner(a0, a1)),
                   (inner(a1, a0), inner(a1, a1))))
    # Curvature
    deriv_a2 = spline.parametricGrad(a2)
    b = -as_matrix(((inner(a0, deriv_a2[:, 0]), inner(a0, deriv_a2[:, 1])),
                    (inner(a1, deriv_a2[:, 0]), inner(a1, deriv_a2[:, 1]))))
    return (a0, a1, a2, deriv_a2, a, b)

def root_distention_v1(spline,stepper_BC):
    format_header('root distention model v1')
        # define BC
    def cart2cyl(X):
        x = X[0]
        y = X[1]
        z = X[2]
        X_rho = sqrt(x**2 + y**2)
        X_phi = atan_2(y, x)
        X_height = z
        return(X_rho, X_phi, X_height)

    x0, x1 = SpatialCoordinate(spline.mesh)
    X = spline.F
    X_rho,X_phi,X_height = cart2cyl(X)
    phi_L = conditional(le(x0,1.5),-47/180.*np.pi,conditional(ge(x0,3.5),-np.pi,47/180.*np.pi))
    phi_R = conditional(le(x0,1.5),47/180.*np.pi,conditional(ge(x0,3.5),-47/180.*np.pi,np.pi))
    height_L = conditional(le(x0,1.5),0.8151,conditional(ge(x0,3.5),1.0184,0.8151))
    height_R = conditional(le(x0,1.5),0.8151,conditional(ge(x0,3.5),0.8151,1.0184))
    leaflet_height = (X_phi-phi_L)/(phi_R-phi_L)*height_R + (phi_R-X_phi)/(phi_R-phi_L)*height_L
    commissure_radial_distention = 200/599.
    '''
    mid_basal_attachment_radial_distention = conditional(le(x0,1.5),-150/599.*(1.),conditional(ge(x0,3.5),300/599.,300/599.))
    '''
    # we decide to set the radial component of the anterior leaflet to zero
    mid_basal_attachment_radial_distention = conditional(le(x0,1.5),0.,conditional(ge(x0,3.5),300/599.,300/599.))
    vertical_distention = conditional(le(x0,1.5),-200/599.,conditional(ge(x0,3.5),-200/599.,-200/599.))
    radial_distention = (mid_basal_attachment_radial_distention-commissure_radial_distention)/(phi_R-phi_L)**2*4*(X_phi-phi_L)*(phi_R-X_phi)+commissure_radial_distention

    y0 = as_vector((radial_distention*cos(X_phi)*stepper_BC,\
            radial_distention*sin(X_phi)*stepper_BC,\
            vertical_distention*(-X_height+leaflet_height)/leaflet_height*stepper_BC))
 
    return y0

def root_distention_v2(spline,stepper_BC):
    format_header('root distention model')
    # define BC
    def cart2cyl(X):
        x = X[0]
        y = X[1]
        z = X[2]
        X_rho = sqrt(x**2 + y**2)
        X_phi = atan_2(y, x)
        X_height = z
        return(X_rho, X_phi, X_height)

    x0, x1 = SpatialCoordinate(spline.mesh)
    X = spline.F
    X_rho,X_phi,X_height = cart2cyl(X)
    phi_L = conditional(le(x0,1.5),-47/180.*np.pi,conditional(ge(x0,3.5),-np.pi,47/180.*np.pi))
    phi_R = conditional(le(x0,1.5),47/180.*np.pi,conditional(ge(x0,3.5),-47/180.*np.pi,np.pi))
    height_L = conditional(le(x0,1.5),0.8151,conditional(ge(x0,3.5),1.0184,0.8151))
    height_R = conditional(le(x0,1.5),0.8151,conditional(ge(x0,3.5),0.8151,1.0184))
    leaflet_height = (X_phi-phi_L)/(phi_R-phi_L)*height_R + (phi_R-X_phi)/(phi_R-phi_L)*height_L
    commissure_radial_distention = 200/599.
    '''
    mid_basal_attachment_radial_distention = conditional(le(x0,1.5),-150/599.*(1.),conditional(ge(x0,3.5),300/599.,300/599.))
    '''
    # we decide to set the radial component of the anterior leaflet to zero
    mid_basal_attachment_radial_distention = conditional(le(x0,1.5),0.,conditional(ge(x0,3.5),200/599.,200/599.))
    vertical_distention = conditional(le(x0,1.5),-140/599.,conditional(ge(x0,3.5),-140/599.,-140/599.))
    radial_distention = (mid_basal_attachment_radial_distention-commissure_radial_distention)/(phi_R-phi_L)**2*4*(X_phi-phi_L)*(phi_R-X_phi)+commissure_radial_distention

    y0 = as_vector((radial_distention*cos(X_phi)*stepper_BC,\
            radial_distention*sin(X_phi)*stepper_BC,\
            vertical_distention*(-X_height+leaflet_height)/leaflet_height*stepper_BC))
 
    return y0

def transition_function(x,T1,T2):
    def f(x):
        if x <= 0:
            return 0
        else:
            return np.exp(-1./x)

    if x < T1:
        return 0
    elif x > T2:
        return 1
    else: 
        return f(x-T1)/(f(x-T1)+f(T2-x))

def root_distention(spline,stepper_BC_r,stepper_BC_z):
    format_header('root distention model')
    # define BC
    def cart2cyl(X):
        x = X[0]
        y = X[1]
        z = X[2]
        X_rho = sqrt(x**2 + y**2)
        X_phi = atan_2(y, x)
        X_height = z
        return(X_rho, X_phi, X_height)

    x0, x1 = SpatialCoordinate(spline.mesh)
    X = spline.F

    X_rho,X_phi,X_height = cart2cyl(X)
    phi_L = conditional(le(x0,1.5),-54/180.*np.pi,conditional(ge(x0,3.5),-np.pi,54/180.*np.pi))
    phi_R = conditional(le(x0,1.5),54/180.*np.pi,conditional(ge(x0,3.5),-54/180.*np.pi,np.pi))
    x0_L = conditional(le(x0,1.5),1.,conditional(ge(x0,3.5),5.,3.))
    x0_R = conditional(le(x0,1.5),0.,conditional(ge(x0,3.5),4.,2.))
 
    X_phi_adjusted = phi_R*(x0-x0_L)/(x0_R-x0_L) + phi_L*(x0_R-x0)/(x0_R-x0_L)
    height_L = conditional(le(x0,1.5),0.8151,conditional(ge(x0,3.5),1.0184,0.8151))
    height_R = conditional(le(x0,1.5),0.8151,conditional(ge(x0,3.5),0.8151,1.0184))
    height_Lc = conditional(le(x0,1.5),0.5142,conditional(ge(x0,3.5),0.6725,0.5142))
    height_Rc = conditional(le(x0,1.5),0.5142,conditional(ge(x0,3.5),0.5142,0.6725))
    leaflet_height = (X_phi-phi_L)/(phi_R-phi_L)*height_R + (phi_R-X_phi)/(phi_R-phi_L)*height_L
    leaflet_height_c = (X_phi-phi_L)/(phi_R-phi_L)*height_Rc + (phi_R-X_phi)/(phi_R-phi_L)*height_Lc
    leaflet_height_adjusted = (X_phi_adjusted-phi_L)/(phi_R-phi_L)*height_R + (phi_R-X_phi_adjusted)/(phi_R-phi_L)*height_L
    leaflet_height_c_adjusted = (X_phi_adjusted-phi_L)/(phi_R-phi_L)*height_Rc + (phi_R-X_phi_adjusted)/(phi_R-phi_L)*height_Lc
 
    commissure_radial_distention = 200/599.
    
    mid_basal_attachment_radial_distention = conditional(le(x0,1.5),-140/599.,conditional(ge(x0,3.5),200/599.,200/599.))
    vertical_distention = conditional(le(x0,1.5),-140/599.,conditional(ge(x0,3.5),-140/599.,-140/599.))
    radial_distention = (mid_basal_attachment_radial_distention-commissure_radial_distention)/(phi_R-phi_L)**2*4*(X_phi-phi_L)*(phi_R-X_phi)+commissure_radial_distention
    radial_distention_adjusted = (mid_basal_attachment_radial_distention-commissure_radial_distention)/(phi_R-phi_L)**2*4*(X_phi_adjusted-phi_L)*(phi_R-X_phi_adjusted)+commissure_radial_distention

    # original version
    # not working as good the adjusted version
    #y0 = as_vector((radial_distention*cos(X_phi)*stepper_BC_r,\
    #        radial_distention*sin(X_phi)*stepper_BC_r,\
    #        Min(0,vertical_distention*(-X_height+leaflet_height_c)/leaflet_height_c*stepper_BC_z)))

    # adjusted version
    y0 = as_vector((radial_distention_adjusted*cos(X_phi_adjusted)*stepper_BC_r,\
            radial_distention_adjusted*sin(X_phi_adjusted)*stepper_BC_r,\
            Min(0,vertical_distention*(-X_height+leaflet_height_c_adjusted)/leaflet_height_c_adjusted*stepper_BC_z)))
    #y0 = as_vector((0.2*cos(X_phi)*tepper_BC_r,\
    #        0.2*sin(X_phi)*stepper_BC_r,\
    #        0.*stepper_BC_z))

    return y0,radial_distention_adjusted,X_phi_adjusted


def thickness_model_1(spline,thickness):
    def cart2cyl(X):
        x = X[0]
        y = X[1]
        z = X[2]
        X_rho = sqrt(x**2 + y**2)
        X_phi = atan_2(y,x)
        X_height = z
        return(X_rho, X_phi, X_height)

    x0, x1 = SpatialCoordinate(spline.mesh)
    X = spline.F
    X_rho,X_phi,X_height = cart2cyl(X)
    phi_L = conditional(le(x0,1.5),-47/180.*np.pi,conditional(ge(x0,3.5),-np.pi,47/180.*np.pi))
    phi_R = conditional(le(x0,1.5),47/180.*np.pi,conditional(ge(x0,3.5),-47/180.*np.pi,np.pi))
    buldge_coeff = conditional(le(x0,1.5),4.3,conditional(ge(x0,3.5),4.3,4.3))
    r_center = conditional(le(x0,1.5),0.9,conditional(ge(x0,3.5),0.9,0.9))
    r_span = conditional(le(x0,1.5),0.15,conditional(ge(x0,3.5),0.11,0.11)) # in reference position
    c_center = (phi_L+phi_R)/2.
    c_span = conditional(le(x0,1.5),0.2,conditional(ge(x0,3.5),0.2,0.2)) # in radian
   
    '''
    x_modifier = conditional(le(X_phi,c_center-c_span/2.),0,conditional(ge(X_phi,c_center+c_span/2.),0,(X_phi-c_center+c_span/2.)*(c_span/2.+c_center-X_phi)/c_span**2.*4.))
    y_modifier = conditional(le(x1,r_center-r_span/2.),0,conditional(ge(x1,r_center+r_span/2.),0,(x1-r_center+r_span/2.)*(r_center+r_span/2.-x1)/r_span**2.*4.))
    modifier = 1.+(buldge_coeff-1.)*x_modifier*y_modifier
    '''
    modifier = 1.+(buldge_coeff-1.)*exp(-(X_phi-c_center)**2/2/c_span**2)*exp(-(x1-r_center)**2/2/r_span**2)
    '''
    tmp = project(modifier,spline.V_control)
    tmpFile = File("results/test_thickness/tmp.pvd")
    tmpFile << (tmp,0)
    exit()
    '''
    h_th = modifier*thickness

    return h_th

def thickness_model(spline,thickness):
    def cart2cyl(X):
        x = X[0]
        y = X[1]
        z = X[2]
        X_rho = sqrt(x**2 + y**2)
        X_phi = atan_2(y,x)
        X_height = z
        return(X_rho, X_phi, X_height)

    x0, x1 = SpatialCoordinate(spline.mesh)
    X = spline.F
    X_rho,X_phi,X_height = cart2cyl(X)
    phi_L = conditional(le(x0,1.5),-54/180.*np.pi,conditional(ge(x0,3.5),-np.pi,54/180.*np.pi))
    phi_R = conditional(le(x0,1.5),54/180.*np.pi,conditional(ge(x0,3.5),-54/180.*np.pi,np.pi))
    buldge_coeff = conditional(le(x0,1.5),3.5,conditional(ge(x0,3.5),3.5,3.5))
    r_center = conditional(le(x0,1.5),0.9,conditional(ge(x0,3.5),0.9,0.9))
    r_span = conditional(le(x0,1.5),0.15,conditional(ge(x0,3.5),0.11,0.11)) # in reference position
    c_center = (phi_L+phi_R)/2.
    c_span = conditional(le(x0,1.5),0.2,conditional(ge(x0,3.5),0.35,0.35)) # in radian
   
    '''
    x_modifier = conditional(le(X_phi,c_center-c_span/2.),0,conditional(ge(X_phi,c_center+c_span/2.),0,(X_phi-c_center+c_span/2.)*(c_span/2.+c_center-X_phi)/c_span**2.*4.))
    y_modifier = conditional(le(x1,r_center-r_span/2.),0,conditional(ge(x1,r_center+r_span/2.),0,(x1-r_center+r_span/2.)*(r_center+r_span/2.-x1)/r_span**2.*4.))
    modifier = 1.+(buldge_coeff-1.)*x_modifier*y_modifier
    '''
    modifier = 1.+(buldge_coeff-1.)*exp(-(X_phi-c_center)**2/2/c_span**2)*exp(-(x1-r_center)**2/2/r_span**2)
    '''
    tmp = project(modifier,spline.V_control)
    tmpFile = File("results/test_thickness/tmp.pvd")
    tmpFile << (tmp,0)
    exit()
    '''
    h_th = modifier*thickness

    return h_th

def coaptation_model(spline):
    mesh = spline.mesh
    x0, x1 = SpatialCoordinate(mesh)
    coaptation_type = 'DG'
    if coaptation_type is 'DG':
        '''
        Discontinuous change
        '''
        # 1: belly  
        # 0: coaptation
        #domain_material_A = conditional(x1 >= 0.86-1.36*(x0-0.5)**2,0,1) 
        #domain_material_L = conditional(x1 >= 0.86-1.36*(x0-2.5)**2,0,1) 
        #domain_material_R = conditional(x1 >= 0.86-1.36*(x0-4.5)**2,0,1) 
        domain_material_A = conditional(x1 >= 0.86-3.44*(x0-0.5)**2,0,1) 
        domain_material_L = conditional(x1 >= 0.86-3.44*(x0-2.5)**2,0,1) 
        domain_material_R = conditional(x1 >= 0.86-3.44*(x0-4.5)**2,0,1) 
    if coaptation_type is 'CG':
        '''
        Continuous change
        (TBD)
        '''
        pass
 
    domain_material = conditional(le(x0,1.5),domain_material_A, conditional(ge(x0,3.5),domain_material_R,domain_material_L))
    return domain_material

class MyLoadStepper:
    def __init__(self,DELTA_T,t=0.):
        self.DELTA_T = DELTA_T
        self.tval = t
        self.t = Expression("t",t=self.tval,degree=0)
    def advance(self):
        self.tval += float(self.DELTA_T)
        self.t.t = self.tval
        print('Advance to %f'%(self.tval))

class MyLoadStepper_simple:
    def __init__(self,DELTA_T,checkout_list=[],t=0.0):
        self.DELTA_T0 = DELTA_T
        self.DELTA_T = DELTA_T
        checkout_list.append(np.Inf)
        #checkout_list = [x+self.DELTA_T for x in checkout_list]
        self.checkout_list = checkout_list
        self.tval_prev = t # previous time value
        self.tval = t # current time value
        self.t = Expression("t",t=self.tval,degree=0)
        self.next_checkout_point = min([x for x in self.checkout_list if x > self.tval])
        self.view_log()
    def view_log(self):
        print('tval = {:2.3}'.format(self.tval))
        print('tval_prev = {:2.3}'.format(self.tval_prev))
        print('dt = {}'.format(self.DELTA_T))
        print('next checkout point = {:2.3}'.format(self.next_checkout_point))
        print('{}'.format(self.is_on_checkout_point()))
    def is_on_checkout_point(self):
        return min([abs(x-self.tval) for x in self.checkout_list]) < 1e-5
    def advance(self):
        print('LoadStepper: advance')
        self.tval_prev = self.tval
        self.tval += float(self.DELTA_T)
        if self.tval >= self.next_checkout_point-1e-5:
            print('LoadStepper: Set to the next check out point {:2.3}'.format(self.next_checkout_point))
            self.tval = self.next_checkout_point
            self.next_checkout_point = min([x for x in self.checkout_list if x > self.next_checkout_point])
        self.t.t = self.tval
        self.view_log()
    def stepback(self):
        if self.is_on_checkout_point:
            self.next_checkout_point = self.tval
            self.DELTA_T = self.tval-self.tval_prev
        self.tval = self.tval_prev
        self.t.t = self.tval_prev
        print('LoadStepper: Step back to t = {:2.3}'.format(self.tval))
    def reset(self,DELTA_T):
        if DELTA_T < self.DELTA_T0/50:
            print('LoadStepper: Exit due to nonconvergence.')
            exit()
        if DELTA_T > 0.05:  
            DELTA_T = 0.05
            print('LoadStepper: Set dt to max {:2.3}'.format(DELTA_T))
        if DELTA_T > self.DELTA_T:
            print('LoadStepper: Increase dt to {:2.3}'.format(DELTA_T))
        elif DELTA_T < self.DELTA_T:
            print('LoadStepper: Decrease dt to {:2.3}'.format(DELTA_T))
        self.DELTA_T = DELTA_T

class MyStabilizationStepper(UserExpression):
    def __init__(self,t,**kwargs):
        super().__init__(**kwargs)
        self.t = t

    def eval(self, values, x):
        if self.t < 1./3:
            values[0] = (1.-3.*self.t)**2
        else:
            values[0] = 0.

def generate_valve(save_dir='./',N1=10,N2=10):
    params = {}
    ax = [] 
    '''
    RSGM version 15
    see v4/results/fitting/RSGM/representativeGeometry/
    '''
    x0_geo = [-0.8552113334772213,-0.798,0.005171421154282905,-0.8552113334772213,-0.798,0.07746598434158469,598.9689332728603,530.0,54.0,180.0,9.2,0.8617361429496344,0.65,-0.4805687569425492,0.2848680825131336,-0.023674101402830217,1.2443200528322418,1.058384803709024] # v15, final
    
    opt_params = dict()
    opt_params['leaflet1'] = dict()
    opt_params['leaflet2'] = dict()
    opt_params['leaflet3'] = dict()
    opt_params['global'] = dict()

    opt_params['leaflet1']['alpha'] = x0_geo[0]/np.pi*180
    opt_params['leaflet1']['phi'] = 0
    opt_params['leaflet1']['r_pb'] = x0_geo[1]
    opt_params['leaflet1']['al_pf'] = x0_geo[2]
    opt_params['leaflet2']['alpha'] = x0_geo[3]/np.pi*180
    opt_params['leaflet2']['phi'] = 0
    opt_params['leaflet2']['r_pb'] = x0_geo[4]
    opt_params['leaflet2']['al_pf'] = x0_geo[5]
    opt_params['leaflet3']['alpha'] = x0_geo[3]/np.pi*180
    opt_params['leaflet3']['phi'] = 0
    opt_params['leaflet3']['r_pb'] = x0_geo[4]
    opt_params['leaflet3']['al_pf'] = x0_geo[5]

    Rb = 1./x0_geo[17]
    opt_params['global']['Rb'] = Rb
    opt_params['global']['scale'] = x0_geo[6]
    opt_params['global']['H'] = x0_geo[7]/x0_geo[6]
    opt_params['global']['angX'] = x0_geo[8]
    opt_params['global']['angY'] = x0_geo[9]
    opt_params['global']['beta'] = x0_geo[10]
    opt_params['global']['Rc'] = x0_geo[11]
    opt_params['global']['r_Hc'] = x0_geo[12]
    opt_params['global']['r_Ht'] = x0_geo[13] # height of the triple point

    opt_params['leaflet1']['r_xshift'] = x0_geo[14]#1.14
    opt_params['leaflet2']['r_xshift'] = x0_geo[15]#1.12
    opt_params['leaflet3']['r_xshift'] = x0_geo[15]#1.12

    opt_params['leaflet1']['k_trefoil'] = x0_geo[16]#1.14
    opt_params['leaflet2']['k_trefoil'] = x0_geo[17]#1.12
    opt_params['leaflet3']['k_trefoil'] = x0_geo[17]#1.12

    Rb = opt_params['global']['Rb']

    params['Rb'] = Rb
    params['r_Rc'] = opt_params['global']['Rc']/Rb
    params['H'] = opt_params['global']['H']
    params['r_Hc'] = opt_params['global']['r_Hc']
    params['r_Ht'] = opt_params['global']['r_Ht']
    params['beta_deg'] = -opt_params['global']['beta']

    params['basal_shape'] = 0. 
    params['ctrl_shape'] = 0.
    ang_gap_deg = 1.5

    params['SHOW_off'] = True 
 
    al_pf0 = opt_params['leaflet2']['al_pf']
    al_pf = al_pf0
    sys.stdout.flush()
     
    opt_params['leaflet2']['al_pf'] = al_pf
    opt_params['leaflet3']['al_pf'] = al_pf

    opt_params['global']['N1'] = N1
    opt_params['global']['N2'] = N2
 
    params['quiet'] = False 
    params['N1'] = N1
    params['N2'] = N2
    params['ang_gap_deg'] = ang_gap_deg
    params['ang1_deg'] = opt_params['global']['angX']-ang_gap_deg
    params['ang2_deg'] = -opt_params['global']['angX']+ang_gap_deg
    params['phi'] = opt_params['leaflet1']['phi']/180*np.pi 
    params['alpha'] = opt_params['leaflet1']['alpha']/180*np.pi
    params['r_pb'] = opt_params['leaflet1']['r_pb'] 
    params['al_pf'] = opt_params['leaflet1']['al_pf']
    params['k_trefoil'] = opt_params['leaflet1']['k_trefoil']
    params['r_xshift'] = opt_params['leaflet1']['r_xshift']
    srf1, srf1_is_valid, sinus1, full_params1 = make_single_leaflet(ax, params)
    params['ang1_deg'] = opt_params['global']['angY']-ang_gap_deg
    params['ang2_deg'] = opt_params['global']['angX']+ang_gap_deg
    params['phi'] = opt_params['leaflet2']['phi']/180*np.pi
    params['alpha'] = opt_params['leaflet2']['alpha']/180*np.pi
    params['r_pb'] = opt_params['leaflet2']['r_pb'] 
    params['al_pf'] = opt_params['leaflet2'] ['al_pf']
    params['k_trefoil'] = opt_params['leaflet2']['k_trefoil']
    params['r_xshift'] = opt_params['leaflet2']['r_xshift']
    srf2, srf2_is_valid, sinus2, full_params2 = make_single_leaflet(ax, params)
    params['ang1_deg'] = 360-opt_params['global']['angX']-ang_gap_deg
    params['ang2_deg'] = opt_params['global']['angY']+ang_gap_deg
    params['phi'] = opt_params['leaflet3']['phi']/180*np.pi
    params['alpha'] = opt_params['leaflet3']['alpha']/180*np.pi
    params['r_pb'] = opt_params['leaflet3']['r_pb']
    params['al_pf'] = opt_params['leaflet3'] ['al_pf']
    params['k_trefoil'] = opt_params['leaflet3']['k_trefoil']
    params['r_xshift'] = opt_params['leaflet3']['r_xshift']
    srf3, srf3_is_valid, sinus3, full_params3 = make_single_leaflet(ax, params)

    '''
    TODO: to make refinement work, one needs to update the postprocessing module
    srf1.refine(0,[0.45,0.55])
    srf2.refine(0,[0.45,0.55])
    srf3.refine(0,[0.45,0.55])
    '''

    valve = Valve(srf1, srf2, srf3, sinus1, sinus2, sinus3)
    valve.attach_full_params(full_params1, full_params2, full_params3)
    if valve.is_valid(None) and srf1_is_valid and srf2_is_valid and srf3_is_valid:
        valve_is_valid = True
    else:
        valve_is_valid = False
    
    print(srf1_is_valid)
    print(srf2_is_valid)
    print(srf3_is_valid)

    if True:
        shift1 = [0.,0.,0.];
        shift2 = [0.,0.,0.];
        shift3 = [0.,0.,0.];
        valve.write_valve_skeleton(os.path.join(save_dir,'meshn'),-0.019)
        valve.save(os.path.join(save_dir,'meshn','leaflet_tmp_'),-0.019,shift1,shift2,shift3)
        valve.write_valve_skeleton(os.path.join(save_dir,'mesh'))
        valve.save(os.path.join(save_dir,'mesh','leaflet_tmp_'),0,shift1,shift2,shift3)
        valve.write_valve_skeleton(os.path.join(save_dir,'meshp'),0.019)
        valve.save(os.path.join(save_dir,'meshp','leaflet_tmp_'),0.019,shift1,shift2,shift3)
    
    return valve, valve_is_valid, opt_params 


def generate_valve_v10(save_dir='./'):
    params = {}
    ax = [] 
    '''
    RSGM version 10
    The geometry has been confirmed to be the same as the rsGM
    '''
    x0_geo = [
            -0.8552113334772213,0.0,0.20709699936409548,-0.8552113334772213,-0.7988043665168549,0.21313351041265716,598.9689332728603,540.0,48.87158880960192,180.0,9.2,0.7555069451108054,0.6764682309637581,-0.40175853402318085,1.2443200528322418,1.058384803709024
            ]
    
    opt_params = dict()
    opt_params['leaflet1'] = dict()
    opt_params['leaflet2'] = dict()
    opt_params['leaflet3'] = dict()
    opt_params['global'] = dict()

    opt_params['leaflet1']['alpha'] = x0_geo[0]/np.pi*180
    opt_params['leaflet1']['phi'] = 10
    opt_params['leaflet1']['r_pb'] = x0_geo[4]
    opt_params['leaflet1']['al_pf'] = x0_geo[2]
    opt_params['leaflet2']['alpha'] = x0_geo[3]/np.pi*180
    opt_params['leaflet2']['phi'] = 10
    opt_params['leaflet2']['r_pb'] = x0_geo[4]
    opt_params['leaflet2']['al_pf'] = x0_geo[5]
    opt_params['leaflet3']['alpha'] = x0_geo[3]/np.pi*180
    opt_params['leaflet3']['phi'] =  10
    opt_params['leaflet3']['r_pb'] = x0_geo[4]
    opt_params['leaflet3']['al_pf'] = x0_geo[5]

    Rb = 1./x0_geo[15]
    opt_params['global']['Rb'] = Rb
    opt_params['global']['scale'] = x0_geo[6]
    opt_params['global']['H'] = x0_geo[7]/x0_geo[6]
    opt_params['global']['angX'] = x0_geo[8]
    opt_params['global']['angY'] = x0_geo[9]
    opt_params['global']['beta'] = x0_geo[10]
    opt_params['global']['Rc'] = x0_geo[11]
    opt_params['global']['r_Hc'] = x0_geo[12]
    opt_params['global']['r_Ht'] = x0_geo[13] # height of the triple point

    opt_params['leaflet1']['k_trefoil'] = x0_geo[14]#1.14
    opt_params['leaflet2']['k_trefoil'] = x0_geo[15]#1.12
    opt_params['leaflet3']['k_trefoil'] = x0_geo[15]#1.12

    #Rb = 0.97
    Rb = opt_params['global']['Rb']

    params['Rb'] = Rb
    params['r_Rc'] = opt_params['global']['Rc']/Rb
    params['H'] = opt_params['global']['H']
    params['r_Hc'] = opt_params['global']['r_Hc']
    params['r_Ht'] = opt_params['global']['r_Ht']
    params['beta_deg'] = -opt_params['global']['beta']

    params['basal_shape'] = 0. 
    params['ctrl_shape'] = 0.
    ang_gap_deg = 1.5

    params['SHOW_off'] = True 
 
    al_pf0 = opt_params['leaflet2']['al_pf']
    al_pf = al_pf0
    sys.stdout.flush()
     
    opt_params['leaflet2']['al_pf'] = al_pf
    opt_params['leaflet3']['al_pf'] = al_pf

    N1 = 5
    N2 = 5
    opt_params['global']['N1'] = N1
    opt_params['global']['N2'] = N2
 
    params['quiet'] = False 
    params['N1'] = N1
    params['N2'] = N2
    params['ang_gap_deg'] = ang_gap_deg
    params['ang1_deg'] = opt_params['global']['angX']-ang_gap_deg
    params['ang2_deg'] = -opt_params['global']['angX']+ang_gap_deg
    params['phi'] = opt_params['leaflet1']['phi']/180*np.pi 
    params['alpha'] = opt_params['leaflet1']['alpha']/180*np.pi
    params['r_pb'] = opt_params['leaflet1']['r_pb'] 
    params['al_pf'] = opt_params['leaflet1']['al_pf']
    params['k_trefoil'] = opt_params['leaflet1']['k_trefoil']
    srf1, srf1_is_valid, sinus1, full_params1 = make_single_leaflet(ax, params)
    params['ang1_deg'] = opt_params['global']['angY']-ang_gap_deg
    params['ang2_deg'] = opt_params['global']['angX']+ang_gap_deg
    params['phi'] = opt_params['leaflet2']['phi']/180*np.pi
    params['alpha'] = opt_params['leaflet2']['alpha']/180*np.pi
    params['r_pb'] = opt_params['leaflet2']['r_pb'] 
    params['al_pf'] = opt_params['leaflet2'] ['al_pf']
    params['k_trefoil'] = opt_params['leaflet2']['k_trefoil']
    srf2, srf2_is_valid, sinus2, full_params2 = make_single_leaflet(ax, params)
    params['ang1_deg'] = 360-opt_params['global']['angX']-ang_gap_deg
    params['ang2_deg'] = opt_params['global']['angY']+ang_gap_deg
    params['phi'] = opt_params['leaflet3']['phi']/180*np.pi
    params['alpha'] = opt_params['leaflet3']['alpha']/180*np.pi
    params['r_pb'] = opt_params['leaflet3']['r_pb']
    params['al_pf'] = opt_params['leaflet3'] ['al_pf']
    params['k_trefoil'] = opt_params['leaflet3']['k_trefoil']
    srf3, srf3_is_valid, sinus3, full_params3 = make_single_leaflet(ax, params)

    '''
    TODO: to make refinement work, one needs to update the postprocessing module
    srf1.refine(0,[0.45,0.55])
    srf2.refine(0,[0.45,0.55])
    srf3.refine(0,[0.45,0.55])
    '''

    valve = Valve(srf1, srf2, srf3, sinus1, sinus2, sinus3)
    valve.attach_full_params(full_params1, full_params2, full_params3)
    if valve.is_valid(None) and srf1_is_valid and srf2_is_valid and srf3_is_valid:
        valve_is_valid = True
    else:
        valve_is_valid = False
    
    print(srf1_is_valid)
    print(srf2_is_valid)
    print(srf3_is_valid)

    if True:
        shift1 = [0.,0.,0.];
        shift2 = [0.,0.,0.];
        shift3 = [0.,0.,0.];
        valve.write_valve_skeleton(os.path.join(save_dir,'meshn'),-0.019)
        valve.save(os.path.join(save_dir,'meshn','leaflet_tmp_'),-0.019,shift1,shift2,shift3)
        valve.write_valve_skeleton(os.path.join(save_dir,'mesh'))
        valve.save(os.path.join(save_dir,'mesh','leaflet_tmp_'),0,shift1,shift2,shift3)
        valve.write_valve_skeleton(os.path.join(save_dir,'meshp'),0.019)
        valve.save(os.path.join(save_dir,'meshp','leaflet_tmp_'),0.019,shift1,shift2,shift3)
    
    return valve, valve_is_valid, opt_params 

## -------------------------------------------------------------------------------------------

def format_header(string,mode = None):
    if mode == 'begin':
        print('>'*10 + ' Begin ' + string)
    if mode == 'end':
        print('<'*5 + ' End ' + string)
    if mode is None:
        print('>'*10 + ' ' + string)

def query_Function(func):
    print('+'*10+'  '+str(func.vector().norm('linf')))
    print('+'*10+'  '+str(func.vector().norm('l2')))

def inspect_fun(func):
    print(os.path.abspath(inspect.getfile(func)))

def mesh_quad2tri(mesh):
    '''
    convert a quadrilateral mesh to a triangular mesh
    '''
    #print(mesh.coordinates())
    #print(mesh.cells())
    cells_tri = mesh.cells()
    cells_tri = [[[x,y,z],[y,z,w]] for x,y,z,w in mesh.cells()]
    cells_tri = [val for sublist in cells_tri for val in sublist]
    mesh_tri = meshio.Mesh(np.array(mesh.coordinates()),[("triangle",np.array(cells_tri))])

    # output to a temporary file and then load it back
    #meshio.write('./tmp.xml',mesh_tri)
    #mesh_tri_df = Mesh('./tmp.xml')
    
    # use MeshEditor to directly write to a triangular mesh
    mesh_tri_df = Mesh()
    editor = MeshEditor()
    editor.open(mesh_tri_df,"triangle",mesh.geometry().dim(),mesh.topology().dim())
    editor.init_vertices(len(mesh_tri.points))
    editor.init_cells(len(cells_tri))

    [editor.add_vertex(i,n) for i,n in enumerate(mesh_tri.points)]
    [editor.add_cell(i,n) for i,n in enumerate(cells_tri)]
    editor.close()
    return mesh_tri_df

def scalar_func_quad2tri(function):
    '''
    input is a scalar function defined on a quadrilateral mesh
    '''
    function_space = function.function_space()
    mesh = function_space.mesh()
    vertex_values = function.compute_vertex_values(mesh)
    mesh_tri = mesh_quad2tri(mesh)
    function_space_tri = FunctionSpace(mesh_tri,'CG',1)
    function_tri = Function(function_space_tri)
    v2d = vertex_to_dof_map(function_space_tri)
    d2v = dof_to_vertex_map(function_space_tri)

    for i in range(function_space_tri.dim()):
        function_tri.vector()[i] = vertex_values[d2v[i]]

    return function_tri
   
def calc_curve_length(curve):
    '''
    Calculate the length of a 3D curve given a list of points on the curve.
   Example:
        curve = [[np.cos(t),np.sin(t),0] for t in np.linspace(0,2*np.pi,100)]
        l = calc_curve_length(curve)
        print(l)
    '''
    numpy_curve = np.array(curve)
    length = 0
    for i in range(numpy_curve.shape[0]-1):
        length += sqrt(np.power(curve[i][0]-curve[i+1][0], 2) +
                    np.power(curve[i][1]-curve[i+1][1], 2) +
                    np.power(curve[i][2]-curve[i+1][2], 2))
    return length

def calc_curve_length_proj(curve):
    '''
    Calculate the length of a 3D curve projected on the XY plane 
    given a list of points on the curve.
    Example:
        curve = [[np.cos(t),np.sin(t),t] for t in np.linspace(0,2*np.pi,100)]
        l = calc_curve_length(curve)
        print(l)
    '''
    numpy_curve = np.array(curve)
    length = 0
    for i in range(numpy_curve.shape[0]-1):
        length += sqrt(np.power(curve[i][0]-curve[i+1][0], 2) +
                    np.power(curve[i][1]-curve[i+1][1], 2))
    return length

def equal_spacing_curve(x,y,z,N,string):
    '''
    Redistribute N+1 equally-spacing points on the specified 3D curve
    '''
    totLen = 0
    partLen = 0
    x_redist = [x[0]]
    y_redist = [y[0]]
    z_redist = [z[0]]
    N_redist = 1
    for n in range(len(x)-1):
        seg = [x[n]-x[n+1],y[n]-y[n+1],z[n]-z[n+1]]
        totLen = totLen + np.linalg.norm(seg)
    for n in range(len(x)-1):
        seg = [x[n]-x[n+1], y[n]-y[n+1], z[n]-z[n+1]]
        partLen = partLen + np.linalg.norm(seg)
        if partLen > totLen/N:
            x_redist.append(x[n+1])
            y_redist.append(y[n+1])
            z_redist.append(z[n+1])
            N_redist = N_redist+1
            partLen = partLen-totLen/N
    if N_redist < N+1:
        x_redist.append(x[-1])
        y_redist.append(y[-1])
        z_redist.append(z[-1])

    return x_redist, y_redist, z_redist

def transition_function_v1(x):
    def f(x):
        if x <= 0:
            return 0
        else:
            return np.exp(-1./x)

    return f(x)/(f(x)+f(1.-x))

def print_info(var_name,target,fit):
    '''
    print info for a gQOI given the target and fitted values
    '''
    print('[%s] target =  %4.4f,\t fit = %4.4f,\t error = %4.4f,\t r-error = %4.4f%%'%(var_name,target,fit,abs(fit-target),abs(fit-target)/target*100))

def get_R2_curve_processed(leaflet_id, curve):
    # fit by a plane
    # and then normalized
    with open('./mesh/R2_'+leaflet_id+'-redist-3D.csv',mode='w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([p for p in curve])
    curve_processed = process_R_cross_section(curve)
    
    #if(curve_processed[0,0] > curve_processed[-1,0]):
    #    curve_processed = curve_processed[::-1]
    return curve_processed

def get_C2_curve_processed(leaflet_id, curve):
    # fit by a plane
    # and then normalized
    with open('./mesh/C2_'+leaflet_id+'-redist-3D.csv',mode='w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([p for p in curve])
    curve_processed = process_C_cross_section(curve)
    
    if(curve_processed[0,0] > curve_processed[-1,0]):
        curve_processed = curve_processed[::-1]
    return curve_processed

def process_R_cross_section(curve):
    def fun(x):
        val = sum([abs(x[3]+x[0]*p[0]+x[1]*p[1]+x[2]*p[2])/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) for p in curve]) 
        return val

    curve = np.array(curve)

    N = len(curve)
    p1 = curve[0,:]
    p2 = curve[-1,:]
    p3 = curve[int(N/2),:]
    # plane: x0[0]*x+x0[1]*y+x0[2]*z+x0[3]=0
    x0 = list()
    x0.append((p1[1]-p2[1])*(p1[2]-p3[2])-(p1[2]-p2[2])*(p1[1]-p3[1]))
    x0.append(-(p1[0]-p2[0])*(p1[2]-p3[2])+(p1[2]-p2[2])*(p1[0]-p3[0]))
    x0.append((p1[0]-p2[0])*(p1[1]-p3[1])-(p1[1]-p2[1])*(p1[0]-p3[0]))
    x0.append(-(x0[0]*p1[0]+x0[1]*p1[1]+x0[2]*p1[2]))
    #print('x0=')
    #print(x0)
    #print(x0[0]*p1[0]+x0[1]*p1[1]+x0[2]*p1[2]+x0[3])
    #print(x0[0]*p2[0]+x0[1]*p2[1]+x0[2]*p2[2]+x0[3])
    #print(x0[0]*p3[0]+x0[1]*p3[1]+x0[2]*p3[2]+x0[3])
    res = minimize(fun,x0,method='L-BFGS-B',
                options={'disp':None,'ftol':1e-6,'gtol':1e-6,'maxiter':1000})
    #print(res)
    x = res.x
    x = x/np.linalg.norm([x[0],x[1],x[2]])
    normal = [x[0],x[1],x[2]]
    curve_fitted = [[p[0]-x[0]*(x[0]*p[0]+x[1]*p[1]+x[2]*p[2]+x[3])/(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]),p[1]-x[1]*(x[0]*p[0]+x[1]*p[1]+x[2]*p[2]+x[3])/(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]),p[2]-x[2]*(x[0]*p[0]+x[1]*p[1]+x[2]*p[2]+x[3])/(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])] for p in curve]
    curve_fitted = np.array(curve_fitted)
    
    e2 = [0,0,1]
    e2 = np.array(e2)-np.dot(np.array(e2),np.array(normal))*np.array(normal)
    e2 = e2/np.linalg.norm(e2)
    e1 = np.cross(normal,e2)

    p0 = curve_fitted[0,:]
    curve_plane = [[np.dot(p-p0,e1),np.dot(p-p0,e2),0] for p in curve_fitted]
    curve_plane = np.array(curve_plane)
    #print(max(curve_plane[:,0])-min(curve_plane[:,0]))

    #Xmax = curve_plane[-1,0]
    #Hmax = curve_plane[-1,1]
    Xmax = 1
    Hmax = 1
    #print(max(curve_plane[:,0])-min(curve_plane[:,1]))
    #Xmax = max(curve_plane[:,0])-min(curve_plane[:,0])
    #Hmax = Xmax
    curve_processed = [[p[0]/Xmax,p[1]/Hmax,0] for p in curve_plane]
    curve_processed = np.array(curve_processed)

    return curve_processed

def process_C_cross_section(curve):
    print('process_C_cross_section')
    def fun(x):
        val = sum([abs(x[3]+x[0]*p[0]+x[1]*p[1]+x[2]*p[2])/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) for p in curve]) 
        return val

    curve = np.array(curve)

    N = len(curve)
    p1 = curve[0,:]
    p2 = curve[-1,:]
    p3 = curve[int(N/2),:]
    # plane: x0[0]*x+x0[1]*y+x0[2]*z+x0[3]=0
    x0 = list()
    x0.append((p1[1]-p2[1])*(p1[2]-p3[2])-(p1[2]-p2[2])*(p1[1]-p3[1]))
    x0.append(-(p1[0]-p2[0])*(p1[2]-p3[2])+(p1[2]-p2[2])*(p1[0]-p3[0]))
    x0.append((p1[0]-p2[0])*(p1[1]-p3[1])-(p1[1]-p2[1])*(p1[0]-p3[0]))
    x0.append(-(x0[0]*p1[0]+x0[1]*p1[1]+x0[2]*p1[2]))
    #print('x0=')
    #print(x0)
    #print(x0[0]*p1[0]+x0[1]*p1[1]+x0[2]*p1[2]+x0[3])
    #print(x0[0]*p2[0]+x0[1]*p2[1]+x0[2]*p2[2]+x0[3])
    #print(x0[0]*p3[0]+x0[1]*p3[1]+x0[2]*p3[2]+x0[3])
    res = minimize(fun,x0,method='L-BFGS-B',
                options={'disp':None,'ftol':1e-6,'gtol':1e-6,'maxiter':1000})
    #print(res)
    x = res.x
    x = x/np.linalg.norm([x[0],x[1],x[2]])
    normal = [x[0],x[1],x[2]]
    curve_fitted = [[p[0]-x[0]*(x[0]*p[0]+x[1]*p[1]+x[2]*p[2]+x[3])/(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]),p[1]-x[1]*(x[0]*p[0]+x[1]*p[1]+x[2]*p[2]+x[3])/(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]),p[2]-x[2]*(x[0]*p[0]+x[1]*p[1]+x[2]*p[2]+x[3])/(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])] for p in curve]
    curve_fitted = np.array(curve_fitted)
    
    e2 = [0,0,1]
    e2 = np.array(e2)-np.dot(np.array(e2),np.array(normal))*np.array(normal)
    e2 = e2/np.linalg.norm(e2)
    e1 = np.cross(normal,e2)

    tmp = list(curve_fitted[:,2])
    I = tmp.index(min(tmp))
    p0 = curve_fitted[I,:]
    curve_plane = [[np.dot(p-p0,e1),np.dot(p-p0,e2),0] for p in curve_fitted]
    curve_plane = np.array(curve_plane)
    curve_plane[:,1] = curve_plane[:,1]-curve_plane[I,1]+p0[1]
    #print(e1)
    #print(e2)
    #print(max(curve_plane[:,0])-min(curve_plane[:,0]))
    #print(curve_plane[1,:])
    #print(curve_plane[-1,:])
    #exit()

    #Xmax = curve_plane[-1,0]
    #Hmax = curve_plane[-1,1]
    Xmax = 1
    Hmax = 1       
    #Xmax = (max(curve_plane[:,0])-min(curve_plane[:,0]))/2.
    #Hmax = Xmax
    curve_processed = [[p[0]/Xmax,p[1]/Hmax,0] for p in curve_plane]
    curve_processed = np.array(curve_processed)

    return curve_processed

def eval_nurbs(obj,x1,x2):
   U = obj.knots[0] 
   V = obj.knots[1]
   p = obj.degree[0]
   q = obj.degree[1]
   
   return obj((1-x1)*U[p]+x1*U[-p-1], (1-x2)*V[q]+x2*V[-q-1])

class ProbeValve():
    def __init__(self,spline,disp,VERBOSE=False):
        self.probe_leaflet1 = ProbeLeaflet(spline,disp,1,VERBOSE)
        self.probe_leaflet2 = ProbeLeaflet(spline,disp,2,VERBOSE)
        self.probe_leaflet3 = ProbeLeaflet(spline,disp,3,VERBOSE)
        self.VERBOSE = VERBOSE

    def get_probe_leaflet(self,i):
        if i == 1:
            return self.probe_leaflet1
        if i == 2:
            return self.probe_leaflet2
        if i == 3:
            return self.probe_leaflet3

    def distance_commissure_points(self,i):
        # compute the distance between the mid point of the free edge
        # , also one of the ends of R2 curve
        probe_leaflet_i = self.get_probe_leaflet(i)
        F0_func_tri_i = probe_leaflet_i.F0_func_tri
        F1_func_tri_i = probe_leaflet_i.F1_func_tri
        F2_func_tri_i = probe_leaflet_i.F2_func_tri
        xmin_i = probe_leaflet_i.xmin
            
        pt_i1 = Point(xmin_i,1)
        F_pt_i1 = np.array([F0_func_tri_i(pt_i1),F1_func_tri_i(pt_i1),F2_func_tri_i(pt_i1)])
        pt_i2 = Point(xmin_i+1,1)
        F_pt_i2 = np.array([F0_func_tri_i(pt_i2),F1_func_tri_i(pt_i2),F2_func_tri_i(pt_i2)])

        #print("<<")
        #print(F_pt_i1)
        #print(F_pt_i2)
        #print(">>")
        return np.linalg.norm(F_pt_i1-F_pt_i2)
   
    def distance_midpoint_free_edge(self,i,j):
        # compute the distance between the mid point of the free edge
        # , also one of the ends of R2 curve
        probe_leaflet_i = self.get_probe_leaflet(i)
        probe_leaflet_j = self.get_probe_leaflet(j)
        F0_func_tri_i = probe_leaflet_i.F0_func_tri
        F1_func_tri_i = probe_leaflet_i.F1_func_tri
        F2_func_tri_i = probe_leaflet_i.F2_func_tri
        F0_func_tri_j = probe_leaflet_j.F0_func_tri
        F1_func_tri_j = probe_leaflet_j.F1_func_tri
        F2_func_tri_j = probe_leaflet_j.F2_func_tri
        xmin_i = probe_leaflet_i.xmin
        xmin_j = probe_leaflet_j.xmin
            
        pt_i = Point(xmin_i+0.5,1)
        F_pt_i = np.array([F0_func_tri_i(pt_i),F1_func_tri_i(pt_i),F2_func_tri_i(pt_i)])
        pt_j = Point(xmin_j+0.5,1)
        F_pt_j = np.array([F0_func_tri_j(pt_j),F1_func_tri_j(pt_j),F2_func_tri_j(pt_j)])
        #print("<<")
        #print(F_pt_i)
        #print(F_pt_j)
        #print(">>")
        return np.linalg.norm(F_pt_i-F_pt_j)

    def get_valve_height(self):
        # , also one of the ends of R2 curve
        height = 0

        probe_leaflet_i = self.get_probe_leaflet(1)
        F0_func_tri_i = probe_leaflet_i.F0_func_tri
        F1_func_tri_i = probe_leaflet_i.F1_func_tri
        F2_func_tri_i = probe_leaflet_i.F2_func_tri
        xmin_i = probe_leaflet_i.xmin
            
        pt_i = Point(xmin_i,1)
        F_pt_i = np.array([F0_func_tri_i(pt_i),F1_func_tri_i(pt_i),F2_func_tri_i(pt_i)])
        height = height+F2_func_tri_i(pt_i)
        #print("<<")
        #print(F_pt_i)
        #print(">>")
        pt_i = Point(xmin_i+1,1)
        F_pt_i = np.array([F0_func_tri_i(pt_i),F1_func_tri_i(pt_i),F2_func_tri_i(pt_i)])
        height = height+F2_func_tri_i(pt_i)
        #print("<<")
        #print(F_pt_i)
        #print(">>")

        probe_leaflet_i = self.get_probe_leaflet(2)
        F0_func_tri_i = probe_leaflet_i.F0_func_tri
        F1_func_tri_i = probe_leaflet_i.F1_func_tri
        F2_func_tri_i = probe_leaflet_i.F2_func_tri
        xmin_i = probe_leaflet_i.xmin
            
        pt_i = Point(xmin_i,1)
        F_pt_i = np.array([F0_func_tri_i(pt_i),F1_func_tri_i(pt_i),F2_func_tri_i(pt_i)])
        height = height+F2_func_tri_i(pt_i)
        #print("<<")
        #print(F_pt_i)
        #print(">>")
        pt_i = Point(xmin_i+1,1)
        F_pt_i = np.array([F0_func_tri_i(pt_i),F1_func_tri_i(pt_i),F2_func_tri_i(pt_i)])
        height = height+F2_func_tri_i(pt_i)
        #print("<<")
        #print(F_pt_i)
        #print(">>")

        probe_leaflet_i = self.get_probe_leaflet(3)
        F0_func_tri_i = probe_leaflet_i.F0_func_tri
        F1_func_tri_i = probe_leaflet_i.F1_func_tri
        F2_func_tri_i = probe_leaflet_i.F2_func_tri
        xmin_i = probe_leaflet_i.xmin
            
        pt_i = Point(xmin_i,1)
        F_pt_i = np.array([F0_func_tri_i(pt_i),F1_func_tri_i(pt_i),F2_func_tri_i(pt_i)])
        height = height+F2_func_tri_i(pt_i)
        #print("<<")
        #print(F_pt_i)
        #print(">>")
        pt_i = Point(xmin_i+1,1)
        F_pt_i = np.array([F0_func_tri_i(pt_i),F1_func_tri_i(pt_i),F2_func_tri_i(pt_i)])
        height = height+F2_func_tri_i(pt_i)
        #print("<<")
        #print(F_pt_i)
        #print(">>")
        #print(height/6.)
        def D(x):
            return F2_func_tri_i(Point(x,0))
        x0 = np.array([2.5])
        epsilon = 1e-5
        linear_constraint = LinearConstraint([[1.]],[epsilon],[1-epsilon])
        res = minimize(D,x0,method="SLSQP",constraints=linear_constraint)
        #print(res.fun)

        return height/6.-res.fun

    def distance_ij(self,i,j):
        probe_leaflet_i = self.get_probe_leaflet(i)
        probe_leaflet_j = self.get_probe_leaflet(j)
        F0_func_tri_i = probe_leaflet_i.F0_func_tri
        F1_func_tri_i = probe_leaflet_i.F1_func_tri
        F2_func_tri_i = probe_leaflet_i.F2_func_tri
        F0_func_tri_j = probe_leaflet_j.F0_func_tri
        F1_func_tri_j = probe_leaflet_j.F1_func_tri
        F2_func_tri_j = probe_leaflet_j.F2_func_tri
        xmin_i = probe_leaflet_i.xmin
        xmin_j = probe_leaflet_j.xmin

        def D(x):
            x1,y1,x2,y2 = x
            x1 = min([max([x1,0]),1])
            y1 = min([max([y1,0]),1])
            x2 = min([max([x2,0]),1])
            y2 = min([max([y2,0]),1])
            
            pt_i = Point(xmin_i+x1,y1)
            F_pt_i = np.array([F0_func_tri_i(pt_i),F1_func_tri_i(pt_i),F2_func_tri_i(pt_i)])
            pt_j = Point(xmin_j+x2,y2)
            F_pt_i = np.array([F0_func_tri_i(pt_i),F1_func_tri_i(pt_i),F2_func_tri_i(pt_i)])
            F_pt_j = np.array([F0_func_tri_j(pt_j),F1_func_tri_j(pt_j),F2_func_tri_j(pt_j)])
            return np.linalg.norm(F_pt_i-F_pt_j)

        x0 = np.array([0.5,0.5,0.5,0.5])
        epsilon = 1e-5
        linear_constraint = LinearConstraint([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],[epsilon]*4,[1-epsilon]*4)
        res = minimize(D,x0,method="SLSQP",constraints=linear_constraint)

        '''
        print(res.x)
        print(res.fun)
        x1,y1,x2,y2 = res.x
        p1 = Point(xmin_i+x1,y1)
        F_pt_i = np.array([F0_func_tri_i(p1),F1_func_tri_i(p1),F2_func_tri_i(p1)])
        p2 = Point(xmin_j+x2,y2)
        F_pt_j = np.array([F0_func_tri_j(p2),F1_func_tri_j(p2),F2_func_tri_j(p2)])

        print(F_pt_i)
        print(F_pt_j)
        exit()
        '''
        return res.fun

    def run(self): 
        result = dict()
        result['pressure'] = p
        result['valve'] = dict()
        result['leaflet1'] = dict()
        result['leaflet2'] = dict()
        result['leaflet3'] = dict()
        result['monitor'] = dict()
        if self.VERBOSE:
            print('--- leaflet 1 ---')
        obj = self.probe_leaflet1
        result['leaflet1'] = obj.run_leaflet()
        if self.VERBOSE:
            print('--- leaflet 2 ---')
        obj = self.probe_leaflet2
        result['leaflet2'] = obj.run_leaflet()
        if self.VERBOSE:
            print('--- leaflet 3 ---')
        obj = self.probe_leaflet3
        result['leaflet3'] = obj.run_leaflet()

        area1 = result['leaflet1']['area']
        area2 = result['leaflet2']['area']
        area3 = result['leaflet3']['area']

        l_fe1 = result['leaflet1']['length_fe']
        l_fe2 = result['leaflet2']['length_fe']
        l_fe3 = result['leaflet3']['length_fe']
        l_ba1 = result['leaflet1']['length_ba']
        l_ba2 = result['leaflet2']['length_ba']
        l_ba3 = result['leaflet3']['length_ba']
        l_ba1_proj = result['leaflet1']['length_ba_proj']
        l_ba2_proj = result['leaflet2']['length_ba_proj']
        l_ba3_proj = result['leaflet3']['length_ba_proj']
        circumference = l_ba1_proj+l_ba2_proj+l_ba3_proj
       
        dL_12 = self.distance_ij(1,2)
        dL_13 = self.distance_ij(1,3)
        dL_23 = self.distance_ij(2,3)
        dC_1 = self.distance_commissure_points(1)
        dC_2 = self.distance_commissure_points(2)
        dC_3 = self.distance_commissure_points(3)
        dP_12 = self.distance_midpoint_free_edge(1,2)
        dP_13 = self.distance_midpoint_free_edge(1,3)
        dP_23 = self.distance_midpoint_free_edge(2,3)
        result['monitor']['distance_leaflets_12'] = dL_12
        result['monitor']['distance_leaflets_13'] = dL_13
        result['monitor']['distance_leaflets_23'] = dL_23
        result['monitor']['distance_commissure_points_1'] = dC_1
        result['monitor']['distance_commissure_points_2'] = dC_2
        result['monitor']['distance_commissure_points_3'] = dC_3
        result['monitor']['distance_midpoint_fe_12'] = dP_12
        result['monitor']['distance_midpoint_fe_13'] = dP_13
        result['monitor']['distance_midpoint_fe_23'] = dP_23
        height = self.get_valve_height()
        result['valve']['height'] = height

        #if self.VERBOSE:
        if True:
            print('--- Summary ---')
            print("Free edge (1) = %2.3f"%(l_fe1))
            print("Free edge (2) = %2.3f"%(l_fe2))
            print("Free edge (3) = %2.3f"%(l_fe3))
            print("Basal attachment (1) = %2.3f"%(l_ba1))
            print("Basal attachment (2) = %2.3f"%(l_ba2))
            print("Basal attachment (3) = %2.3f"%(l_ba3))
            print("Circumference = %2.3f"%(circumference))

            print('Distance between leaflet1 and leaflet2 = {:2.4f}'.format(dL_12))
            print('Distance between leaflet1 and leaflet3 = {:2.4f}'.format(dL_13))
            print('Distance between leaflet2 and leaflet3 = {:2.4f}'.format(dL_23))
            print('Distance between the free edge mid point of leaflet1 and leaflet2 = {:2.4f}'.format(dP_12))
            print('Distance between the free edge mid point of leaflet1 and leaflet3 = {:2.4f}'.format(dP_13))
            print('Distance between the free edge mid point of leaflet2 and leaflet3 = {:2.4f}'.format(dP_23))
            print('Height of the valve = {:2.4f}'.format(height))
            print("Height/Circumference = %2.3f"%(height/circumference))
            print('Area (1) = {:2.4f}'.format(area1))
            print('Area (2) = {:2.4f}'.format(area2))
            print('Area (3) = {:2.4f}'.format(area3))

            print('Area (1) = {:2.4f}'.format(area1*scale**2))
            print('Area (2) = {:2.4f}'.format(area2*scale**2))
            print('Area (3) = {:2.4f}'.format(area3*scale**2))

        return result

class ProbeLeaflet():
    def __init__(self,spline,disp,leaflet_id,VERBOSE=False):
        '''
        leaflet = 1 (anterior), 2 (left), 3 (right)
        '''
        xmin = 2*leaflet_id-2
        xmax = 2*leaflet_id-1
        self.xmin = xmin
        self.xmax = xmax
        self.spline = spline
        mesh = spline.mesh
        self.mesh = mesh
        self.VERBOSE = VERBOSE
        class Leaflet(SubDomain):
            def inside(self,x,on_boundary):
                return x[0] <= xmax and x[0] >= xmin
        class FreeEdge(SubDomain):
            def inside(self,x,on_boundary):
                return on_boundary and x[0] <= xmax and x[0] >= xmin and near(x[1],1.)
        class BasalAttachment_0(SubDomain):
            '''
            Main part of the basal attachment
            '''
            def inside(self,x,on_boundary):
                return on_boundary and x[0] <= xmax and x[0] >= xmin and near(x[1],0.)
        class BasalAttachment_1(SubDomain):
            '''
            Commissure lines, part of the basal attachment
            '''
            def inside(self,x,on_boundary):
                return on_boundary and x[0] <= xmax and x[0] >= xmin and (near(x[0],xmin) or near(x[0],xmax))

        self.leaflet = Leaflet()
        self.free_edge = FreeEdge()
        self.basal_attachment_0 = BasalAttachment_0()
        self.basal_attachment_1 = BasalAttachment_1()
        
        boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1,0)
        boundary_markers.set_all(0)
        free_edge = self.free_edge
        basal_attachment_0 = self.basal_attachment_0
        basal_attachment_1 = self.basal_attachment_1
        free_edge.mark(boundary_markers,1)
        basal_attachment_0.mark(boundary_markers,2)
        basal_attachment_1.mark(boundary_markers,3)

        domain_markers = MeshFunction("size_t", mesh, mesh.topology().dim(),0)
        domain_markers.set_all(0)
        leaflet = self.leaflet
        leaflet.mark(domain_markers,1)
        
        self.boundary_markers = boundary_markers
        self.domain_markers = domain_markers

        F = spline.F+disp
        g = getMetric(F)
        F_proj = as_vector([spline.F[0]+disp[0],spline.F[1]+disp[1]])
        g_proj = getMetric(F_proj)
        
        self.F = F
        self.g = g
        self.F_proj = F_proj
        self.g_proj = g_proj

        # convert quadrilateral mesh to triangular meshes
        # so that interpolation is possible
        F0_func = project(F[0],spline.V_control)
        F1_func = project(F[1],spline.V_control)
        F2_func = project(F[2],spline.V_control)
        disp0_func = project(disp[0],spline.V_control)
        disp1_func = project(disp[1],spline.V_control)
        disp2_func = project(disp[2],spline.V_control)

        F0_func_tri = scalar_func_quad2tri(F0_func)
        F1_func_tri = scalar_func_quad2tri(F1_func)
        F2_func_tri = scalar_func_quad2tri(F2_func)
        disp0_func_tri = scalar_func_quad2tri(disp0_func)
        disp1_func_tri = scalar_func_quad2tri(disp1_func)
        disp2_func_tri = scalar_func_quad2tri(disp2_func)
        
        self.F0_func_tri = F0_func_tri
        self.F1_func_tri = F1_func_tri
        self.F2_func_tri = F2_func_tri
        self.disp0_func_tri = disp0_func_tri
        self.disp1_func_tri = disp1_func_tri
        self.disp2_func_tri = disp2_func_tri

    def run_leaflet(self):
        result = dict()
        l_fe = self.get_free_edge_length()
        # proper
        l_ba_proper, l_ba_proper_proj = self.get_basal_attachment_length_sub(2)
        # CA
        l_ba_commissure, l_ba_commissure_proj = self.get_basal_attachment_length_sub(3)
        l_ba, l_ba_proj = self.get_basal_attachment_length()
        l_R2_curve, l_R2_curve_proj = self.get_R2_curve_length()
        l_C2_curve, l_C2_curve_proj = self.get_C2_curve_length()
        area = self.get_leaflet_area()
        result['length_fe'] = l_fe
        result['length_ba_proper'] = l_ba_proper
        result['length_ba_commissure'] = l_ba_commissure
        result['length_ba'] = l_ba
        result['length_ba_proj'] = l_ba_proj
        result['length_R2'] = l_R2_curve
        result['length_C2'] = l_C2_curve
        result['R2_curve'] = self.get_R2_curve()
        result['C2_curve'] = self.get_C2_curve()
        result['area'] = area
        return result

    def get_R2_curve(self):
        xmin = self.xmin
        xmax = self.xmax

        F0_func_tri = self.F0_func_tri 
        F1_func_tri = self.F1_func_tri 
        F2_func_tri = self.F2_func_tri 
        disp0_func_tri = self.disp0_func_tri 
        disp1_func_tri = self.disp1_func_tri 
        disp2_func_tri = self.disp2_func_tri 

        #scalar_func_quad2tri(F0_func)
        R2_curve = []
        for x in np.linspace(0,1,40+1):
            pt = Point(xmin+0.5,x)
            F_pt = [F0_func_tri(pt),F1_func_tri(pt),F2_func_tri(pt)]
            disp_pt = [disp0_func_tri(pt),disp1_func_tri(pt),disp2_func_tri(pt)]
            R2_curve.append(F_pt)
            #print("point = [{},{},{}]".format(F_pt[0],F_pt[1],F_pt[2]))
            #print("disp = [{},{},{}]".format(disp_pt[0],disp_pt[1],disp_pt[2]))
        #print(R2_curve)
        return R2_curve
        
    def get_R2_curve_length(self):
        R2_curve = self.get_R2_curve()
        length_R2 = calc_curve_length(R2_curve)
        length_R2_proj = calc_curve_length_proj(R2_curve)
        # result checked with view_geo_design.py
        if self.VERBOSE:
            print("The length of R2 curve = %2.3f"%(length_R2))
            print("The length of R2 curve (proj) = %2.3f"%(length_R2_proj))
        return length_R2, length_R2_proj

    def get_C2_curve(self):
        xmin = self.xmin
        xmax = self.xmax

        F0_func_tri = self.F0_func_tri 
        F1_func_tri = self.F1_func_tri 
        F2_func_tri = self.F2_func_tri 
        disp0_func_tri = self.disp0_func_tri 
        disp1_func_tri = self.disp1_func_tri 
        disp2_func_tri = self.disp2_func_tri 

        #scalar_func_quad2tri(F0_func)
        C2_curve = []
        for x in np.linspace(0,1,40+1):
            pt = Point(xmin+x,0.5)
            F_pt = [F0_func_tri(pt),F1_func_tri(pt),F2_func_tri(pt)]
            disp_pt = [disp0_func_tri(pt),disp1_func_tri(pt),disp2_func_tri(pt)]
            C2_curve.append(F_pt)
        return C2_curve
        
    def get_C2_curve_length(self):
        C2_curve = self.get_C2_curve()
        length_C2 = calc_curve_length(C2_curve)
        length_C2_proj = calc_curve_length_proj(C2_curve)
        # result checked with view_geo_design.py
        if self.VERBOSE:
            print("The length of C2 curve = %2.3f"%(length_C2))
            print("The length of C2 curve (proj) = %2.3f"%(length_C2_proj))
        return length_C2, length_C2_proj

    def get_free_edge_length(self):
        ds = Measure("ds",subdomain_data=self.boundary_markers)
        F = self.F
        g = self.g
        F_proj = self.F_proj
        g_proj = self.g_proj
        length_fe = assemble(sqrt(g[0,0])*ds(1))
        # result checked with view_geo_design.py
        if self.VERBOSE:
            print("The length of free edge = %2.3f"%(length_fe))
        return length_fe

    def get_basal_attachment_length(self):
        ds = Measure("ds",subdomain_data=self.boundary_markers)
        F = self.F
        g = self.g
        F_proj = self.F_proj
        g_proj = self.g_proj
        length_ba = assemble(sqrt(g[0,0])*ds(2))+\
                assemble(sqrt(g[1,1])*ds(3))
        length_ba_proj = assemble(sqrt(g_proj[0,0])*ds(2))+\
                assemble(sqrt(g_proj[1,1])*ds(3))
        # result checked with view_geo_design.py
        if self.VERBOSE:
            print("The length of basal attachment = %2.3f"%(length_ba))
            print("The length of basal attachment (proj) = %2.3f"%(length_ba_proj))
        return length_ba, length_ba_proj

    def get_basal_attachment_length_sub(self,mode):
        ds = Measure("ds",subdomain_data=self.boundary_markers)
        F = self.F
        g = self.g
        F_proj = self.F_proj
        g_proj = self.g_proj
        if mode == 2:
            length_ba = assemble(sqrt(g[0,0])*ds(2))
            length_ba_proj = assemble(sqrt(g_proj[0,0])*ds(2))
            string = 'proper'
        if mode == 3:
            length_ba = assemble(sqrt(g[1,1])*ds(3))
            length_ba_proj = assemble(sqrt(g_proj[1,1])*ds(3))
            string = 'commissure'

        # result checked with view_geo_design.py
        if self.VERBOSE:
            print("The length of basal attachment (%s) = %2.3f"%(string,length_ba))
            print("The length of basal attachment (%s, proj) = %2.3f"%(string,length_ba_proj))
        return length_ba, length_ba_proj

    def get_leaflet_area(self):
        dx = Measure("dx",subdomain_data=self.domain_markers)
        F = self.F
        g = self.g
        F_proj = self.F_proj
        g_proj = self.g_proj
        area = assemble(sqrt(det(g))*dx(1))
        if self.VERBOSE:
            print("The area of the leaflet = %2.3f"%(area))
        return area 


