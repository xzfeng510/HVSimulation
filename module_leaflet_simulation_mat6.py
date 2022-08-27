import sys
import os

sys.path.append('/home/fenics/shared')
sys.path.append('/home/fenics/shared/tIGAr')
import igakit as ik
from tIGAr import *
from tIGAr.RhinoTSplines import *
from tIGAr.BSplines import *
from tIGAr.NURBS import *
from tIGAr.timeIntegration import *
from ShNAPr.SVK import *
from ShNAPr.contact import *
from ShNAPr.kinematics import *
from ShNAPr.hyperelastic import *
import meshio
import mshr
import fenics

import module_geo_design_v11
from module_mat import *
from util import *

import numpy as np
import copy
import time
from inspect import getmembers, isfunction
import json
import logging
from shutil import copyfile
from scipy.optimize import minimize, basinhopping, Bounds, LinearConstraint
from scipy.io import savemat
import random
import string
import csv
import pandas
from pyevtk.hl import gridToVTK

USE_ADAPTIVE_SCHEME = False 
global_config = dict()
global_config['COPYFILES'] = True # < setup > copy the current files into info directory if True
global_config['CHECKOUT'] = True # < checkout > output vtk files if True
global_config['OUTPUT_CS_PROFILES'] = False # < evaluate_cost > write cross sectional profiles if True
global_config['VERBOSE'] = True 
global_config['DOLFIN_LOG_LEVEL'] = 20
global_config['INCLUDE_MISMATCH_for_A'] = True
global_config['INCLUDE_MISMATCH_for_LR'] = True

if(mpisize > 1):
    if(mpirank == 0):
        raise("This code does not work in parallel.")

start = time.time()
_QUAD_DEG = 6 # default is 6
parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["representation"] = "tsfc"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = _QUAD_DEG
#parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
sys.setrecursionlimit(10000)

if global_config['VERBOSE']:
    sys.stdout = sys.__stdout__ # enable stdout
else:
    sys.stdout = open(os.devnull,'w')  # disable stdout


class leafletSim():
    def __init__(self,valve,material,contact,opt_params=[],ID=1,result_dir = []):
        format_header('leafletSim::__init__')
        '''
        Initialization
        '''
        fenics.set_log_level(global_config['DOLFIN_LOG_LEVEL'])
        #fenics.set_log_active(False)

        self.ID = ID
        self.ID_random = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        self.valve = valve
        self.bounding_planes_list = valve.generate_bounding_planes()
        self.material = material
        self.contact = contact
        self.opt_params = opt_params # opt_params talks with the optimization code. By default, opt_params is set to []. For logging purpose only. 
        if not result_dir:
            result_dir = os.getcwd()
        if result_dir[-1] is not '/':
            result_dir = result_dir+'/'
        self.result_dir = result_dir
        self.result_dir_str = sys.argv[1]
        self.message = sys.argv[2]
        self.vPatch = [0,1,2]
        self.nPatch = len(self.vPatch)
        self.thickness = 0.048
        self.d = 3
        self.pressure_max = 1.0 # correspond to 30mmHg
        self.PRESSURE = Constant(self.pressure_max)
        self.N_QUAD_PTS = 4 # default = 4
        #self.N_STEPS = 600 # actual run
        self.N_STEPS = 6000 # test only
        self.DELTA_T0 = 1./self.N_STEPS # default initial step
        N = 12 # default = 6
        self.checkout_list = [1.0*n/N*self.pressure_max for n in range(N+1)] # list of pressure at which results are to be checked out
        #self.stepper = MyLoadStepper(self.DELTA_T0,self.checkout_list) # global time (from 0 to 1)
        self.stepper = MyLoadStepper(self.DELTA_T0) # global time (from 0 to 1)
        #self.stepper_BC = Expression("t",t=0.,degree=0) 
        self.stepper_BC_r = Expression("t",t=0.,degree=0) 
        self.stepper_BC_z = Expression("t",t=0.,degree=0) 
        self.stepper_c0 = MyStabilizationStepper(t=0.)
        self.MIPE_EXIT_THRESHOLD = 10. # exit the simulation when average MIPE exceeds the threshold
        self.N1 = opt_params['global'] ['N1']
        self.N2 = opt_params['global'] ['N2']
        #self.BC = 'clamped' # 'clamped' or 'pinned'
        self.BC = 'pinned' # 'clamped' or 'pinned'
        self.k_cp = contact['0']['k_cp']
        self.GAP_OFFSET = contact['0']['GAP_OFFSET']

        self.parameters = dict()
        self.parameters['thickness'] = self.thickness
        self.parameters['N_QUAD_PTS'] = self.N_QUAD_PTS
        self.parameters['DELTA_T0'] = self.DELTA_T0
        self.parameters['MIPE_EXIT_THRESHOLD'] = self.MIPE_EXIT_THRESHOLD
        self.parameters['N1'] = self.N1
        self.parameters['N2'] = self.N2
        self.parameters['BC'] = self.BC
        self.parameters['_QUAD_DEG'] = _QUAD_DEG

        self.full_results = []
        self.full_results_geo = [] # list to record the result (geo) at different pressure
        self.full_results_mech = [] # list to record the result (mech) at different pressure
        self.current_pressure = 0.
        self.L2_error = dict()
        self.geometry_mismatch = dict()
        '''
        Setup the nonlinear solver
        '''
        #list_krylov_solver_methods()
        #list_krylov_solver_preconditioners()
        #list_linear_algebra_backends()
        #list_linear_solver_methods()
        #list_lu_solver_methods()
        #list_krylov_solver_preconditioners()
        solver = PETScSNESSolver()
        with open("./misc/solver_params_default.txt",'w') as f_default_params:
            for key in solver.parameters.keys():
                f_default_params.write('parameters[%s] = %s\r\n'%(key,solver.parameters[key]))
            f_default_params.flush()
        solver.parameters["linear_solver"] = "mumps"
        solver.parameters["relative_tolerance"] = 1e-2
        solver.parameters["maximum_iterations"] = 100
        solver.parameters["preconditioner"] = "ilu"
        solver.parameters["method"] = "newtonls"
        solver.parameters["line_search"] = "bt"
        solver.parameters["solution_tolerance"] = 1e-10
        #solver.parameters["krylov_solver"]["report"] = False 
        #solver.parameters["krylov_solver"]["error_on_nonconvergence"] = False
        #solver.parameters["krylov_solver"]["monitor_convergence"] = True 
        #solver.parameters["krylov_solver"]["nonzero_initial_guess"] = True
        #solver.parameters["lu_solver"]["report"] = True
        #solver.parameters["lu_solver"]["verbose"] = False

        solver_implicit = PETScSNESSolver()
        solver_implicit.parameters["linear_solver"] = "mumps"
        solver_implicit.parameters["relative_tolerance"] = 1e-3
        solver_implicit.parameters["maximum_iterations"] = 100
        solver_implicit.parameters["preconditioner"] = "ilu"
        solver_implicit.parameters["method"] = "newtonls"
        solver_implicit.parameters["line_search"] = "bt"
        solver_implicit.parameters["solution_tolerance"] = 1e-5
        if global_config['DOLFIN_LOG_LEVEL'] >= 30:
            solver.parameters['report'] = False
            solver_implicit.parameters['report'] = False
       
        with open("./misc/solver_params.txt",'w') as f_params:
            for key in solver.parameters.keys():
                f_params.write('parameters[%s] = %s\r\n'%(key,solver.parameters[key]))
            f_params.flush()
        with open("./misc/solver_implicit_params.txt",'w') as f_params:
            for key in solver_implicit.parameters.keys():
                f_params.write('parameters[%s] = %s\r\n'%(key,solver_implicit.parameters[key]))
            f_params.flush()
        
        self.solver = solver
        self.solver_implicit = solver_implicit
        '''
        Setup the output files 
        '''
        result_dir_str = self.result_dir_str        
        self.testFile = File(result_dir+"results/"+result_dir_str+"/test.pvd")
        self.testFile2 = File(result_dir+"results/"+result_dir_str+"/test2.pvd")
        self.testFile3 = File(result_dir+"results/"+result_dir_str+"/test3.pvd")

        self.y00File = File(result_dir+"results/"+result_dir_str+"/y0-x.pvd")
        self.y01File = File(result_dir+"results/"+result_dir_str+"/y0-y.pvd")
        self.y02File = File(result_dir+"results/"+result_dir_str+"/y0-z.pvd")
        self.d0File = File(result_dir+"results/"+result_dir_str+"/disp-x.pvd")
        self.d1File = File(result_dir+"results/"+result_dir_str+"/disp-y.pvd")
        self.d2File = File(result_dir+"results/"+result_dir_str+"/disp-z.pvd")
        self.F0File = File(result_dir+"results/"+result_dir_str+"/F-x.pvd")
        self.F1File = File(result_dir+"results/"+result_dir_str+"/F-y.pvd")
        self.F2File = File(result_dir+"results/"+result_dir_str+"/F-z.pvd")
        self.F3File = File(result_dir+"results/"+result_dir_str+"/F-w.pvd")
        self.E2D_tr_File = File(result_dir+"results/"+result_dir_str+"/E2D-tr.pvd")
        self.E2D_det_File = File(result_dir+"results/"+result_dir_str+"/E2D-det.pvd")
        self.E00File = File(result_dir+"results/"+result_dir_str+"/E-11.pvd")
        self.E11File = File(result_dir+"results/"+result_dir_str+"/E-22.pvd")
        self.E01File = File(result_dir+"results/"+result_dir_str+"/E-12.pvd")
        self.E10File = File(result_dir+"results/"+result_dir_str+"/E-21.pvd")
        self.EttFile = File(result_dir+"results/"+result_dir_str+"/E-tt.pvd")
        self.EtrFile = File(result_dir+"results/"+result_dir_str+"/E-tr.pvd")
        self.ErtFile = File(result_dir+"results/"+result_dir_str+"/E-rt.pvd")
        self.ErrFile = File(result_dir+"results/"+result_dir_str+"/E-rr.pvd")
        self.EwwFile = File(result_dir+"results/"+result_dir_str+"/E-ww.pvd")
        self.JFile = File(result_dir+"results/"+result_dir_str+"/J.pvd")
        self.SttFile = File(result_dir+"results/"+result_dir_str+"/S-tt.pvd")
        self.StrFile = File(result_dir+"results/"+result_dir_str+"/S-tr.pvd")
        self.SrtFile = File(result_dir+"results/"+result_dir_str+"/S-rt.pvd")
        self.SrrFile = File(result_dir+"results/"+result_dir_str+"/S-rr.pvd")
        self.mipeFile = File(result_dir+"results/"+result_dir_str+"/MIPE.pvd")
        self.LrFile = File(result_dir+"results/"+result_dir_str+"/Lambda-r.pvd")
        self.LtFile = File(result_dir+"results/"+result_dir_str+"/Lambda-t.pvd")
        self.LwFile = File(result_dir+"results/"+result_dir_str+"/Lambda-w.pvd")
        self.A_det_File = File(result_dir+"results/"+result_dir_str+"/A_det_ref.pvd")
        self.a_det_File = File(result_dir+"results/"+result_dir_str+"/a_det_cur.pvd")
        self.Art_0_File = File(result_dir+"results/"+result_dir_str+"/Angle-0-rt.pvd")
        self.Art_File = File(result_dir+"results/"+result_dir_str+"/Angle-rt.pvd")
        self.a0File = File(result_dir+"results/"+result_dir_str+"/a0.pvd")
        self.a1File = File(result_dir+"results/"+result_dir_str+"/a1.pvd")
        self.a2File = File(result_dir+"results/"+result_dir_str+"/a2.pvd")
        self.wFile = File(result_dir+"results/"+result_dir_str+"/w.pvd")
        self.bcFile = File(result_dir+"results/"+self.result_dir_str+"/bc.pvd")
        self.r0File = File(self.result_dir+"results/"+self.result_dir_str+"/r-x.pvd")
        self.r1File = File(self.result_dir+"results/"+self.result_dir_str+"/r-y.pvd")
        self.r2File = File(self.result_dir+"results/"+self.result_dir_str+"/r-z.pvd")
        self.c0File = File(self.result_dir+"results/"+self.result_dir_str+"/c-x.pvd")
        self.c1File = File(self.result_dir+"results/"+self.result_dir_str+"/c-y.pvd")
        self.c2File = File(self.result_dir+"results/"+self.result_dir_str+"/c-z.pvd")

        self.c0File_lower = File(self.result_dir+"results/"+self.result_dir_str+"/c-x-lower.pvd")
        self.c1File_lower = File(self.result_dir+"results/"+self.result_dir_str+"/c-y-lower.pvd")
        self.c2File_lower = File(self.result_dir+"results/"+self.result_dir_str+"/c-z-lower.pvd")
        self.c0File_upper = File(self.result_dir+"results/"+self.result_dir_str+"/c-x-upper.pvd")
        self.c1File_upper = File(self.result_dir+"results/"+self.result_dir_str+"/c-y-upper.pvd")
        self.c2File_upper = File(self.result_dir+"results/"+self.result_dir_str+"/c-z-upper.pvd")
 
        self.domain_material_File = File(result_dir+"results/"+self.result_dir_str+"/domain_material.pvd")
        self.penetration_File = File(result_dir+"results/"+self.result_dir_str+"/penetration.pvd")
        self.energySurfaceDensityFile = File(result_dir+"results/"+result_dir_str+"/energySurfaceDensity.pvd")
        self.energySurfaceDensity1File = File(result_dir+"results/"+result_dir_str+"/energySurfaceDensity1.pvd")
        self.energySurfaceDensity2File = File(result_dir+"results/"+result_dir_str+"/energySurfaceDensity2.pvd")
        self.energySurfaceDensity3File = File(result_dir+"results/"+result_dir_str+"/energySurfaceDensity3.pvd")

        self.resultsFile = os.path.join(result_dir,"results/"+result_dir_str+"/results.json")
        self.resultsFile_geo = os.path.join(result_dir,"results/"+result_dir_str+"/results_geo.json")
        self.resultsFile_mech = os.path.join(result_dir,"results/"+result_dir_str+"/results_mech.json")
        self.contactFiles_base = os.path.join(result_dir,'results',result_dir_str,'saved_contact_pairs')

        '''
        Setup the log file which contains critical info to keep track of the current simulation
        w: only include the current simulation
        a+: include all simulation
        '''
        self.logFile = open(result_dir+'results/'+result_dir_str+"/log_simulation","a+")
        self.logFile.write('-'*30+'\n')
        #TODO: save key optimization parameters here 
        self.logFile.write('Valve ID = {} ({})\n'.format(self.ID,self.ID_random))
        self.logFile.write('PID = {}\n'.format(os.getpid()))
        self.logFile.write("output dir = results/"+self.result_dir_str+'\n')
        self.logFile.write("BC = "+self.BC+'\n')
        self.logFile.write(json.dumps(self.opt_params))
        self.logFile.write('\n')
        self.logFile.write(json.dumps(global_config))
        self.logFile.write('\n')
        self.logFile.flush()

        print('*'*30)
        print('Valve ID = {} ({})'.format(self.ID,self.ID_random))
        print('PID = {}'.format(os.getpid()))
        print('message = {}'.format(self.message))
        print('start time = {}'.format(time.strftime("%H:%M:%S",time.localtime())))
        print('N_STEPS = {}'.format(self.N_STEPS))
        print('*'*15)
        print("output dir = "+os.path.join(self.result_dir,"results/",self.result_dir_str))
        print("use adapative scheme = {}".format(USE_ADAPTIVE_SCHEME))
        print('>>> Parameters for the simulation:')
        print(self.parameters)
        print(self.opt_params)
        print(contact)
        print('>>> Valve bounding planes:')
        print(self.bounding_planes_list)
        sys.stdout.flush()
        '''
        Copy used files for record 
        '''
        if global_config['COPYFILES']:
            copyfile('./misc/solver_params_default.txt',os.path.join(result_dir,"results/"+self.result_dir_str,'info','solver_params_default.txt'))
            copyfile('./misc/solver_params.txt',os.path.join(result_dir,"results/"+self.result_dir_str,'info','solver_params.txt'))
            copyfile('./misc/solver_implicit_params.txt',os.path.join(result_dir,"results/"+self.result_dir_str,'info','solver_implicit_params.txt'))
            copyfile('./module_leaflet_simulation.py',os.path.join(result_dir,"results/"+self.result_dir_str,'info','module_leaflet_simulation.py'))
            copyfile('./module_geo_design_v11.py',os.path.join(result_dir,"results/"+self.result_dir_str,'info','module_geo_design_v11.py'))
            copyfile('./util.py',os.path.join(result_dir,"results/"+self.result_dir_str,'info','util.py'))

    def init_mesh(self, prefix='./mesh/leaflet_tmp_'):
        format_header('leafletSim::init_mesh')
        '''
        Initialize the IGA mesh with tiGAr and set up
        the boundary condition
        '''
        print('Load mesh from '+prefix+'*.dat')
        nPatch = self.nPatch 
        print('nPatch = {:}'.format(nPatch))
        d = self.d
        controlMesh = LegacyMultipatchControlMesh(prefix, nPatch, '.dat')

        splineGenerator = EqualOrderSpline(d, controlMesh)
        scalarSpline = splineGenerator.getControlMesh().getScalarSpline()

        # Set boundary conditions
        # TODO: double check boundary condition!
        # Pinned condition takes longer time to converge
        nLayers = 0
        if self.BC == 'pinned':
            print('pinned BC is used')
            nLayers = 1
        if self.BC == 'clamped':
            print('clamped BC is used')
            nLayers = 2
        if not nLayers:
            print('Unrecognized boundary condition!')
            exit()

        for patch in range(nPatch):
            for side, direction in zip([0,1,0],[1,0,0]):
                # if nLayers=1, pinned BC is applied
                # if nLayers=2, clamped BC is applied
                sideDofs = scalarSpline.getPatchSideDofs(
                        patch, direction, side, nLayers=nLayers)
                for i in range(0, d):
                    splineGenerator.addZeroDofs(i, sideDofs)
        
        # Write the extraction data.
        DIR = "./extraction"
        splineGenerator.writeExtraction(DIR)

        print("Forming extracted spline...")
        # Read an extracted spline back in.
        QUAD_DEG = _QUAD_DEG
        spline = ExtractedSpline(splineGenerator, QUAD_DEG)
        # Allow many nonlinear iterations.
        spline.maxIters = 1000

        self.splineGenerator = splineGenerator
        self.scalarSpline = scalarSpline
        self.spline = spline

    def init_mesh_mp_version(self):
        format_header('leafletSim::init_mesh_mp_version')

        nPatch = self.nPatch 
        print('nPatch = {:}'.format(nPatch))
        d = self.d

        splineGenerator = self.splineGenerator
        scalarSpline = splineGenerator.getControlMesh().getScalarSpline()

        # Set boundary conditions
        # TODO: double check boundary condition!
        # Pinned condition takes longer time to converge
        nLayers = 0
        if self.BC == 'pinned':
            print('pinned BC is used')
            nLayers = 1
        if self.BC == 'clamped':
            print('clamped BC is used')
            nLayers = 2
        if not nLayers:
            print('Unrecognized boundary condition!')
            exit()

        for patch in range(nPatch):
            for side, direction in zip([0,1,0],[1,0,0]):
                # if nLayers=1, pinned BC is applied
                # if nLayers=2, clamped BC is applied
                sideDofs = scalarSpline.getPatchSideDofs(
                        patch, direction, side, nLayers=nLayers)
                for i in range(0, d):
                    splineGenerator.addZeroDofs(i, sideDofs)
        
        print("Forming extracted spline...")
        # Read an extracted spline back in.
        QUAD_DEG = _QUAD_DEG
        spline = ExtractedSpline(splineGenerator, QUAD_DEG)
        # Allow many nonlinear iterations.
        spline.maxIters = 1000

        self.splineGenerator = splineGenerator
        self.scalarSpline = scalarSpline
        self.spline = spline

    def init_contact(self):
        format_header('leafletSim::init_contact')
        
        result = self.postprocess_geometry()
        if self.nPatch == 3:
            dL_12 = result['monitor']['distance_leaflets_12']
            dL_13 = result['monitor']['distance_leaflets_13']
            dL_23 = result['monitor']['distance_leaflets_23']
        if self.nPatch == 1:
            dL_12 = 0.039
            dL_13 = 0.039
            dL_23 = 0.039
        '''
        print(dL_12)
        print(dL_13)
        print(dL_23)
        exit()
        '''
        min_dist = np.min([dL_12,dL_13,dL_23])
        #min_dist = 0.06
        self.min_dist0 = min_dist
        print('Initial gap among neighboring leaflets = {:3.3}'.format(min_dist))
        if min_dist < 1e-3:
            print('Leaflets are too close to each other initially.')
            exit()

        contact = self.contact
        spline = self.spline
        
        for key in contact:
            if key is '0':
                continue
            print('Loaded contact model = %s'%(key))
            contact_model = int(key)
            params = contact[key]

        if contact_model == 2:
            '''
            Initialize the contact problem context object
            '''
            r_in_scheme2 = min_dist*params['r_in_rel']
            r_out_scheme2 = min_dist*params['r_out_rel']
            k_scheme2 = params['k_c']            
            p_scheme2 = params['p']
            R_self_scheme2 = min_dist*params['R_self_rel']
 
            c1_scheme2 = p_scheme2*k_scheme2/2/(r_out_scheme2-r_in_scheme2)/pow(r_in_scheme2,p_scheme2+1)
            c2_scheme2 = k_scheme2/pow(r_in_scheme2,p_scheme2)-c1_scheme2*pow(r_in_scheme2-r_out_scheme2,2)

            def phiPrime_scheme2(r):
                if(r > r_out_scheme2):
                    return 0.0
                elif(r < r_in_scheme2):
                    return -k_scheme2/pow(r,p_scheme2)+c2_scheme2
                else:
                    return -c1_scheme2*pow(r-r_out_scheme2,2)
   
            def phiDoublePrime_scheme2(r):
                if(r > r_out_scheme2):
                    return 0.0
                elif(r < r_in_scheme2):
                    return k_scheme2*p_scheme2/pow(r,p_scheme2+1)
                else:
                    return -2*c1_scheme2*(r-r_out_scheme2)

            self.contactContext = MyShellContactContext(spline, R_self_scheme2, r_out_scheme2, 
                                                   phiPrime_scheme2, phiDoublePrime_scheme2)

    def define_target_geometry_info(self,p):
        tgi = dict()
        tgi['pressure'] = p
        if p == 0:
            # DM derived
            tgi['annular_perimeter_length'] = 4455 
            # tgi['valve_height'] = NA 
            tgi['free_edge_length_A'] = 1192 
            tgi['free_edge_length_LR'] = 1307
            tgi['basal_attachment_length_A'] = 1446
            tgi['basal_attachment_length_LR'] = 1533
            tgi['commissure_attachment_length_A'] = 365
            tgi['commissure_attachment_length_LR'] = 403
            #tgi['commissure_point_distance_A'] = 663
            #tgi['commissure_point_distance_LR'] = 789
            tgi['leaflet_surface_area_A'] = 4.068*1e5*1.04
            tgi['leaflet_surface_area_LR'] = 5.120*1e5/1.04
        if p == 10:
            # mPV-IGM derived
            tgi['annular_perimeter_length'] = 4785 
            tgi['valve_height'] =  524
            tgi['free_edge_length_A'] = 1343
            tgi['free_edge_length_LR'] = 1549
            tgi['basal_attachment_length_A'] = 1356 
            tgi['basal_attachment_length_LR'] = 1715
            tgi['commissure_attachment_length_A'] = 367
            tgi['commissure_attachment_length_LR'] = 407
            tgi['commissure_point_distance_A'] = 858
            tgi['commissure_point_distance_LR'] = 1019
            tgi['leaflet_surface_area_A'] = 6.7510*1e5
            tgi['leaflet_surface_area_LR'] = 8.2653*1e5
        if p == 20:
            # mPV-IGM derived
            tgi['annular_perimeter_length'] = 5547 
            tgi['valve_height'] = 644
            tgi['free_edge_length_A'] = 1478
            tgi['free_edge_length_LR'] = 1728
            tgi['basal_attachment_length_A'] = 1505 
            tgi['basal_attachment_length_LR'] = 2021
            tgi['commissure_attachment_length_A'] = 377
            tgi['commissure_attachment_length_LR'] = 415
            tgi['commissure_point_distance_A'] = 1071
            tgi['commissure_point_distance_LR'] = 1245
            tgi['leaflet_surface_area_A'] = 6.3618*1e5
            tgi['leaflet_surface_area_LR'] = 9.8376*1e5
        if p == 30:
            # mPV-IGM derived
            tgi['annular_perimeter_length'] = 5181
            tgi['valve_height'] = 602
            tgi['free_edge_length_A'] = 1381
            tgi['free_edge_length_LR'] = 1463
            tgi['basal_attachment_length_A'] = 1574
            tgi['basal_attachment_length_LR'] = 1804
            tgi['commissure_attachment_length_A'] = 351
            tgi['commissure_attachment_length_LR'] = 388
            tgi['commissure_point_distance_A'] = 1123
            tgi['commissure_point_distance_LR'] = 1124
            tgi['leaflet_surface_area_A'] = 6.3644*1e5
            tgi['leaflet_surface_area_LR'] = 7.9181*1e5
        return tgi

    def get_geometry_info(self,p):
        print('Geometry info:')
        mismatch = 0
        tgi = self.define_target_geometry_info(p)
        print(tgi)

        if p == 0:
            scale = self.opt_params['global']['scale']
            print('Scale = %4.2fum.'% (scale))
            v = self.valve.get_height(verbose=False)
            vt = float('nan')
            print_info('H',vt,v*scale)
            v = self.valve.get_annular_perimeter_length(verbose=False)
            vt = tgi['annular_perimeter_length']
            print_info('Lp (matched)',vt,v*scale)
            v = self.valve.get_PA_annular_perimeter_length(verbose=False)
            vt = float('nan')
            print_info('perimeter_PA_ANN',vt,v*scale)
            v = self.valve.get_PA_STJ_perimeter_length(verbose=False)
            vt = float('nan')
            print_info('perimeter_PA_STJ',vt,v*scale)
     
            v = self.valve.get_free_edge_length(0,verbose=False)
            vt = tgi['free_edge_length_A']
            print_info('L_FE_A (matched)',vt,v*scale)
            if global_config['INCLUDE_MISMATCH_for_A']:
                mismatch = mismatch + (v*scale-vt)**2
            v = self.valve.get_free_edge_length(1,verbose=False)
            vt = tgi['free_edge_length_LR'] 
            print_info('L_FE_LR (matched)',vt,v*scale)
            if global_config['INCLUDE_MISMATCH_for_LR']:
                mismatch = mismatch + (v*scale-vt)**2
     
            v = self.valve.get_basal_attachment_length(0,'total',verbose=False)
            vt = float('nan')
            print_info('L_BA_A+L_CA_A',vt,v*scale)
            v = self.valve.get_basal_attachment_length(0,'proper',verbose=False)
            vt = tgi['basal_attachment_length_A']
            print_info('L_BA_A (matched)',vt,v*scale)
            v = self.valve.get_basal_attachment_length(0,'comm',verbose=False)
            vt = tgi['commissure_attachment_length_A']
            print_info('L_CA_A (matched)',vt,v*scale)
            v = self.valve.get_basal_attachment_length(1,'total',verbose=False)
            vt = float('nan')
            print_info('L_BA_LR+L_CA_LR',vt,v*scale)
            v = self.valve.get_basal_attachment_length(1,'proper',verbose=False)
            vt = tgi['basal_attachment_length_LR']
            print_info('L_BA_LR (matched)',vt,v*scale)
            v = self.valve.get_basal_attachment_length(1,'comm',verbose=False)
            vt = tgi['commissure_attachment_length_LR']
            print_info('L_CA_LR (matched)',vt,v*scale)

        if p in [10,20,30]:
            scale = self.opt_params['global']['scale']
            print('Scale = %4.2fum.'% (scale))
            v = self.valve.get_height(verbose=False)
            vt = tgi['valve_height']
            print_info('L_H',vt,v*scale)
            v = self.valve.get_annular_perimeter_length(verbose=False)
            vt = tgi['annular_perimeter_length']
            print_info('Lp',vt,v*scale)
            v = self.valve.get_PA_annular_perimeter_length(verbose=False)
            vt = float('nan')
            print_info('perimeter_PA_ANN',vt,v*scale)
            v = self.valve.get_PA_STJ_perimeter_length(verbose=False)
            vt = float('nan')
            print_info('perimeter_PA_STJ',vt,v*scale)
     
            v = self.valve.get_free_edge_length(0,verbose=False)
            vt = tgi['free_edge_length_A']
            print_info('L_FE_A (matched)',vt,v*scale)
            if global_config['INCLUDE_MISMATCH_for_A']:
                mismatch = mismatch + (v*scale-vt)**2
            v = self.valve.get_free_edge_length(1,verbose=False)
            vt = tgi['free_edge_length_LR'] 
            print_info('L_FE_LR (matched)',vt,v*scale)
            if global_config['INCLUDE_MISMATCH_for_LR']:
                mismatch = mismatch + (v*scale-vt)**2
     
            v = self.valve.get_basal_attachment_length(0,'total',verbose=False)
            vt = float('nan')
            print_info('L_BA_A+L_CA_A',vt,v*scale)
            v = self.valve.get_basal_attachment_length(0,'proper',verbose=False)
            vt = tgi['basal_attachment_length_A']
            print_info('L_BA_A',vt,v*scale)
            v = self.valve.get_basal_attachment_length(0,'comm',verbose=False)
            vt = tgi['commissure_attachment_length_A']
            print_info('L_CA_A',vt,v*scale)
            v = self.valve.get_basal_attachment_length(1,'total',verbose=False)
            vt = float('nan')
            print_info('L_BA_A+L_CA_A',vt,v*scale)
            v = self.valve.get_basal_attachment_length(1,'proper',verbose=False)
            vt = tgi['basal_attachment_length_LR']
            print_info('L_BA_LR',vt,v*scale)
            v = self.valve.get_basal_attachment_length(1,'comm',verbose=False)
            vt = tgi['commissure_attachment_length_LR']
            print_info('L_CA_LR',vt,v*scale)
            
            v = self.valve.get_commissure_point_distance(0,verbose=False)
            vt = tgi['commissure_point_distance_A']
            print_info('d_comm_A',vt,v*scale)
            v = self.valve.get_commissure_point_distance(1,verbose=False)
            vt = tgi['commissure_point_distance_LR']
            print_info('d_comm_LR',vt,v*scale)
           
        print('mismatch = %f'%(sqrt(mismatch)))
        return mismatch

    def setup(self,DELTA_T):

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
 
        format_header('leafletSim::setup')

        spline = self.spline
        y_hom = Function(spline.V) 
        y_old_hom = Function(spline.V)
        DELTA_T = 1./self.N_STEPS
        ydot_hom = Constant(1.0/DELTA_T)*y_hom+Constant(-1.0/DELTA_T)*y_old_hom
        ydot_old_hom = Function(spline.V)
        yddot_hom = (ydot_hom-ydot_old_hom)/DELTA_T
        y = spline.rationalize(y_hom) 
        ydot = spline.rationalize(ydot_hom)
        yddot = spline.rationalize(yddot_hom)

        self.y_hom = y_hom
        self.y = y
        self.y_old_hom = y_old_hom
        self.ydot_hom = ydot_hom
        self.ydot_old_hom = ydot_old_hom
        self.yddot_hom = yddot_hom
        self.ydot = ydot
        self.yddot = yddot
 
        # Reference configuration:
        X = spline.F
        # Current configuration:
        x = X + y
        self.X = X
        self.x = x

        # Obtain shell geometry for reference and current configuration midsurfaces
        A0, A1, A2, deriv_A2, A, B = midsurfaceGeometry(X, spline)
        a0, a1, a2, deriv_a2, a, b = midsurfaceGeometry(x, spline)

        thickness = self.thickness
        #h_th = Constant(thickness)
        h_th = thickness_model(spline,thickness)
        self.h_th = h_th
 
        x_lower = x + a2*h_th/2.
        x_upper = x - a2*h_th/2.
        self.x_lower = x_lower
        self.x_upper = x_upper

        #tmp = project(A[0,0],spline.V_control) 
        #print(tmp.vector().norm('linf'))
        #tmp = project(A[1,1],spline.V_control) 
        #print(tmp.vector().norm('linf'))

        self.A0 = A0
        self.A1 = A1
        self.A2 = A2
        self.deriv_A2 = deriv_A2
        self.A = A
        self.B = B

        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.deriv_a2 = deriv_a2
        self.a = a
        self.b = b
 
        #print(os.path.abspath(inspect.getfile(metricKL)))
        G = metricKL(A,B,0)
        g = metricKL(a,b,0)
        E_flat = 0.5*(g - G)
        G0,G1 = curvilinearBasisKL(A0,A1,deriv_A2,0)
        E_2D = covariantRank2TensorToCartesian2D(E_flat,G,G0,G1)
        C_2D = 2.0*E_2D + Identity(2)
        C22 = 1.0/det(C_2D)
        E22 = 0.5*(C22-1.0)
        E = as_tensor([[E_2D[0,0], E_2D[0,1], 0.0],
                       [E_2D[1,0], E_2D[1,1], 0.0],
                       [0.0,       0.0,       E22]])
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        I4 = C[0,0]

        self.G = G
        self.g = g
        self.E_flat = E_flat
        self.G0 = G0
        self.G1 = G1
        self.E_2D = E_2D
        self.C_2D = C_2D
        self.E = E
        self.C = C
        self.I1 = I1
        self.I4 = I4
        self.J = sqrt(det(C))

        # Return a 3D elastic strain energy density, given E in Cartesian coordinates.
        '''
        cs_A = self.material['4']['A']
        cs_LR = self.material['4']['LR']
        '''
        '''
        cs_A = self.material['5']['A']
        cs_LR = self.material['5']['LR']
        domain_material = coaptation_model(spline)
        self.domain_material = domain_material
        '''
        cs_A = self.material['6']['A']
        cs_LR = self.material['6']['LR']
        domain_material = coaptation_model(spline)
        self.domain_material = domain_material
 
        '''
        # material = 4
        c0 = conditional(le(x0,1.5),cs_A['c0'],conditional(ge(x0,3.5),cs_LR['c0'],cs_LR['c0']))*self.stepper_c0
        c1 = conditional(le(x0,1.5),cs_A['c1'],conditional(ge(x0,3.5),cs_LR['c1'],cs_LR['c1']))
        c2 = conditional(le(x0,1.5),cs_A['c2'],conditional(ge(x0,3.5),cs_LR['c2'],cs_LR['c2']))
        c3 = conditional(le(x0,1.5),cs_A['c3'],conditional(ge(x0,3.5),cs_LR['c3'],cs_LR['c3']))
        c4 = conditional(le(x0,1.5),cs_A['c4'],conditional(ge(x0,3.5),cs_LR['c4'],cs_LR['c4']))
        '''
        '''
        # material = 5
        c0_c = conditional(le(x0,1.5),cs_A['c0_c'],conditional(ge(x0,3.5),cs_LR['c0_c'],cs_LR['c0_c']))*self.stepper_c0
        c1_c = conditional(le(x0,1.5),cs_A['c1_c'],conditional(ge(x0,3.5),cs_LR['c1_c'],cs_LR['c1_c']))
        c2_c = conditional(le(x0,1.5),cs_A['c2_c'],conditional(ge(x0,3.5),cs_LR['c2_c'],cs_LR['c2_c']))
        c3_c = conditional(le(x0,1.5),cs_A['c3_c'],conditional(ge(x0,3.5),cs_LR['c3_c'],cs_LR['c3_c']))
        c4_c = conditional(le(x0,1.5),cs_A['c4_c'],conditional(ge(x0,3.5),cs_LR['c4_c'],cs_LR['c4_c']))

        c0_b = conditional(le(x0,1.5),cs_A['c0_b'],conditional(ge(x0,3.5),cs_LR['c0_b'],cs_LR['c0_b']))*self.stepper_c0
        c1_b = conditional(le(x0,1.5),cs_A['c1_b'],conditional(ge(x0,3.5),cs_LR['c1_b'],cs_LR['c1_b']))
        c2_b = conditional(le(x0,1.5),cs_A['c2_b'],conditional(ge(x0,3.5),cs_LR['c2_b'],cs_LR['c2_b']))
        c3_b = conditional(le(x0,1.5),cs_A['c3_b'],conditional(ge(x0,3.5),cs_LR['c3_b'],cs_LR['c3_b']))
        c4_b = conditional(le(x0,1.5),cs_A['c4_b'],conditional(ge(x0,3.5),cs_LR['c4_b'],cs_LR['c4_b']))

        c0 = c0_c + (c0_b-c0_c)*domain_material
        c1 = c1_c + (c1_b-c1_c)*domain_material
        c2 = c2_c + (c2_b-c2_c)*domain_material
        c3 = c3_c + (c3_b-c3_c)*domain_material
        c4 = c4_c + (c4_b-c4_c)*domain_material

        C = 2.0*E + Identity(3)
        I1 = tr(C)
        I4 = C[0,0]
        
        return 0.5*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1) + 0.5*c3*(exp(c4*pow(I4 - 1.0,2.0))-1)
        #return 0.5*c0*(I1-3.0) + c1*tr(E*E)
        '''
        # material = 6
        x0, x1 = SpatialCoordinate(spline.mesh)
        b0_c = conditional(le(x0,1.5),cs_A['b0_c'],conditional(ge(x0,3.5),cs_LR['b0_c'],cs_LR['b0_c']))*self.stepper_c0
        b1_c = conditional(le(x0,1.5),cs_A['b1_c'],conditional(ge(x0,3.5),cs_LR['b1_c'],cs_LR['b1_c']))
        b2_c = conditional(le(x0,1.5),cs_A['b2_c'],conditional(ge(x0,3.5),cs_LR['b2_c'],cs_LR['b2_c']))
        c0_c = conditional(le(x0,1.5),cs_A['c0_c'],conditional(ge(x0,3.5),cs_LR['c0_c'],cs_LR['c0_c']))
        c1_c = conditional(le(x0,1.5),cs_A['c1_c'],conditional(ge(x0,3.5),cs_LR['c1_c'],cs_LR['c1_c']))
        c2_c = conditional(le(x0,1.5),cs_A['c2_c'],conditional(ge(x0,3.5),cs_LR['c2_c'],cs_LR['c2_c']))
        c3_c = conditional(le(x0,1.5),cs_A['c3_c'],conditional(ge(x0,3.5),cs_LR['c3_c'],cs_LR['c3_c']))
        c4_c = conditional(le(x0,1.5),cs_A['c4_c'],conditional(ge(x0,3.5),cs_LR['c4_c'],cs_LR['c4_c']))

        b0_b = conditional(le(x0,1.5),cs_A['b0_b'],conditional(ge(x0,3.5),cs_LR['b0_b'],cs_LR['b0_b']))*self.stepper_c0
        b1_b = conditional(le(x0,1.5),cs_A['b1_b'],conditional(ge(x0,3.5),cs_LR['b1_b'],cs_LR['b1_b']))
        b2_b = conditional(le(x0,1.5),cs_A['b2_b'],conditional(ge(x0,3.5),cs_LR['b2_b'],cs_LR['b2_b']))
        c0_b = conditional(le(x0,1.5),cs_A['c0_b'],conditional(ge(x0,3.5),cs_LR['c0_b'],cs_LR['c0_b']))
        c1_b = conditional(le(x0,1.5),cs_A['c1_b'],conditional(ge(x0,3.5),cs_LR['c1_b'],cs_LR['c1_b']))
        c2_b = conditional(le(x0,1.5),cs_A['c2_b'],conditional(ge(x0,3.5),cs_LR['c2_b'],cs_LR['c2_b']))
        c3_b = conditional(le(x0,1.5),cs_A['c3_b'],conditional(ge(x0,3.5),cs_LR['c3_b'],cs_LR['c3_b']))
        c4_b = conditional(le(x0,1.5),cs_A['c4_b'],conditional(ge(x0,3.5),cs_LR['c4_b'],cs_LR['c4_b']))

        b0 = b0_c + (b0_b-b0_c)*domain_material
        b1 = b1_c + (b1_b-b1_c)*domain_material
        b2 = b2_c + (b2_b-b2_c)*domain_material
        c0 = c0_c + (c0_b-c0_c)*domain_material
        c1 = c1_c + (c1_b-c1_c)*domain_material
        c2 = c2_c + (c2_b-c2_c)*domain_material
        c3 = c3_c + (c3_b-c3_c)*domain_material
        c4 = c4_c + (c4_b-c4_c)*domain_material

        def incompressibleKL_mat6(spline,X,x):
            '''
                diff does not work for this case
                use analytic expressions
            '''
            # material = 6
            C_inv = inv(C)
            e0,e1 = localCartesianBasis(A0, A1)                
            e2 = unit(cross(e0,e1))
            A0_newbasis = change2basis(unit(A0),e0,e1,e2)     
            A1_newbasis = change2basis(unit(A1),e0,e1,e2) 
            
            Ec = inner(A0_newbasis,E*A0_newbasis)
            Er = inner(A1_newbasis,E*A1_newbasis)
            Ephi = inner(A0_newbasis,E*A1_newbasis)
            Ez = inner(e2,E*e2)
            
            def psi(xi2):
                G = metricKL(A,B,xi2)
                g = metricKL(a,b,xi2)
                E_flat = 0.5*(g - G)
                G0,G1 = curvilinearBasisKL(A0,A1,deriv_A2,xi2)
                E_2D = covariantRank2TensorToCartesian2D(E_flat,G,G0,G1)
                #E_2D = covariantRank2TensorToCartesian2D(E_flat,A,A0,A1)
                C_2D = 2.0*E_2D + Identity(2)
                C22 = 1.0/det(C_2D)
                E22 = 0.5*(C22-1.0)
                E = as_tensor([[E_2D[0,0], E_2D[0,1], 0.0],
                               [E_2D[1,0], E_2D[1,1], 0.0],
                               [0.0,       0.0,       E22]])
                C = 2.0*E + Identity(3)
                J = sqrt(det(C))
                
                return 0.5*b0*(I1-3.0) + 0.5*(b1*exp(b2*(I1-3)**2)-1) + 0.5*c0*(exp(c1*Ec**2+c2*Er**2+c3*Ephi**2+2*c4*Ec*Er)-1)
 
            Q = c1*Ec**2+c2*Er**2+c3*Ephi**2+2*c4*Ec*Er
 
            dQ_dE = as_tensor([[2*c1*Ec+2*c4*Er, 2*c3*Ephi, Constant(0.)],
                    [2*c3*Ephi, 2*c2*Er+2*c4*Ec, Constant(0.)],
                    [Constant(0.), Constant(0.), Constant(0.)]]
                    )
            dpsi_dI1 = b1*b2*(I1-3)*exp(b2*(I1-3)**2) 
            dpsi_dQ = 0.5*c0*exp(Q)
            dpsi_el_dE = 2*dpsi_dI1*Identity(3) + dpsi_dQ*dQ_dE 
            C22  = 2.0*E[2,2] + 1.0
            p = C22*dpsi_el_dE[2,2]

            S = dpsi_el_dE - p*C_inv
            
            return psi, S

        penetration = Function(spline.V)
        self.penetration = penetration

        h_th = self.h_th
        # Obtain a through-thickness integration measure:
        N_QUAD_PTS = 4
        dxi2 = throughThicknessMeasure(N_QUAD_PTS,h_th)
        # Potential energy density, including Lagrange multiplier term for
        # incompressibility:
        psi, S = incompressibleKL_mat6(spline,X,x)
        self.S = S
        # Total internal energy:
        Wint = psi*dxi2*spline.dx 

        # Take the Gateaux derivative of Wint in test function direction z_hom.
        z_hom = TestFunction(spline.V)
        z = spline.rationalize(z_hom)
        dWint = derivative(Wint,y_hom,z_hom)

        _,_,_,_,A,_ = surfaceGeometry(spline,X)
        _,_,a2,_,a,_ = surfaceGeometry(spline,x)
        PRESSURE = self.PRESSURE
        stepper = self.stepper
        dWext = -(PRESSURE*stepper.t)*sqrt(det(a)/det(A))*inner(a2, z)*spline.dx

        DENS = Constant(1e-3)
        # DAMP = Constant(1e3)
        DAMP = Constant(3e2) # testing
        print('DENS = {}'.format(DENS.values()))
        print('DAMP = {}'.format(DAMP.values()))
        dWmass = DENS*h_th*inner(yddot,z)*spline.dx
        dWdamp = DAMP*DENS*h_th*inner(ydot,z)*spline.dx
        
        KCP = Constant(self.k_cp)
        dEc = - (KCP*stepper.t)*inner(penetration,z)*sqrt(det(a)/det(A))*spline.dx

        #res = dWint + dWext + dWmass + dWdamp
        res = dWint + dWext + dWmass + dWdamp + dEc
        #res = dWint + dWext
        self.res1 = dWint + dWext
        self.res2 = dWmass
        self.res3 = dWdamp
        self.res4 = dEc

        Dres = derivative(res, y_hom)

        self.res = res
        self.Dres = Dres

        #y0 = root_distention_v2(spline,self.stepper_BC_r)
        #self.y0 = y0

        y0, y0_test, y0_test2 = root_distention(spline,self.stepper_BC_r,self.stepper_BC_z)
        self.y0 = y0
        self.y0_test = y0_test
        self.y0_test2 = y0_test2

    def apply_BC_to_y(self):
        #working version
        format_header("leafletSim::apply_BC_to_y")

        y_hom = self.y_hom
        y0 = self.y0
        y = self.y
        spline = self.spline

        y0_hom = spline.project(y0,rationalize=False,lumpMass=True)
        y0_zeroedBC_hom = spline.project(y0,rationalize=False,lumpMass=True,
                                     applyBCs=True)
        y0_zeroedInterior_hom = Function(spline.V)
        y0_zeroedInterior_hom.assign(y0_hom - y0_zeroedBC_hom)
        yIGA = spline.FEtoIGA(y_hom)
        as_backend_type(yIGA).vec().setValues\
            (spline.zeroDofs, zeros(spline.zeroDofs.getLocalSize()))
        as_backend_type(yIGA).vec().assemble()
        y_zeroedBC_hom = Function(spline.V)
        y_zeroedBC_hom.vector()[:] = (spline.M*yIGA)[:]

        y_hom.assign(y_zeroedBC_hom + y0_zeroedInterior_hom)
        self.y0_zeroedInterior_hom = y0_zeroedInterior_hom
     
    def calc_penetration(self):
        format_header('leafletSim::calc_penetration')

        penetration = self.penetration
        X = self.X
        x = self.x_lower
        spline = self.spline
        '''
        print('Before')
        print(penetration.vector().norm('linf'))
        print(penetration.vector().norm('l2'))
        '''
        _ref_coor = project(X, spline.V)
        _cur_coor = project(x, spline.V)
        num_dofs_control = spline.V_control.dim()
        rx = _ref_coor.sub(0)
        ry = _ref_coor.sub(1)
        rz = _ref_coor.sub(2)
        cx = _cur_coor.sub(0)
        cy = _cur_coor.sub(1)
        cz = _cur_coor.sub(2)

        dof_coordinates = spline.V.tabulate_dof_coordinates()
        dofs_0 = spline.V.sub(0).dofmap().dofs()
        dofs_1 = spline.V.sub(1).dofmap().dofs()
        dofs_2 = spline.V.sub(2).dofmap().dofs()
        dof_coordinates_0 = dof_coordinates[dofs_0,:]
        dof_coordinates_1 = dof_coordinates[dofs_1,:]
        dof_coordinates_2 = dof_coordinates[dofs_2,:]

        cpFuncs = spline.cpFuncs
        nsd = spline.nsd
        bounding_planes_list = self.bounding_planes_list
        bounding_planes0 = bounding_planes_list[0]
        bounding_planes1 = bounding_planes_list[1]
        bounding_planes2 = bounding_planes_list[2]

        min_dist0 = self.min_dist0

        for dof_0,dof_1,dof_2 in zip(dofs_0,dofs_1,dofs_2):
            r_coor = [rx.vector()[dof_0], ry.vector()[dof_1], rz.vector()[dof_2]]
            c_coor = [cx.vector()[dof_0], cy.vector()[dof_1], cz.vector()[dof_2]]
            if np.dot(r_coor-bounding_planes0[0]['point'], bounding_planes0[0]['normal']) > 0 and  np.dot(r_coor-bounding_planes0[1]['point'], bounding_planes0[1]['normal']) > 0:
                leaflet = 0
                bp = bounding_planes0
            elif np.dot(r_coor-bounding_planes1[0]['point'], bounding_planes1[0]['normal']) > 0 and  np.dot(r_coor-bounding_planes1[1]['point'], bounding_planes1[1]['normal']) > 0:
                leaflet = 1
                bp = bounding_planes1
            elif np.dot(r_coor-bounding_planes2[0]['point'], bounding_planes2[0]['normal']) > 0 and  np.dot(r_coor-bounding_planes2[1]['point'], bounding_planes2[1]['normal']) > 0:
                leaflet = 2
                bp = bounding_planes2
            else:
               leaflet = -1
               bp = []
            
            temp_normal = np.array([bp[0]['normal'],bp[1]['normal']])

            temp_vec = np.array([c_coor,c_coor])-np.array([bp[0]['point'],np.array(bp[1]['point'])])
            dist = np.array([temp_normal[0].dot(temp_vec[0]), temp_normal[1].dot(temp_vec[1])])

            #GAP_OFFSET = self.GAP_OFFSET
            #GAP_OFFSET = min_dist0/2.
            GAP_OFFSET = min_dist0/20.
            v = dist.min() - GAP_OFFSET
            if v < 0:
                idx = dist.argmin()
                values = [
                    -float(temp_normal[idx, 0] * v),
                    -float(temp_normal[idx, 1] * v),
                    -float(temp_normal[idx, 2] * v),
                ]
            else:
                values = [0., 0., 0.]
            penetration.vector()[dof_0,dof_1,dof_2] = values
        '''
        print('After')
        print(penetration.vector().norm('linf'))
        print(penetration.vector().norm('linf'))
        print(penetration.vector().norm('l2'))
        '''

    def run_implicit(self,view_mesh_only=False):
        format_header("leafletSim::run_implicit")
        problem = MyShellContactNonlinearProblem(self.contactContext,self.res,self.Dres,self.y_hom)
        self.problem = problem
        extSolver = ExtractedNonlinearSolver(problem,self.solver_implicit)
        self.extSolver = extSolver

        N_STEPS = self.N_STEPS
        OUTPUT_FREQ = int(N_STEPS/30)
        stepper = self.stepper
        stepper_c0 = self.stepper_c0
        stepper_BC_r = self.stepper_BC_r
        stepper_BC_z = self.stepper_BC_z
        y_hom = self.y_hom
        ydot_hom = self.ydot_hom
        yddot_hom = self.yddot_hom
        y_old_hom = self.y_old_hom
        ydot_old_hom = self.ydot_old_hom
        spline = self.spline
 
        # Iterate over load steps.
        start = time.time()
        for i in range(0,N_STEPS+1):
            if(mpirank==0):
                print("------- Step: "+str(i)+" , t = "+str(stepper.tval*30)+" (mmHg) -------")
            self.current_pressure = stepper.tval*30
            p = self.current_pressure
            stepper_c0.t = stepper.tval
            
            #print('transition funciton')
            #print(stepper.tval)
            #print(transition_function(stepper.tval,0,2/3.))
            #print(transition_function(stepper.tval,0,1.))
            stepper_BC_r.t = transition_function(stepper.tval,0,2/3.) # root_distention
            stepper_BC_z.t = transition_function(stepper.tval,0,1.) # root_distention
            self.apply_BC_to_y()
            self.calc_penetration()
            sys.stdout.flush()
            # Execute nonlinear solve.
            try:
                extSolver.solve()
                SOLVER_FAILURE = False 
            except:
                SOLVER_FAILURE = True
            # postprocess
            contact_pairs = problem.contact_pairs
            print('Number of point pairs in contact = {:}.'.format(len(contact_pairs)))

            ydot_hom_proj = project(ydot_hom,spline.V)
            yddot_hom_proj = project(yddot_hom,spline.V)
            print('|velocity| = {}'.format(ydot_hom_proj.vector().norm('l2')))
            print('|acceleration| = {}'.format(yddot_hom_proj.vector().norm('l2')))
            print('Residuals = ')
            res1_norm = assemble(self.res1).norm('l2')
            res2_norm = assemble(self.res2).norm('l2')
            res3_norm = assemble(self.res3).norm('l2')
            res4_norm = assemble(self.res4).norm('l2')
            try:
                print('{:10} {:6.3} ({:2.3f}%)'.format('Wint+Wext',res1_norm,100.*res1_norm/(res1_norm+res2_norm+res3_norm+res4_norm)))
                print('{:10} {:6.3} ({:2.3f}%)'.format('Wmass',res2_norm,100.*res2_norm/(res1_norm+res2_norm+res3_norm+res4_norm)))
                print('{:10} {:6.3} ({:2.3f}%)'.format('Wdamp',res3_norm,100.*res3_norm/(res1_norm+res2_norm+res3_norm+res4_norm)))
                print('{:10} {:6.3} ({:2.3f}%)'.format('Ec',res4_norm,100.*res4_norm/(res1_norm+res2_norm+res3_norm+res4_norm)))
            except:
                pass

            if (i % OUTPUT_FREQ == 0 or SOLVER_FAILURE):
                result_geo = self.postprocess_geometry(VERBOSE=False)
                result_mech = self.postprocess_mechanics(VERBOSE=False)
                combined_result = dict()
                combined_result['geometry'] = result_geo
                combined_result['mechanics'] = result_mech
                if self.nPatch == 3:
                    dL_12 = result_geo['monitor']['distance_leaflets_12']
                    dL_13 = result_geo['monitor']['distance_leaflets_13']
                    dL_23 = result_geo['monitor']['distance_leaflets_23']
                    dP_12 = result_geo['monitor']['distance_midpoint_fe_12']
                    dP_13 = result_geo['monitor']['distance_midpoint_fe_13']
                    dP_23 = result_geo['monitor']['distance_midpoint_fe_23']
                    mipe_ave = result_mech['monitor']['mipe_ave']
                    print('<'*10+'-'*30+'>'*10)
                    print('Distance between leaflets = [{:2.4f},{:2.4f},{:2.4f}]'.format(dL_12,dL_13,dL_23))
                    print('Distance between midpoints on the free edges = [{:2.4f},{:2.4f},{:2.4f}]'.format(dP_12,dP_13,dP_23))

                    '''
                    Currently, exit_flag is not activated
                    '''
                    if min([dL_12,dL_13,dL_23]) < 1e-3:
                        # check if the leaflets are intersecting each other
                        # exit if it is the case
                        print('exit_flag_geometry set to true')
                        print(dL_12)
                        print(dL_13)
                        print(dL_23)
                        exit_flag_geometry = True
                    if mipe_ave > self.MIPE_EXIT_THRESHOLD: 
                        # check if the average MIPE exceeds the threshold
                        exit_flag = True

                self.full_results_geo.append(result_geo)
                self.full_results_mech.append(result_mech)
                self.full_results.append(combined_result)

                self.checkout()
                if any(abs(p-np.array([0,10,20,30])) < 1e-3):
                    p = round(p)
                    mismatch = self.get_geometry_info(p)
                    result = self.postprocess_geometry()
                    scale = self.opt_params['global']['scale']
                    tgi = self.define_target_geometry_info(p)
                    mismatch_area_A = abs(result['leaflet1']['area']*scale*scale - tgi['leaflet_surface_area_A'])
                    mismatch_area_LR = abs(result['leaflet2']['area']*scale*scale - tgi['leaflet_surface_area_LR'])
                    if global_config['INCLUDE_MISMATCH_for_A']: 
                        mismatch = mismatch + mismatch_area_A
                    if global_config['INCLUDE_MISMATCH_for_LR']: 
                        mismatch = mismatch + mismatch_area_LR

                    mismatch = sqrt(mismatch)
                    print('Final mismatch value = %f.'%(mismatch))
                    self.geometry_mismatch[str(round(p))] = mismatch

                    error1 = self.evaluate_L2_error(1)
                    error2 = self.evaluate_L2_error(2)
                    error3 = self.evaluate_L2_error(3)
                    error = error1 + error2 + error3
                    print('L2 error at {} mmHg = {}'.format(round(p),error))
                    self.L2_error[str(round(p))] = error
 
                # print('Done')
                # exit()

            end = time.time()
            time_elapsed = end-start
            print("Time elapsed = %.1fh"%(time_elapsed/3600))

            # Move to the next time step:
            ydot_old_hom.assign(ydot_hom)
            y_old_hom.assign(y_hom)
         
            # Advance to next load step.
            stepper.advance()
            sys.stdout.flush()
            if SOLVER_FAILURE:
                break
         
        if SOLVER_FAILURE:
            print('Simulation fails!',flush=True)
        else:
            print('Simulation finishes!',flush=True)
        
        return not SOLVER_FAILURE

    def main(self):
        format_header("leafletSim::main")
        
        self.init_mesh(prefix='./mesh/leaflet_tmp_')

        #self.init_mesh(prefix='./mesh/leaflet_right_')
        self.setup(1./self.N_STEPS) 
        '''
        result = self.postprocess_geometry()
        mismatch_area = result['monitor']['mismatch_in_area']
        mismatch = sqrt(mismatch**2+sqrt(mismatch_area))
        print('VALVE_ID = %d'%(self.ID))
        print('Final mismatch value = %f.'%(mismatch))
        '''
        self.init_contact()
        SOLVER_SUCCESS = self.run_implicit(view_mesh_only=False)

        self.logFile.write("PID {} exited at pressure {} mmHg\n".format(os.getpid(),round(self.current_pressure)))
        self.logFile.flush()

        if SOLVER_SUCCESS:
            val = self.evaluate_cost()
            #val = self.L2_error['10']+self.L2_error['20']+self.L2_error['30']
        else:
            val = 1e6
       
        self.logFile.write('cost value = {:2.4f}\n'.format(val))
        self.cleanup()

        return val

    def load_mPV_GM(self):
        self.mPV_GM_0mmHg_rep = dict()
        self.mPV_GM_10mmHg_rep = dict()
        self.mPV_GM_20mmHg_rep = dict()
        self.mPV_GM_30mmHg_rep = dict()
        self.mPV_GM_0mmHg_rep['1'] = readLegacy_igakit(os.path.join('../mPV_GM','0_rep','mesh','leaflet_tmp_1.dat'))
        self.mPV_GM_0mmHg_rep['2'] = readLegacy_igakit(os.path.join('../mPV_GM','0_rep','mesh','leaflet_tmp_2.dat'))
        self.mPV_GM_0mmHg_rep['3'] = readLegacy_igakit(os.path.join('../mPV_GM','0_rep','mesh','leaflet_tmp_3.dat'))
        self.mPV_GM_10mmHg_rep['1'] = readLegacy_igakit(os.path.join('../mPV_GM','10_rep','mesh','leaflet_tmp_1.dat'))
        self.mPV_GM_10mmHg_rep['2'] = readLegacy_igakit(os.path.join('../mPV_GM','10_rep','mesh','leaflet_tmp_2.dat'))
        self.mPV_GM_10mmHg_rep['3'] = readLegacy_igakit(os.path.join('../mPV_GM','10_rep','mesh','leaflet_tmp_3.dat'))
        self.mPV_GM_20mmHg_rep['1'] = readLegacy_igakit(os.path.join('../mPV_GM','20_rep','mesh','leaflet_tmp_1.dat'))
        self.mPV_GM_20mmHg_rep['2'] = readLegacy_igakit(os.path.join('../mPV_GM','20_rep','mesh','leaflet_tmp_2.dat'))
        self.mPV_GM_20mmHg_rep['3'] = readLegacy_igakit(os.path.join('../mPV_GM','20_rep','mesh','leaflet_tmp_3.dat'))
        self.mPV_GM_30mmHg_rep['1'] = readLegacy_igakit(os.path.join('../mPV_GM','30_rep','mesh','leaflet_tmp_1.dat'))
        self.mPV_GM_30mmHg_rep['2'] = readLegacy_igakit(os.path.join('../mPV_GM','30_rep','mesh','leaflet_tmp_2.dat'))
        self.mPV_GM_30mmHg_rep['3'] = readLegacy_igakit(os.path.join('../mPV_GM','30_rep','mesh','leaflet_tmp_3.dat'))

        N1 = 65
        N2 = 65
        dx, dy = 1./N1, 1./N2
        x = np.random.rand((N1+1)*(N2+1)).reshape((N1+1,N2+1,1))
        y = np.random.rand((N1+1)*(N2+1)).reshape((N1+1,N2+1,1))
        z = np.random.rand((N1+1)*(N2+1)).reshape((N1+1,N2+1,1))
        scale = self.opt_params['global']['scale']

        for p in [0,10,20,30]:
            for leaflet in [1,2,3]:
                if p == 0:
                    mPV_GM = self.mPV_GM_0mmHg_rep[str(leaflet)]
                if p == 10:
                    mPV_GM = self.mPV_GM_10mmHg_rep[str(leaflet)]
                if p == 20:
                    mPV_GM = self.mPV_GM_20mmHg_rep[str(leaflet)]
                if p == 30:
                    mPV_GM = self.mPV_GM_30mmHg_rep[str(leaflet)]

                for j in range(N2+1):
                    for i in range(N1+1):
                        x1 = dx*i
                        x2 = dy*j
                        pos_GM = eval_nurbs(mPV_GM,x2,x1)*scale
                        x[i,j,0] = pos_GM[0]
                        y[i,j,0] = pos_GM[1]
                        z[i,j,0] = pos_GM[2]

                if global_config['CHECKOUT'] == True:
                    gridToVTK(
                            os.path.join(self.result_dir,'results',self.result_dir_str,'mPV_GM_'+str(p)+'mmHg_'+str(leaflet)),
                            x,
                            y,
                            z,
                            )
        print('Done saving mPV GM')

    def evaluate_L2_error(self,leaflet):
        '''
        leaflet = 1,2,3
        '''
        print('evaluate_L2_error')
        print('pressure = {}'.format(round(self.current_pressure)))
        print('leaflet = {}'.format(leaflet))

        scale = self.opt_params['global']['scale']
   
        #print('------------')
        mPV_GM = None
        if np.isclose(self.current_pressure,0):
            mPV_GM = self.mPV_GM_0mmHg_rep[str(leaflet)]
        if np.isclose(self.current_pressure,10):
            mPV_GM = self.mPV_GM_10mmHg_rep[str(leaflet)]
        if np.isclose(self.current_pressure,20):
            mPV_GM = self.mPV_GM_20mmHg_rep[str(leaflet)]
        if np.isclose(self.current_pressure,30):
            mPV_GM = self.mPV_GM_30mmHg_rep[str(leaflet)]
        '''
        print(self.eval_pos(1,0.,0.1)*scale)    # c, r
        print(self.eval_pos(1,1.,0.1)*scale)
        mPV_GM = self.mPV_GM_0mmHg_rep[str(1)]
        print(eval_nurbs(mPV_GM,0.1,0.)*scale)  # r, c
        print(eval_nurbs(mPV_GM,0.1,1.)*scale)
        '''

        N1, N2 = self.N1, self.N2
        dx, dy = 1./N1, 1./N2
        x = np.random.rand((N1+1)*(N2+1)).reshape((N1+1,N2+1,1))
        y = np.random.rand((N1+1)*(N2+1)).reshape((N1+1,N2+1,1))
        z = np.random.rand((N1+1)*(N2+1)).reshape((N1+1,N2+1,1))
        xx = np.random.rand((N1+1)*(N2+1)).reshape((N1+1,N2+1,1))
        yy = np.random.rand((N1+1)*(N2+1)).reshape((N1+1,N2+1,1))
        zz = np.random.rand((N1+1)*(N2+1)).reshape((N1+1,N2+1,1))
        L2_error = np.random.rand((N1+1)*(N2+1)).reshape((N1+1,N2+1,1))
        rL2_error = np.random.rand((N1+1)*(N2+1)).reshape((N1+1,N2+1,1))

        L2_error_tmp = np.Inf
        L2_error_min = np.Inf
        pos_GM_tmp = []
        if mPV_GM:
            for j in range(N2+1):
                for i in range(N1+1):
                    a1 = dx*i
                    b1 = dy*j
                    '''
                    Note, the order of coordinates is different!
                    '''
                    pos_sim = self.eval_pos(leaflet,a1,b1)*scale
                    x[i,j,0] = pos_sim[0]
                    y[i,j,0] = pos_sim[1]
                    z[i,j,0] = pos_sim[2]
                    '''
                    approach I
                    Corresponding points on the parametric domain
                    '''
                    #pos_GM = eval_nurbs(mPV_GM,b1,a1)*scale
                    #L2_error[i,j,0] = np.linalg.norm(np.array(pos_sim)-np.array(pos_GM))
                    #rL2_error[i,j,0] = L2_error[i,j,0]/2/scale*100
                    #xx[i,j,0] = pos_GM[0]
                    #yy[i,j,0] = pos_GM[1]
                    #zz[i,j,0] = pos_GM[2]
                    '''
                    approach II
                    Minimal distance on the physical surfaces
                    '''
                    L2_error_min = np.Inf
                    for jj in range(65+1):
                        for ii in range(65+1):
                            aa1 = 1./65*ii
                            bb1 = 1./65*jj
                            pos_GM_tmp = eval_nurbs(mPV_GM,bb1,aa1)*scale
                            L2_error_tmp = np.linalg.norm(np.array(pos_sim)-np.array(pos_GM_tmp))
                            if L2_error_tmp < L2_error_min:
                                L2_error_min = L2_error_tmp
                                aa1_min = aa1
                                bb1_min = bb1
                                pos_GM = pos_GM_tmp
                    L2_error[i,j,0] = L2_error_min
                    rL2_error[i,j,0] = L2_error[i,j,0]/2/scale*100
                    xx[i,j,0] = pos_GM[0]
                    yy[i,j,0] = pos_GM[1]
                    zz[i,j,0] = pos_GM[2]
                    #print('xxxxxxxxx')
                    #print('Sim: ({},{})'.format(a1,b1))
                    #print(np.array(pos_sim))
                    #print('mPV_GM: ({},{})'.format(aa1_min,bb1_min))
                    #print(np.array(pos_GM))
            
            if global_config['CHECKOUT'] == True:
                gridToVTK(
                        os.path.join(self.result_dir,'results',self.result_dir_str,'L2_error_'+str(int(round(self.current_pressure)))+'mmHg_'+str(leaflet)),
                        x,
                        y,
                        z,
                        pointData={'L2_error':L2_error,'rL2_error':rL2_error},
                        )
                gridToVTK(
                        os.path.join(self.result_dir,'results',self.result_dir_str,'mPV_GM_downsampled_'+str(int(round(self.current_pressure)))+'mmHg_'+str(leaflet)),
                        xx,
                        yy,
                        zz,
                        )
            
            print('Done saving L2 error')
            return np.linalg.norm(L2_error)/sqrt(N1+1)/sqrt(N2+1)

    def eval_pos(self,leaflet,x1,x2):
        #print('--- eval_pos ---')
        ''' 
        leaflet = 1 (anterior), 2 (left), 3 (right)
        '''
        spline = self.spline
        mesh = spline.mesh
        obj = ProbeLeaflet(spline,self.y,leaflet,False)
        F0_func_tri = obj.F0_func_tri 
        F1_func_tri = obj.F1_func_tri 
        F2_func_tri = obj.F2_func_tri 
        disp0_func_tri = obj.disp0_func_tri 
        disp1_func_tri = obj.disp1_func_tri 
        disp2_func_tri = obj.disp2_func_tri 

        xmin = obj.xmin
       
        pt = Point(xmin+x1,x2)
        F_pt = [F0_func_tri(pt),F1_func_tri(pt),F2_func_tri(pt)]
        #print("point = [{},{},{}]".format(F_pt[0],F_pt[1],F_pt[2]))
        #print("disp = [{},{},{}]".format(disp_pt[0],disp_pt[1],disp_pt[2]))

        return np.array(F_pt)

    def main_mp_version(self,controlMesh,splineGenerator):
        format_header("leafletSim::main_mp_version")
        
        self.controlMesh = controlMesh
        self.splineGenerator = splineGenerator
        self.init_mesh_mp_version()

        #self.init_mesh(prefix='./mesh/leaflet_right_')
        self.setup(1./self.N_STEPS) 
        '''
        result = self.postprocess_geometry()
        mismatch_area = result['monitor']['mismatch_in_area']
        mismatch = sqrt(mismatch**2+sqrt(mismatch_area))
        print('VALVE_ID = %d'%(self.ID))
        print('Final mismatch value = %f.'%(mismatch))
        '''
        self.init_contact()
        self.load_mPV_GM()
        SOLVER_SUCCESS = self.run_implicit(view_mesh_only=False)

        self.logFile.write("PID {} finished at pressure {} mmHg\n".format(os.getpid(),round(self.current_pressure)))
        self.logFile.flush()

        if SOLVER_SUCCESS:
            val = self.evaluate_cost()
            #val = self.L2_error['10']+self.L2_error['20']+self.L2_error['30']
        else:
            val = 1e6
       
        self.logFile.write('cost value = {:2.4f}\n'.format(val))
        self.cleanup()
        print('Done')

        return val


    def cleanup(self):
        self.logFile.close()

    def evaluate_cost(self):
        format_header('leafletSim::evaluate_cost')

        OUTPUT = global_config['OUTPUT_CS_PROFILES'] 
        '''
        Note: cross sections in mPV_GM/ are in physical unit (um) and coordinates (i.e., with BC applied)
        '''
        def add_sub_cost_function_cross_section(leaflet,add_R2=True,add_C2=True):
            #format_header('add_sub_cost_function (p = {})'.format(round(pressure)))
            value = 0
            value_r = 0
            value_c = 0
            flip_c = True 
            #################################
            # Radial direction
            #################################
            '''
            measurement
            '''
            if leaflet == 'a':
                base_filename = 'r2_mPV_GM_'+str(int(pressure))+'mmHg_A_planar'
            if leaflet in ['l','r']:
                base_filename = 'r2_mPV_GM_'+str(int(pressure))+'mmHg_LR_planar'

            with open('./mPV_GM/'+base_filename+".csv",newline='') as csvfile:
                reader = csv.reader(csvfile)
                X = list()
                Y = list()
                Z = list()
                for row in reader:
                    X.append(float(row[0]))
                    Y.append(float(row[1]))
                    Z.append(0.)
            Xr_redist,Yr_redist,Zr_redist = equal_spacing_curve(X,Y,Z,40,'')
            #normalizationX_me = max(Xr_redist)-min(Xr_redist)
            #normalizationH_me = normalizationX_me
            normalizationX_me = 1
            normalizationH_me = 1
            Xr_processed = [(p-Xr_redist[0])/normalizationX_me for p in Xr_redist]
            Yr_processed = [(p-Yr_redist[0])/normalizationH_me for p in Yr_redist]
            if OUTPUT:
                with open(os.path.join('results',sys.argv[1],'cs','measurement-R2-processed-'+leaflet+'-'+str(int(pressure))+'mmHg.csv'),mode='w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([[x,y] for (x,y) in zip(Xr_processed,Yr_processed)])
            '''
            model
            '''
            curve = R2_curve
            model_X_redist,model_Y_redist,model_Z_redist = equal_spacing_curve(curve[:,0],curve[:,1],curve[:,2],40,'')
            #normalizationX_mo = max(curve[:,0])-min(curve[:,0])
            #normalizationH_mo = normalizationX_mo
            normalizationX_mo = 1
            normalizationH_mo = 1
            model_X_processed = [(p-model_X_redist[0])/normalizationX_mo*scale for p in model_X_redist]
            model_Y_processed = [(p-model_Y_redist[0])/normalizationH_mo*scale for p in model_Y_redist]
            N = len(model_X_processed)
            weights = np.ones((N,))
            for i in range(N):
                if i > N/2:
                    weights[i] = 3 # biased toward the FE end
            if OUTPUT:
                with open(os.path.join('results',sys.argv[1],'cs','model-R2-processed-'+leaflet+'-'+str(int(pressure))+'mmHg.csv'),mode='w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([[x,y] for (x,y) in zip(model_X_processed,model_Y_processed)])
            if add_R2:
                value_r = np.sqrt(np.mean([w*pow(x1-x2,2) for (x1,x2,w) in zip(Xr_processed,model_X_processed,weights)])+np.mean([w*pow(y1-y2,2) for (y1,y2,w) in zip(Yr_processed,model_Y_processed,weights)]))
                print('Mismatch in cross section ({}-r) = {:2.4}'.format(leaflet, value_r))
        
            print('xxxxxxxxxxxxxxxxx')
            print(pandas.DataFrame(zip(Xr_processed,Yr_processed,model_X_processed,model_Y_processed,weights)))

            #################################
            # Circumferential direction
            #################################
            '''
            measurement
            '''
            if leaflet == 'a':
                base_filename = 'c2_mPV_GM_'+str(int(pressure))+'mmHg_A_planar'
            if leaflet in ['l','r']:
                base_filename = 'c2_mPV_GM_'+str(int(pressure))+'mmHg_LR_planar'

            with open('./mPV_GM/'+base_filename+".csv",newline='') as csvfile:
                reader = csv.reader(csvfile)
                X = list()
                Y = list()
                Z = list()
                for row in reader:
                    X.append(float(row[0]))
                    Y.append(float(row[1]))
                    Z.append(0.)
            Xc_redist,Yc_redist,Zc_redist = equal_spacing_curve(X,Y,Z,40,'')
            # center the measurement C2 profile
            #Xc_redist_mid = (Xc_redist[0]+Xc_redist[-1])/2.
            Xc_processed = [(p-np.mean(np.array(Xc_redist)))/normalizationX_me for p in Xc_redist]
            #Yc_processed = [(p-np.min(np.array(Yc_redist)))/normalizationH_me for p in Yc_redist]
            Yc_processed = [p for p in Yc_redist]
            if OUTPUT:
                with open(os.path.join('results',sys.argv[1],'cs','measurement-C2-processed-'+leaflet+'-'+str(int(pressure))+'mmHg.csv'),mode='w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([[x,y] for (x,y) in zip(Xc_processed,Yc_processed)])
            '''
            model
            '''
            curve = C2_curve
            model_X_redist,model_Y_redist,model_Z_redist = equal_spacing_curve(curve[:,0],curve[:,1],curve[:,2],40,'')
            #print(max(model_X_redist)-min(model_X_redist))
            model_X_processed = [(p-np.mean(np.array(model_X_redist)))/normalizationX_mo*scale for p in model_X_redist]
            #model_Y_processed = [(p-np.min(np.array(model_Y_redist)))/normalizationH_mo*scale for p in model_Y_redist]
            model_Y_processed = [p*scale for p in model_Y_redist]
            if flip_c:
                model_X_processed = [-p for p in model_X_processed]
                Xc_processed = [-p for p in Xc_processed]
            if OUTPUT:
                with open(os.path.join('results',sys.argv[1],'cs','model-C2-processed-'+leaflet+'-'+str(int(pressure))+'mmHg.csv'),mode='w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([[x,y] for (x,y) in zip(model_X_processed,model_Y_processed)])
            if add_C2:
                value_c = np.sqrt(np.mean([pow(x1-x2,2) for (x1,x2) in zip(Xc_processed,model_X_processed)])+np.mean([pow(y1-y2,2) for (y1,y2) in zip(Yc_processed,model_Y_processed)]))
                print('Mismatch in cross section ({}-c) = {:2.4}'.format(leaflet, value_c))
            value = value_r + value_c
            #value = value/scale
            print('xxxxxxxxxxxxxxxxx')
            print(pandas.DataFrame(zip(Xc_processed,Yc_processed,model_X_processed,model_Y_processed)))

            return value

        val = 0
        val_cs = 0
        val_gQOI = 0
        scale = self.opt_params['global']['scale']
        for n in range(len(self.full_results_geo)):
            results_geo = self.full_results_geo[n]
            pressure = round(results_geo['pressure'])
            if any(abs(pressure - np.array([10,20,30]))<1e-3):
                print('pressure = {}'.format(pressure))                
                print('geometry mismatch = {}'.format(self.geometry_mismatch[str(pressure)]))
                val_gQOI = val_gQOI + 0.5*self.geometry_mismatch[str(pressure)]
                try:
                    if global_config['INCLUDE_MISMATCH_for_A']:
                        R2_curve = get_R2_curve_processed('a',results_geo['leaflet1']['R2_curve'])
                        C2_curve = get_C2_curve_processed('a',results_geo['leaflet1']['C2_curve'])
                        val_cs = val_cs + add_sub_cost_function_cross_section('a',True,True)
                        
                    if global_config['INCLUDE_MISMATCH_for_LR']:
                        R2_curve = get_R2_curve_processed('l',results_geo['leaflet2']['R2_curve'])
                        C2_curve = get_C2_curve_processed('l',results_geo['leaflet2']['C2_curve'])
                        val_cs = val_cs + add_sub_cost_function_cross_section('l',True,True)
                except:
                    print('Invalid code: ADD_SUB_COST_FUNCTION')
                    val_cs = 1e6

        val = val_cs+val_gQOI

        print('The value of the cost function = {}.'.format(val))
        print('The value of the cost function (cs) = {}.'.format(val_cs))
        print('The value of the cost function (gQOI) = {}.'.format(val_gQOI))
        return val
 
    def postprocess_mechanics(self,VERBOSE=False):
        format_header("leafletSim::postprocess_mechanics")
        # Generate a metric tensor at through-thickness coordinate xi2, based on the
        # midsurface metric and curvature.
        # TODO: add stress as outputs
        spline = self.spline
        V = spline.V_control
        mesh = spline.mesh
        p = self.current_pressure

        def metric(a, b, xi2):
            return a - 2.0*xi2*b

        # Raise indices on curvilinear basis
        def tensorToCartesian(T, a, a0, a1):
            ac = inv(a)
            a0c = ac[0, 0]*a0 + ac[0, 1]*a1
            a1c = ac[1, 0]*a0 + ac[1, 1]*a1

            e0, e1 = localCartesianBasis(a0, a1)

            ea = as_matrix(((inner(e0, a0c), inner(e0, a1c)),
                            (inner(e1, a0c), inner(e1, a1c))))
            ae = ea.T
            return ea*T*ae

        def localCartesianBasis(a0, a1):                
            # Perform Gram--Schmidt orthonormalization to get e0 and e1.
            e0 = unit(a0)
            e1 = unit(a1- e0*inner(a1, e0))
            return e0, e1

        # Assuming that g1, g2 and g3 are independent
        def change2basis(v,g1,g2,g3):
            M = as_matrix(((inner(g1,g1),inner(g2,g1),inner(g3,g1)),
                           (inner(g1,g2),inner(g2,g2),inner(g3,g2)),
                           (inner(g1,g3),inner(g2,g3),inner(g3,g3))))
            Mc = inv(M)
            b = as_vector([inner(v,g1),inner(v,g2),inner(v,g3)])
            return Mc*b
        
        A0 = self.A0
        A1 = self.A1
        A2 = self.A2
        deriv_A2 = self.deriv_A2
        A = self.A
        B = self.B
        a0 = self.a0
        a1 = self.a1
        a2 = self.a2
        deriv_a2 = self.deriv_a2
        a = self.a
        b = self.b
        A_det_proj = project(det(A), V)
        a_det_proj = project(det(a), V)
        self.A_det_proj = A_det_proj
        self.a_det_proj = a_det_proj

        xi2 = Constant(0.0)
        # metrics
        G = metric(A, B, xi2)
        g = metric(a, b, xi2)
        E_flat = 0.5*(g - G)
        E_2D = tensorToCartesian(E_flat, G, A0, A1)
        # These field do not provide much physical meaning, consider to remove them from the list of outputs
        E00_proj = project(E_2D[0, 0], V)
        E01_proj = project(E_2D[0, 1], V)
        E10_proj = project(E_2D[1, 0], V)
        E11_proj = project(E_2D[1, 1], V)
        C_2D = 2.0*E_2D+Identity(2)
        C22 = 1.0/det(C_2D)
        E22 = 0.5*(C22-1.0)
        E22_proj  = project(E22, V)
        E = as_tensor([[E_2D[0,0], E_2D[0,1], 0.0],
                        [E_2D[1,0], E_2D[1,1], 0.0],
                        [0.0, 0.0, E22]])
        C = 2.0*E + Identity(3)
        J = sqrt(det(C))
        #e0 = unit(A0)
        #e1 = unit(A1-e0*inner(A1,e0))
        e0,e1 = localCartesianBasis(A0, A1)                
        e2 = unit(cross(e0,e1))
        A0_newbasis = change2basis(unit(A0),e0,e1,e2)     
        A1_newbasis = change2basis(unit(A1),e0,e1,e2) 
        #A0_newbasis = unit(A0)
        #A1_newbasis = unit(A1)
        J_proj = project(J, V)
        Ett_proj = project(inner(A0_newbasis,E*A0_newbasis), V)
        Err_proj = project(inner(A1_newbasis,E*A1_newbasis), V)
        Etr_proj = project(inner(A0_newbasis,E*A1_newbasis), V)
        Ert_proj = project(inner(A1_newbasis,E*A0_newbasis), V)        
        E2D_trace_proj = project(tr(E_2D), V)
        E2D_det_proj = project(det(E_2D), V)
        mipe_proj = project(0.5*tr(E_2D)+sqrt(tr(E_2D)**2/4.-det(E_2D)), V)

        S = self.S        
        Stt_proj = project(inner(A0_newbasis,S*A0_newbasis), V)
        Srr_proj = project(inner(A1_newbasis,S*A1_newbasis), V)
        Str_proj = project(inner(A0_newbasis,S*A1_newbasis), V)
        Srt_proj = project(inner(A1_newbasis,S*A0_newbasis), V)        
        self.Stt_proj = Stt_proj
        self.Srr_proj = Srr_proj
        self.Str_proj = Str_proj
        self.Srt_proj = Srt_proj

        Lt_proj = project(sqrt(1+2*inner(A0_newbasis,E*A0_newbasis)), V)
        Lr_proj = project(sqrt(1+2*inner(A1_newbasis,E*A1_newbasis)), V)
        Lw_proj = project(sqrt(1+2*E22), V)
        Art_0_proj = project(acos(abs(inner(unit(A0),unit(A1)))), V)
        Art_proj = project(acos(abs(inner(unit(a0),unit(a1)))), V)

        # checking block
        try:
            Ett_proj_approach2 = project(E_flat[0, 0]/inner(A0,A0), V)
            Err_proj_approach2 = project(E_flat[1, 1]/inner(A1,A1), V)
            Ert_proj_approach2 = project(E_flat[1, 0]/sqrt(inner(A0,A0)*inner(A1,A1)), V)
            Etr_proj_approach2 = project(E_flat[0, 1]/sqrt(inner(A0,A0)*inner(A1,A1)), V)
            assert(norm(project(Ett_proj-Ett_proj_approach2,V))<1e-5)
            assert(norm(project(Err_proj-Err_proj_approach2,V))<1e-5)
            assert(norm(project(Etr_proj-Etr_proj_approach2,V))<1e-5)
            assert(norm(project(Ert_proj-Ert_proj_approach2,V))<1e-5)
        except:
            print('The two approaches shall always match!')
            print('diff = {}'.format(norm(project(Ett_proj-Ett_proj_approach2, V))))
            print('diff = {}'.format(norm(project(Err_proj-Err_proj_approach2, V))))
            print('diff = {}'.format(norm(project(Ert_proj-Ert_proj_approach2, V))))
            print('diff = {}'.format(norm(project(Etr_proj-Etr_proj_approach2, V))))
            exit()

        self.J_proj = J_proj
        self.Eww_proj = E22_proj
        self.E2D_trace_proj = E2D_trace_proj
        self.E2D_det_proj = E2D_det_proj
        self.Ett_proj = Ett_proj
        self.Etr_proj = Etr_proj
        self.Ert_proj = Ert_proj
        self.Err_proj = Err_proj
        self.mipe_proj = mipe_proj
        self.Lt_proj = Lt_proj
        self.Lr_proj = Lr_proj
        self.Lw_proj = Lw_proj
        self.Art_0_proj = Art_0_proj
        self.Art_proj = Art_proj
        '''
        energySurfaceDensity = self.energySurfaceDensity
        energySurfaceDensity1 = self.energySurfaceDensity1
        energySurfaceDensity2 = self.energySurfaceDensity2
        energySurfaceDensity3 = self.energySurfaceDensity3
        energySurfaceDensity_proj = project(energySurfaceDensity, V)
        energySurfaceDensity1_proj = project(energySurfaceDensity1, V)
        energySurfaceDensity2_proj = project(energySurfaceDensity2, V)
        energySurfaceDensity3_proj = project(energySurfaceDensity3, V)
        
        self.energySurfaceDensity_proj = energySurfaceDensity_proj
        self.energySurfaceDensity1_proj = energySurfaceDensity1_proj
        self.energySurfaceDensity2_proj = energySurfaceDensity2_proj
        self.energySurfaceDensity3_proj = energySurfaceDensity3_proj
        '''
        exit_flag = False
        if(mpirank == 0):
            mipe_max = mipe_proj.vector().norm('linf')
            domains = MeshFunction("size_t",mesh, mesh.topology().dim(),0)
            domains.set_all(0)
            dx = Measure("dx",subdomain_data=domains)
            mipe_ave = assemble(mipe_proj*dx(0,domain=mesh))/assemble(Constant(1)*dx(0,domain=mesh)) 
            print("max MIPE = %2.3f"%(mipe_max))
            print("ave MIPE = %2.3f"%(mipe_ave))
            self.mipe_max = mipe_max
            self.mipe_ave = mipe_ave

        result = dict()
        result['pressure'] = p 
        result['monitor'] = dict()
        result['monitor']['mipe_max'] = mipe_max
        result['monitor']['mipe_ave'] = mipe_ave
        return result

    def postprocess_geometry(self,VERBOSE=False):
        format_header("leafletSim::postprocess_geometry")
        if self.nPatch == 1:
            return []
        spline = self.spline
        disp = self.y # code distention
        N1 = self.N1
        N2 = self.N2
        mesh = spline.mesh
        mesh_tri = mesh_quad2tri(mesh)
        p = self.current_pressure
        scale = self.opt_params['global']['scale']
        '''
        _ref_coor = project(self.X, spline.V)
        rz = Function(spline.V_control)
        cz = Function(spline.V_control)
        assigner = FunctionAssigner(spline.V_control,spline.V.sub(2))
        assigner.assign(rz, _ref_coor.sub(2))
        print(self.current_pressure)
        print(max(rz.vector()))
        print(min(rz.vector()))
        _cur_coor = project(self.x, spline.V)
        #cz = _cur_coor.sub(2)
        assigner.assign(cz, _cur_coor.sub(2))
        print(max(cz.vector()))
        print(min(cz.vector()))
        '''
        class ProbeValve():
            def __init__(self,VERBOSE=False):
                self.probe_leaflet1 = ProbeLeaflet(1,VERBOSE)
                self.probe_leaflet2 = ProbeLeaflet(2,VERBOSE)
                self.probe_leaflet3 = ProbeLeaflet(3,VERBOSE)
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
            def __init__(self,leaflet_id,VERBOSE=False):
                '''
                leaflet = 1 (anterior), 2 (left), 3 (right)
                '''
                xmin = 2*leaflet_id-2
                xmax = 2*leaflet_id-1
                self.xmin = xmin
                self.xmax = xmax
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
        
        probe_valve = ProbeValve(VERBOSE)
        result = probe_valve.run()
        return result

    def checkout(self, p=[]):
        if not global_config['CHECKOUT']:
            return

        format_header("leafletSim::checkout")
        '''
        Save the result to files while requested
        '''
        def unit(v):
            return v/sqrt(inner(v, v))

        if not p:
            p = self.current_pressure
            print('Checking out the result at pressure {:3.3}'.format(p*1.0))
        else:
            print('Checking out the result with suffix {:3.3}'.format(p*1.0))

        spline = self.spline
        y = self.y
        y_hom = self.y_hom

        (d0, d1, d2) = self.y_hom.split()
        d0.rename("d0", "d0")
        d1.rename("d1", "d1")
        d2.rename("d2", "d2")
        self.d0File << (d0, p)
        self.d1File << (d1, p)
        self.d2File << (d2, p)
  
        F0 = spline.cpFuncs[0]
        F1 = spline.cpFuncs[1]
        F2 = spline.cpFuncs[2]
        F3 = spline.cpFuncs[3]
        F0.rename("F0", "F0")
        F1.rename("F1", "F1")
        F2.rename("F2", "F2")
        F3.rename("F3", "F3")
        self.F0File << (F0, p)
        self.F1File << (F1, p)
        self.F2File << (F2, p)
        self.F3File << (F3, p)

        mesh = spline.mesh

        _ref_coor = project(self.X, spline.V)
        _cur_coor = project(self.x, spline.V)
        _cur_coor_lower = project(self.x_lower, spline.V)
        _cur_coor_upper = project(self.x_upper, spline.V)
        rx = _ref_coor.sub(0)
        ry = _ref_coor.sub(1)
        rz = _ref_coor.sub(2)
        cx = _cur_coor.sub(0)
        cy = _cur_coor.sub(1)
        cz = _cur_coor.sub(2)
        cx_lower = _cur_coor_lower.sub(0)
        cy_lower = _cur_coor_lower.sub(1)
        cz_lower = _cur_coor_lower.sub(2)
        cx_upper = _cur_coor_upper.sub(0)
        cy_upper = _cur_coor_upper.sub(1)
        cz_upper = _cur_coor_upper.sub(2)
 
        rx.rename('rx','rx')
        ry.rename('ry','ry')
        rz.rename('rz','rz')
        self.r0File << (rx, p)
        self.r1File << (ry, p)
        self.r2File << (rz, p)

        cx.rename('cx','cx')
        cy.rename('cy','cy')
        cz.rename('cz','cz')
        self.c0File << (cx, p)
        self.c1File << (cy, p)
        self.c2File << (cz, p)
      
        cx_lower.rename('cx_lower','cx_lower')
        cy_lower.rename('cy_lower','cy_lower')
        cz_lower.rename('cz_lower','cz_lower')
        self.c0File_lower << (cx_lower, p)
        self.c1File_lower << (cy_lower, p)
        self.c2File_lower << (cz_lower, p)
 
        cx_upper.rename('cx_upper','cx_upper')
        cy_upper.rename('cy_upper','cy_upper')
        cz_upper.rename('cz_upper','cz_upper')
        self.c0File_upper << (cx_upper, p)
        self.c1File_upper << (cy_upper, p)
        self.c2File_upper << (cz_upper, p)
 
        self.penetration.rename('penetration', 'penetration')
        self.penetration_File << (self.penetration, p)
        # Note: self.postprocess_mechanics() needs to run before checking the solution out
        E2D_trace = self.E2D_trace_proj
        E2D_det = self.E2D_det_proj
        J = self.J_proj
        Eww = self.Eww_proj
        Ett = self.Ett_proj
        Etr = self.Etr_proj
        Ert = self.Ert_proj
        Err = self.Err_proj
        mipe = self.mipe_proj
        Lw = self.Lw_proj
        Lt = self.Lt_proj
        Lr = self.Lr_proj
        Art_0 = self.Art_0_proj
        Art = self.Art_proj
        A_det = self.A_det_proj
        a_det = self.a_det_proj

        Stt = self.Stt_proj
        Srr = self.Srr_proj
        Srt = self.Srt_proj
        Str = self.Str_proj

        E2D_trace.rename('E2D_trace', 'E2D_trace')
        E2D_det.rename('E2D_det', 'E2D_det')
        self.E2D_tr_File << (E2D_trace, p)
        self.E2D_det_File << (E2D_det, p)

        J.rename('J', 'J')
        Eww.rename('E_ww', 'E_ww')
        Ett.rename('E_tt', 'E_tt')
        Etr.rename('E_tr', 'E_tr')
        Ert.rename('E_rt', 'E_rt')
        Err.rename('E_rr', 'E_rr')
        mipe.rename('MIPE', 'MIPE')
        Lw.rename('lambda_w','lambda_w')
        Lt.rename('lambda_t','lambda_t')
        Lr.rename('lambda_r','lambda_r')
        Art_0.rename('angle0_rt','angle0_rt')
        Art.rename('angle_rt','angle_rt')
        A_det.rename('det_A','det_A')
        a_det.rename('det_a','det_a')
        Stt.rename('S_tt', 'S_tt')
        Srr.rename('S_rr', 'S_rr')
        Srt.rename('S_rt', 'S_rt')
        Str.rename('S_tr', 'S_tr')
        self.JFile << (J, p)
        self.EttFile << (Ett, p)
        self.EtrFile << (Etr, p)
        self.ErtFile << (Ert, p)
        self.ErrFile << (Err, p)
        self.EwwFile << (Eww, p)
        self.mipeFile << (mipe, p) 
        self.LwFile << (Lw, p)
        self.LtFile << (Lt, p)
        self.LrFile << (Lr, p)
        self.Art_0_File << (Art_0, p)
        self.Art_File << (Art, p)
        self.A_det_File << (A_det, p)
        self.a_det_File << (a_det, p)
        self.SttFile << (Stt, p)
        self.SrrFile << (Srr, p)
        self.SrtFile << (Srt, p)
        self.StrFile << (Str, p)

        e0 = project(unit(self.a0),spline.V)
        e1 = project(unit(self.a1),spline.V)
        e2 = project(unit(self.a2),spline.V)
        e0.rename('e0','circumferential')
        e1.rename('e1','radial')
        e2.rename('e2','normal')
        self.a0File << (e0, p)
        self.a1File << (e1, p)
        self.a2File << (e2, p)
    
        h_th = self.h_th
        h_th_proj = project(h_th,spline.V_control)
        h_th_proj.rename('w', 'w')
        self.wFile << (h_th_proj, p)
        '''
        energySurfaceDensity = self.energySurfaceDensity_proj
        energySurfaceDensity1 = self.energySurfaceDensity1_proj
        energySurfaceDensity2 = self.energySurfaceDensity2_proj
        energySurfaceDensity3 = self.energySurfaceDensity3_proj
        energySurfaceDensity.rename('Energy','total')
        energySurfaceDensity1.rename('Energy1','term1')
        energySurfaceDensity2.rename('Energy2','term2')
        energySurfaceDensity3.rename('Energy3','term3')
        
        self.energySurfaceDensityFile << (energySurfaceDensity, p)
        self.energySurfaceDensity1File << (energySurfaceDensity1, p)
        self.energySurfaceDensity2File << (energySurfaceDensity2, p)
        self.energySurfaceDensity3File << (energySurfaceDensity3, p)
        '''
        y0_zeroedInterior_hom = self.y0_zeroedInterior_hom 
        BC_hom_proj = project(y0_zeroedInterior_hom,spline.V)
        BC_hom_proj.rename('bc','bc')
        print('BC norm = ')
        print(BC_hom_proj.vector().norm('linf'))
        self.bcFile << (BC_hom_proj, p)

        y0_test_proj = project(self.y0_test,spline.V_control)
        y0_test_proj.rename("y0_test","y0_test")
        self.testFile << (y0_test_proj,p)

        y0_test2_proj = project(self.y0_test2,spline.V_control)
        y0_test2_proj.rename("y0_test2","y0_test2")
        self.testFile2 << (y0_test2_proj,p)
        
        y0_hom = spline.project(self.y0,rationalize=False,lumpMass=True)
        y0_test3_proj = project(y0_hom,spline.V)
        y0_test3_proj.rename("y0_test3","y0_test3")
        self.testFile3 << (y0_test3_proj,p)

        material = self.material
        if '6' in material.keys():
            domain_material = self.domain_material
            domain_material_proj = project(domain_material, FunctionSpace(mesh,'DG',0))
            domain_material_proj.rename('domain','belly_region')
            self.domain_material_File << (domain_material_proj, p)
        
        full_results = self.full_results
        full_results_geo = self.full_results_geo
        full_results_mech = self.full_results_mech
        #for n in range(len(full_results)):
        #    print('pressure = {:2.3}'.format(full_results[n]['pressure']))
        #    print(full_results[n].keys())
        with open(self.resultsFile,'w') as fp:
            json.dump(full_results,fp)
        with open(self.resultsFile_geo,'w') as fp:
            json.dump(full_results_geo,fp)
        with open(self.resultsFile_mech,'w') as fp:
            json.dump(full_results_mech,fp)
 
        contact_pairs = self.problem.contact_pairs
        with open(self.contactFiles_base+'.'+str(round(p*100))+'.csv','w') as f:
            f.write('L1 L2 X1 Y1 Z1 X2 Y2 Z2\n')
            for s in range(len(contact_pairs)):
                f.write('%d %d %f %f %f %f %f %f\n' %
                        (contact_pairs[s][0], contact_pairs[s][1],
                        contact_pairs[s][2], contact_pairs[s][3],
                        contact_pairs[s][4], contact_pairs[s][5],
                        contact_pairs[s][6], contact_pairs[s][7]))

if __name__ == "__main__":
    valve, is_valid, opt_params = generate_valve()
    contact = dict()
    contact['0'] = dict()
    #for the plane method
    contact['0']['k_cp'] = 1e3
    contact['0']['GAP_OFFSET'] = 0.019 

    contact['2'] = dict()
    #for the volumetric potential
    contact['2']['r_in_rel'] = 0.3
    contact['2']['r_out_rel'] = 0.9
    contact['2']['k_c'] = 1e-2
    contact['2']['p'] = 4.
    contact['2']['R_self_rel'] = 0.95
    material = dict()
    '''
    material['4'] = dict()
    #W = 0.5*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1) + 0.5*c3*(exp(c4*pow(I4 - 1.0,2.0))-1)
    material['4']['A'] = dict()
    material['4']['LR'] = dict()

    # unit: 30mmHg ~ 4000Pa
    material['4']['A']['c0'] = 1.
    material['4']['A']['c1'] = 0.1
    material['4']['A']['c2'] = 0.5
    material['4']['A']['c3'] = 0.1
    material['4']['A']['c4'] = 0.5
    material['4']['LR']['c0'] = 1.
    material['4']['LR']['c1'] = 0.05
    material['4']['LR']['c2'] = 0.5
    material['4']['LR']['c3'] = 0.1
    material['4']['LR']['c4'] = 0.5
    '''

    '''
    material['5'] = dict()
    #W = 0.5*c0*(I1-3.0) + 0.5*c1*(exp(c2*pow(I1 - 3.0,2.0))-1) + 0.5*c3*(exp(c4*pow(I4 - 1.0,2.0))-1)
    material['5']['A'] = dict()
    material['5']['LR'] = dict()

    # unit: 30mmHg ~ 4000Pa
    # coaption region
    material['5']['A']['c0_c'] = Constant(1.)
    material['5']['A']['c1_c'] = Constant(1.)
    material['5']['A']['c2_c'] = Constant(0.2)
    material['5']['A']['c3_c'] = Constant(0.1)
    material['5']['A']['c4_c'] = Constant(5.)

    material['5']['LR']['c0_c'] = Constant(1.)
    material['5']['LR']['c1_c'] = Constant(1.)
    material['5']['LR']['c2_c'] = Constant(0.2)
    material['5']['LR']['c3_c'] = Constant(0.1)
    material['5']['LR']['c4_c'] = Constant(5.)
    # belly region
    material['5']['A']['c0_b'] = Constant(1.)
    material['5']['A']['c1_b'] = Constant(1.)
    material['5']['A']['c2_b'] = Constant(0.2)
    material['5']['A']['c3_b'] = Constant(0.1)
    material['5']['A']['c4_b'] = Constant(5.)

    material['5']['LR']['c0_b'] = Constant(1.)
    material['5']['LR']['c1_b'] = Constant(1.)
    material['5']['LR']['c2_b'] = Constant(0.2)
    material['5']['LR']['c3_b'] = Constant(0.1)
    material['5']['LR']['c4_b'] = Constant(5.)
    '''
    material['6'] = dict()
    # Em is the normal strain along fiber 
    # En is the normal strain along direction perpendicular to the fiber 
    # Ephi is the in-plane shear
    # W = 0.5*b0*(I1-3.0) + 0.5*(b1*exp(b2*(I1-3)**2)-1) + 0.5*c0*(exp(c1*Em**2+c2*En**2+c3*Ephi**2+2*c4*Em*En)-1)
    material['6']['A'] = dict()
    material['6']['LR'] = dict()

    # unit: 30mmHg ~ 4000Pa
    # coaption region
    material['6']['A']['b0_c'] = Constant(1.)
    material['6']['A']['b1_c'] = Constant(0.1)
    material['6']['A']['b2_c'] = Constant(2.)
    material['6']['A']['c0_c'] = Constant(0.2)
    material['6']['A']['c1_c'] = Constant(9)
    material['6']['A']['c2_c'] = Constant(0.02)
    material['6']['A']['c3_c'] = Constant(0.1)
    material['6']['A']['c4_c'] = Constant(2)

    material['6']['LR']['b0_c'] = Constant(1.)
    material['6']['LR']['b1_c'] = Constant(0.1)
    material['6']['LR']['b2_c'] = Constant(2.)
    material['6']['LR']['c0_c'] = Constant(0.2)
    material['6']['LR']['c1_c'] = Constant(9)
    material['6']['LR']['c2_c'] = Constant(0.02)
    material['6']['LR']['c3_c'] = Constant(0.1)
    material['6']['LR']['c4_c'] = Constant(2)
    # belly region
    material['6']['A']['b0_b'] = Constant(1.)
    material['6']['A']['b1_b'] = Constant(0.1)
    material['6']['A']['b2_b'] = Constant(2.)
    material['6']['A']['c0_b'] = Constant(0.2)
    material['6']['A']['c1_b'] = Constant(9)
    material['6']['A']['c2_b'] = Constant(0.02)
    material['6']['A']['c3_b'] = Constant(0.1)
    material['6']['A']['c4_b'] = Constant(2)

    material['6']['LR']['b0_b'] = Constant(1.)
    material['6']['LR']['b1_b'] = Constant(0.1)
    material['6']['LR']['b2_b'] = Constant(2.)
    material['6']['LR']['c0_b'] = Constant(0.2)
    material['6']['LR']['c1_b'] = Constant(9)
    material['6']['LR']['c2_b'] = Constant(0.02)
    material['6']['LR']['c3_b'] = Constant(0.1)
    material['6']['LR']['c4_b'] = Constant(2)
 
    global_config['COPYFILES'] = True
    global_config['CHECKOUT'] = True # < checkout > output vtk files if True
    global_config['OUTPUT_CS_PROFILES'] = False # < evaluate_cost > write cross sectional profiles if True
    global_config['DOLFIN_LOG_LEVEL'] = 20
    global_config['INCLUDE_MISMATCH_for_A'] = True
    global_config['INCLUDE_MISMATCH_for_LR'] = True

    obj = leafletSim(valve,material,contact,opt_params,ID=1)
    #obj.main()
    obj.main_mp_version()



