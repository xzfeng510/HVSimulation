import meshio
import numpy as np
import os
import sys
import glob
import xml.etree.ElementTree as ET
from scipy.io import savemat
from math import sqrt

N1 = 20
N2 = 20

def extract_data(directory,filename,key):
    data = meshio.read(os.path.join(directory,filename))
    return data.point_data[key]

def extrude_volume(directory,filename,filename_head,key):
    VERBOSE = False
    [fname,fmt] = filename.split('.')
    if VERBOSE:
        print('filename = '+fname+'.'+fmt)
    if ('_vol' in fname) or (fmt != 'vtu'):
        if VERBOSE:
            print('Skip the file')
        return
    [_,timestep] = fname.split(filename_head)
    if VERBOSE:
        print('time step = '+timestep)
    
    data = meshio.read(os.path.join(directory,filename))
    tmp1 = np.array(data.points)
    tmp2 = np.array(data.points)
    tmp3 = np.array(data.points)
    tmp1[:,2] = tmp1[:,2]-1
    tmp3[:,2] = tmp3[:,2]+1
    modified_points = np.append(tmp1,tmp2,axis=0)
    modified_points = np.append(modified_points,tmp3,axis=0)

    tmp1 = np.array(data.cells[0].data)
    shape = tmp1.shape
    new_shape = np.array(shape)
    new_shape[0] *= 2
    new_shape[1] *= 2
    tmp2 = np.empty(new_shape,dtype=np.int)

    N = len(data.points)
    for n in range(shape[0]):
        tmp2[n,:] = [tmp1[n,0],tmp1[n,1],tmp1[n,2],tmp1[n,3],
                    tmp1[n,0]+N,tmp1[n,1]+N,tmp1[n,2]+N,tmp1[n,3]+N]
    for n in range(shape[0]):
        tmp2[n+shape[0],:] = [tmp1[n,0]+N,tmp1[n,1]+N,tmp1[n,2]+N,tmp1[n,3]+N,
                    tmp1[n,0]+N*2,tmp1[n,1]+N*2,tmp1[n,2]+N*2,tmp1[n,3]+N*2]
    
    modified_cells = []
    modified_cells.append(meshio.CellBlock('hexahedron', tmp2))
    modified_point_data = {}
    tmp1 = np.array(data.point_data[key])
    tmp2 = np.array(data.point_data[key])
    tmp3 = np.array(data.point_data[key])

    a0 = extract_data(directory,'a0'+timestep+'.vtu','e0')
    a1 = extract_data(directory,'a1'+timestep+'.vtu','e1')
    a2 = extract_data(directory,'a2'+timestep+'.vtu','e2')
    Lw = extract_data(directory,'Lambda-w'+timestep+'.vtu','lambda_w')
    Fx = extract_data(directory,'F-x'+timestep+'.vtu','F0')
    Fy = extract_data(directory,'F-y'+timestep+'.vtu','F1')
    Fz = extract_data(directory,'F-z'+timestep+'.vtu','F2')
    dispx = extract_data(directory,'disp-x'+timestep+'.vtu','d0')
    dispy = extract_data(directory,'disp-y'+timestep+'.vtu','d1')
    dispz = extract_data(directory,'disp-z'+timestep+'.vtu','d2')
    spatial_coordinates_x = Fx+dispx
    spatial_coordinates_y = Fy+dispy
    spatial_coordinates_z = Fz+dispz 
    spatial_coordinates = dict()
    spatial_coordinates['x'] = spatial_coordinates_x
    spatial_coordinates['y'] = spatial_coordinates_y
    spatial_coordinates['z'] = spatial_coordinates_z
    savemat(os.path.join(directory,'./mesh/points.mat'),spatial_coordinates)
    nodulus_indices = dict()
    nodulus_indices['A'] = []
    nodulus_indices['L'] = []
    nodulus_indices['R'] = []
    
    thickness = []
    c_span_A = 0.25
    r_span_A = 0.3
    c_span_LR = 0.25
    r_span_LR = 0.25
    buldge_coeff_A = 4.5
    buldge_coeff_LR = 4.5
    for n in range(N):
        po = data.points[n,:]
        p = [po[0]-np.floor(po[0]),po[1]]
        if np.floor(po[0]) >= 0:
            valve = 'anterior'
            c_center = 0.5
            r_center = 0.97
            buldge_coeff = buldge_coeff_A
            c_span = c_span_A
            r_span = r_span_A
        if np.floor(po[0]) >= 2:
            valve = 'left'
            c_center = 0.5
            r_center = 0.97
            buldge_coeff = buldge_coeff_LR
            c_span = c_span_LR
            r_span = r_span_LR
        if np.floor(po[0]) >= 4:
            valve = 'right'
            c_center = 0.5
            r_center = 0.97
            buldge_coeff = buldge_coeff_LR
            c_span = c_span_LR
            r_span = r_span_LR

        if p[1] >= r_center-r_span/2 and p[1] < 1 and p[0] >= c_center-c_span/2 and p[0] <= c_center+c_span/2:
            modifier = 1+(buldge_coeff-1)*(p[1]-r_center+r_span/2)*(r_center+r_span/2-p[1])/r_span**2*(p[0]-c_center+c_span/2)*(c_center+c_span/2-p[0])/c_span**2*16
            if valve == 'anterior':
                nodulus_indices['A'].append(n)
            if valve == 'left':
                nodulus_indices['L'].append(n)
            if valve == 'right':
                nodulus_indices['R'].append(n)
        else:
            modifier = 1
        thickness.append(0.048*modifier*Lw[n])
        #thickness.append(0.048*modifier)
    
    savemat(os.path.join(directory,'./mesh/nodulus_indices.mat'),nodulus_indices)
    ''' 
    curve_R = [[x,y,z] for x,y,z in zip(spatial_coordinates['x'][:], spatial_coordinates['y'][:], spatial_coordinates['z'][:])]
    curve_R_dict = dict()
    curve_R_dict['x'] = [p[0] for p in curve_R]
    curve_R_dict['y'] = [p[1] for p in curve_R]
    curve_R_dict['z'] = [p[2] for p in curve_R]
    savemat(os.path.join(directory,'./mesh/curve_R.mat'),curve_R_dict)
    print(calc_curve_length(curve_R))
    exit()
    '''

    if key == 'F0':
        #tmp1 = tmp2-np.array(a2[:,0]) * 0.1
        #tmp3 = tmp2+np.array(a2[:,0]) * 0.1
        tmp1 = tmp2-np.multiply(np.array(a2[:,0]),np.array(thickness)/2)
        tmp3 = tmp2+np.multiply(np.array(a2[:,0]),np.array(thickness)/2)
        #print(tmp1)
        #print(tmp1.shape)
        #print(np.array(thickness)/2)
        #exit()
 
    if key == 'F1':
        #tmp1 = tmp2-np.array(a2[:,1]) * 0.1
        #tmp3 = tmp2+np.array(a2[:,2]) * 0.1
        tmp1 = tmp2-np.multiply(np.array(a2[:,1]),np.array(thickness)/2)
        tmp3 = tmp2+np.multiply(np.array(a2[:,1]),np.array(thickness)/2)
    if key == 'F2':
        #tmp1 = tmp2-np.array(a2[:,1]) * 0.1
        #tmp3 = tmp2+np.array(a2[:,2]) * 0.1
        tmp1 = tmp2-np.multiply(np.array(a2[:,2]),np.array(thickness)/2)
        tmp3 = tmp2+np.multiply(np.array(a2[:,2]),np.array(thickness)/2)
 
    modified_point_data[key] = np.append(tmp1,tmp2,axis=0)
    modified_point_data[key] = np.append(modified_point_data[key],tmp3,axis=0)

    meshio.Mesh(modified_points,modified_cells,point_data=modified_point_data).write(os.path.join(directory,fname+'_vol.'+fmt))

def calc_curve_length(curve):
    numpy_curve = np.array(curve)
    length = 0
    for i in range(numpy_curve.shape[0]-1):
        length += sqrt(np.power(curve[i][0]-curve[i+1][0], 2) +
                    np.power(curve[i][1]-curve[i+1][1], 2) +
                    np.power(curve[i][2]-curve[i+1][2], 2))
    return length

def get_midpoint(directory,timestep):
    pass

def generate_vol_pvd(directory,filename_head):
    tree = ET.parse(os.path.join(directory,filename_head+'.pvd'))
    root = tree.getroot()
    '''
    print(tree)
    print(root)
    print(root[0][0].attrib)
    print(root[0][1].attrib)
    print(root[0][2].attrib)
    print(len(root[0]))
    '''
    for item in root[0]:
        '''
        print(item)
        print(item.attrib['timestep'])
        print(item.attrib['part'])
        print(item.attrib['file'])
        '''
        fname = item.get('file')
        [_,part] = fname.split(filename_head)
        [timestep,fmt] = part.split('.')
        item.set('file',filename_head+timestep+'_vol.'+fmt)
    tree.write(os.path.join(directory,filename_head+'_vol.pvd'))


def run(directory):
    #directory = 'results/test_mat5_x'
    if directory:
        print('directory = '+directory)
    else:
        print('The directory of the result files needs to be specified!')
        exit()
    print('Processing F-x ...')
    for full_filename in glob.glob(directory+'/F-x*'):
        filename = os.path.basename(full_filename)
        extrude_volume(directory,filename,'F-x','F0')
    generate_vol_pvd(directory,'F-x')
    print('Processing F-y ...')
    for full_filename in glob.glob(directory+'/F-y*'):
        filename = os.path.basename(full_filename)
        extrude_volume(directory,filename,'F-y','F1')
    generate_vol_pvd(directory,'F-y')
    print('Processing F-z ...')
    for full_filename in glob.glob(directory+'/F-z*'):
        filename = os.path.basename(full_filename)
        extrude_volume(directory,filename,'F-z','F2')
    generate_vol_pvd(directory,'F-z')
    print('Processing disp-x ...')
    for full_filename in glob.glob(directory+'/disp-x*'):
        filename = os.path.basename(full_filename)
        extrude_volume(directory,filename,'disp-x','d0')
    generate_vol_pvd(directory,'disp-x')
    print('Processing disp-y ...')
    for full_filename in glob.glob(directory+'/disp-y*'):
        filename = os.path.basename(full_filename)
        extrude_volume(directory,filename,'disp-y','d1')
    generate_vol_pvd(directory,'disp-y')
    print('Processing disp-z ...')
    for full_filename in glob.glob(directory+'/disp-z*'):
        filename = os.path.basename(full_filename)
        extrude_volume(directory,filename,'disp-z','d2')
    generate_vol_pvd(directory,'disp-z')


if __name__ == "__main__":
    run(sys.argv[1])
