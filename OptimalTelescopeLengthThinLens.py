# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:31:55 2019

@author: Shaun
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy import interpolate
import pandas as pd 
from progress.bar import Bar


#q as in the abcd parameter for gaussian optics, calculate and return
def q(R,Y,n,w):
    inverse_q = complex(1/R,-Y/(np.pi*n*w**2))
    return 1/inverse_q

#propagate the beam q forward by distance d
def propagate(q,d):
    abcd = np.matrix([[1,d],[0,1]])
    vect = np.matrix([[q],[1]])
    result = np.dot(abcd,vect)
    return result.item(0,0)/result.item(1,0)

#refraction of beam at interface (nbk-7 at 780 -> n = 1.511)
def interface(q,n1,n2):
    abcd = np.matrix([[1,0],[0,n1/n2]])
    vect = np.matrix([[q],[1]])
    result = np.dot(abcd,vect)
    return result.item(0,0)/result.item(1,0)

def dielectric_slab(q,n1,n2,width):
    q1 = interface(q,n1,n2)
    q2 = propagate(q1,width)
    q3 = interface(q2,n2,n1)
    return q3

#modify the beam q, by lens, focal length f
def thin_lens(q,f):
    abcd = np.matrix([[1,0],[-1/f,1]])
    vect = np.matrix([[q],[1]])
    result = np.dot(abcd,vect)
    return result.item(0,0)/result.item(1,0)

#When is the beam some radius if a lens, focal length f1, is placed at d1
def d2_size(arg):
    #q0 = q(1,780e-9,1,3e-6)
    #d1 = 45e-3
    #f1 = -25e-3
    #radius = 15e-3
    q1 = propagate(q0,d1)
    q2 = thin_lens(q1,focal_1)
    q3 = propagate(q2,(arg-3.2e-3)/2-5e-3)
    # WPQ10E-780 1.6 mm, "n" at 780 = 1.511
    q4 = dielectric_slab(q3,1,1.511,3.2e-3)
    q5 = propagate(q4,(arg-3.2e-3)/2+5e-3)
    invert_q = 1/q5
    im_invert_q = -invert_q.imag
    #print ((780e-9/(np.pi*1*im_invert_q))**0.5,arg)
    return abs((780e-9/(np.pi*1*im_invert_q))**0.5-radius)

#what focal length is required to collimate the beam after the d2_size scenario
def f2_opt(f):
    #q0 = q(1e-6,780e-9,1,3e-6)
    #f1 = -50e-3
    f2 = f
    #d1 = 45e-3
    #d2 = distance_2
    q1 = propagate(q0,d1)
    q2 = thin_lens(q1,focal_1)
    q3 = propagate(q2,(d2-3.2e-3)/2-5e-3)
    q4 = dielectric_slab(q3,1,1.511,3.2e-3)
    q5 = propagate(q4,(d2-3.2e-3)/2+5e-3)
    q6 = thin_lens(q5,f2)
    invert = 1/q6
    return abs(invert.real)



def d1_size(arg):
    #q0 = q(0.1,780e-9,1,3e-6)
    #d1 = 50e-3
    #f1 = -50e-3
    #radius = 15e-3
    q1 = propagate(q0,arg)
    
    invert_q = 1/q1
    im_invert_q = -invert_q.imag
    #print ((780e-9/(np.pi*1*im_invert_q))**0.5,arg)
    return (780e-9/(np.pi*1*im_invert_q))**0.5


#fixed parameters
q0 = q(1,780e-9,1,3e-6)
radius = 15e-3
focal_1 = -15e-3

#arrays for graph
no_points = 10000
distance_1 = np.linspace(0.005,0.1,no_points)
distance_2 = []
focal_2 = []
size_1 = []


#move lens 1 to different places
bar = Bar(r'Processing', max=no_points)
for i in distance_1:
    #where is the lens some radius, so lens 2 can be placed there
    d1 = i
    opt_d2 = opt.minimize(d2_size,-focal_1 ,method='Nelder-Mead')
    d2 = opt_d2.x[0]
    distance_2.append(d2)
    #what focal length is required to collimate this beam
    opt_f2 = opt.minimize(f2_opt,-focal_1 ,method='Nelder-Mead')
    f2 = opt_f2.x[0]
    focal_2.append(f2)
    size_1.append(d1_size(d1))
    bar.next()

bar.finish()


# Scaling
focal_1 = np.multiply(focal_1,1000)
distance_1 = np.multiply(distance_1,1000)
distance_2 = np.multiply(distance_2,1000)
focal_2 = np.multiply(focal_2,1000)
total = distance_1+distance_2
size_1 =np.multiply(size_1,1000)

data = {"Distance 1":distance_1,"Distance 2":distance_2,"Focal Length 1": focal_1,"Focal Length 2": focal_2,"Total":total,"Size of beam at 1": size_1}
df=pd.DataFrame(data)
df.to_csv("telescope_simulation_results_10000.csv")

# Minimum Telescope Length for chosen f1
min_index = np.argmin(total)
min_dis_1 = distance_1[min_index]
min_dis_2 = distance_2[min_index]
min_size_at_1 = size_1[min_index]
min_focal_2 = focal_2[min_index]
min_total = total[min_index]
print("Minimal Telescope Length")
print('f1 %.2f mm \nd1 %.2f mm \nd2 %.2f mm \nf2 %.2f mm \nbeam size at d1 %.2f mm \ntotal length %.2f mm \n'\
     %(focal_1,min_dis_1,min_dis_2,min_focal_2,min_size_at_1,min_total))

# Find values for points where the mirror focal length closest to focal_2_val
focal_2_val = 50.8
fixed_mirror_index = np.argmin(np.abs(focal_2-focal_2_val))
fixed_mirror_dis_1 = distance_1[fixed_mirror_index]
fixed_mirror_dis_2 = distance_2[fixed_mirror_index]
fixed_mirror_focal_2 = focal_2[fixed_mirror_index]
fixed_size_1 = size_1[fixed_mirror_index]
fixed_total = total[fixed_mirror_index]
print("Telescope Lengths with 50.8 mm mirror, series")
print('f1 %.2f mm \nd1 %.2f mm \nd2 %.2f mm \nf2 %.2f mm \nbeam size at d1 %.2f mm \ntotal length %.2f mm \n' \
    %(focal_1,fixed_mirror_dis_1,fixed_mirror_dis_2,fixed_mirror_focal_2,fixed_size_1,fixed_total))

# Interpolate the points as the optimisation routine outputs an intervalled list. 
# This represents the data as continuously, and allows us to obtain the points where the mirror focal length is exactly focal_2_val

int_distance_1_f = interpolate.interp1d(focal_2,distance_1)
int_distance_1 = int_distance_1_f(50.8)
int_distance_2_f = interpolate.interp1d(distance_1, distance_2)
int_distance_2 = int_distance_2_f(int_distance_1)
int_size_f = interpolate.interp1d(distance_1,size_1)
int_size = int_size_f(int_distance_1)

print("Telescope Lengths with 50.8 mm mirror, continuous")
print('f1 %.2f mm \nd1 %.2f mm \nd2 %.2f mm \nf2 %.2f mm \nbeam size at d1 %.2f mm \ntotal length %.2f mm \n'\
     %(focal_1,int_distance_1,int_distance_2,50.8,int_size,(int_distance_1+int_distance_2)))



#prepare plot
plt.xlabel('Distance to lens 1 (mm)')
#plt.ylabel('Distance to lens 2/focal length of lens 2 \n /total distance (mm)')
plt.ylabel('Size (mm)')
plt.title('Minimal telescope with initial focal length %.1f mm' %focal_1)
plt.plot(distance_1,distance_2,'b',label='Distance 2')
plt.plot(distance_1,focal_2,'r',label='Focal length 2')
plt.plot(distance_1,total,'g',label='Total distance')
plt.plot(distance_1,size_1*10,'m',label='Size of beam x 10')
plt.vlines(int_distance_1,0,100)
plt.legend()


plt.show()







