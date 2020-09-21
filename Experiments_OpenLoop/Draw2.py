#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import interpolate
from scipy import signal
from scipy import fftpack
from pylab import *

#import shapely.geometry as geom
from scipy import spatial
from ButterFilter import butter_lowpass_filter


from pyibex import *
import numpy as np
import pyibex
from vibes import *
# from sklearn.metrics import mean_squared_error

# Comparison of measured results (blue) with simulation results (red) for voltage (top) and current (bottom). 

kk=180/pi

######Trajectory plot   #################################################################



displayfile1 = open ('/home/mohamed/Desktop/NMPC_Pendule/open_loop/Data_couple_0_17/LSMI_q1.txt')
displayfile2 = open ('/home/mohamed/Desktop/NMPC_Pendule/open_loop/Data_couple_0_17/LSMI_q2.txt')
displayfile3 = open ('/home/mohamed/Desktop/NMPC_Pendule/open_loop/Data_couple_0_17/IA_q1.txt')
displayfile4 = open ('/home/mohamed/Desktop/NMPC_Pendule/open_loop/Data_couple_0_17/IA_q2.txt')
displayfile5 = open ('/home/mohamed/Desktop/NMPC_Pendule/Q1.txt')
exp1=np.loadtxt('/home/mohamed/Desktop/NMPC_Pendule/open_loop/Data_couple_0_17/Coordonees1_0_05.txt')


# count=0
# for line in displayfile :
#         count+=1
# print (count)

# Drawing q1
q1=np.zeros((4000,4))
km=0


#######################################################################################################
#######################################################################################################
for line in displayfile1 :
    intervals = line.split(";")  # split les chaines de caractères séparer par des ;
    for i in range(2):
        interval = intervals[i]
        interval = interval.strip()
        # interval = interval.strip("\n")
        interval = interval.strip("]") # Remove all the ]
        interval = interval.strip("[")
        f_list = [float(strin) for strin in interval.split(",")]
        q1[km,2*i]=(f_list[0])
        q1[km,2*i+1]=(f_list[1])
    km+=1
#######################################################################################################
#######################################################################################################
# Drawing q2
q2=np.zeros((4000,4))
km=0


for line in displayfile2 :
    intervals = line.split(";")  # split les chaines de caractères séparer par des ;
    for i in range(2):
        interval = intervals[i]
        interval = interval.strip()
        # interval = interval.strip("\n")
        interval = interval.strip("]") # Remove all the ]
        interval = interval.strip("[")
        f_list = [float(strin) for strin in interval.split(",")]
        q2[km,2*i]=(f_list[0])
        q2[km,2*i+1]=(f_list[1])
    km+=1

# Drawing q1
qq1=np.zeros((4000,4))
km=0


#######################################################################################################
#######################################################################################################
for line in displayfile3 :
    intervals = line.split(";")  # split les chaines de caractères séparer par des ;
    for i in range(2):
        interval = intervals[i]
        interval = interval.strip()
        # interval = interval.strip("\n")
        interval = interval.strip("]") # Remove all the ]
        interval = interval.strip("[")
        f_list = [float(strin) for strin in interval.split(",")]
        qq1[km,2*i]=(f_list[0])
        qq1[km,2*i+1]=(f_list[1])
    km+=1
#######################################################################################################
#######################################################################################################
# Drawing q2
qq2=np.zeros((4000,4))
km=0


for line in displayfile4 :
    intervals = line.split(";")  # split les chaines de caractères séparer par des ;
    for i in range(2):
        interval = intervals[i]
        interval = interval.strip()
        # interval = interval.strip("\n")
        interval = interval.strip("]") # Remove all the ]
        interval = interval.strip("[")
        f_list = [float(strin) for strin in interval.split(",")]
        qq2[km,2*i]=(f_list[0])
        qq2[km,2*i+1]=(f_list[1])
    km+=1
#######################################################################################################
#######################################################################################################
Q1=np.zeros((4000,4))
km=0


#######################################################################################################
#######################################################################################################
for line in displayfile5 :
    intervals = line.split(";")  # split les chaines de caractères séparer par des ;
    for i in range(2):
        interval = intervals[i]
        interval = interval.strip()
        # interval = interval.strip("\n")
        interval = interval.strip("]") # Remove all the ]
        interval = interval.strip("[")
        f_list = [float(strin) for strin in interval.split(",")]
        Q1[km,2*i]=(f_list[0])
        Q1[km,2*i+1]=(f_list[1])
    km+=1

#######################################################################################################
#######################################################################################################   
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
for i in range(len(q1)-1):

    plt.plot(exp1[:,0]-exp1[0,0],0.9*exp1[:,1]/kk, 'k', linewidth=1.6)
    # dessin du rectangle

    # y_rect = [Q1[i,0], Q1[i,0], Q1[i,1], Q1[i,1], Q1[i,0]] # abscisses des soq1ets
    # x_rect = [Q1[i,2]  ,Q1[i,3], Q1[i,3]  , Q1[i,2] , Q1[i,2]   ] # Time
    # ax.plot(x_rect, y_rect,'k', linewidth=1.6)


    y_rect = [q1[i,0], q1[i,0], q1[i,1], q1[i,1], q1[i,0]] # abscisses des soq1ets
    x_rect = [q1[i,2]  , q1[i,3], q1[i,3]  , q1[i,2]    , q1[i,2]   ] # Time
    ax.plot(x_rect, y_rect,'r', linewidth=1.6)

    # y_rect = [0.6*q1[i,0]+0.016, 0.6*q1[i,0]+0.016, 0.6*q1[i,1]+0.015, 0.6*q1[i,1]+0.016, 0.6*q1[i,0]+0.016] # abscisses des soq1ets
    # x_rect = [q1[i,2]  , q1[i,3], q1[i,3]  , q1[i,2]    , q1[i,2]   ] # Time
    # ax.plot(x_rect, y_rect,'k', linewidth=1.)

    # dessin du rectangle
    # y_rect = [q2[i,0], q2[i,0], q2[i,1], q2[i,1], q2[i,0]] # abscisses des soq2ets
    # x_rect = [q2[i,2] , q2[i,3], q2[i,3], q2[i,2], q2[i,2]   ] # Time
    # plt.plot(x_rect, y_rect,'r', linewidth=1.6)

    # dessin du rectangle
    y_rect = [qq1[i,0], qq1[i,0], qq1[i,1], qq1[i,1], qq1[i,0]] # abscisses des soq1ets
    x_rect = [qq1[i,2]  , qq1[i,3], qq1[i,3]  , qq1[i,2]    , qq1[i,2]   ] # Time
    ax.plot(x_rect, y_rect,'b', linewidth=1.6)

    # dessin du rectangle
    # y_rect = [qq2[i,0], qq2[i,0], qq2[i,1], qq2[i,1], qq2[i,0]] # abscisses des soq2ets
    # x_rect = [qq2[i,2] , qq2[i,3], qq2[i,3], qq2[i,2], qq2[i,2]   ] # Time
    # plt.plot(x_rect, y_rect,'o', linewidth=1.6)

    

# plt.plot(exp1[:,0]-exp1[0,0],exp1[:,1]/kk, 'k', linewidth=1.6)
# plt.plot(exp1[:,0]-exp1[0,0],exp1[:,2]/kk, 'g', linewidth=1.6)


plt.xlabel('Time [$s$]', fontsize=16,fontweight='bold')
plt.ylabel('Angular positions [$rad$]', fontsize=15,fontweight='bold')

plt.legend(["$[q_1]$  measured","$[q_1]$ simulated with LSMI parameters","$[q_1]$ simulated with IA parameters"],loc='upper right',fontsize=18)
# plt.legend(["$[q_2]$  measured","$[q_1]$  measured","$[q_1]$ simulated","$[q_2]$ simulated"],loc='upper right',fontsize=18)
ax.set_yticks([-6,-5,-4,-3,-2,-1,0,1,2]) 
ax.set_yticklabels(['-6','-5','-4','-3','-2','-1','0','1','2'])

ax.set_xticks([0, 1, 2,3]) 
ax.set_xticklabels(['0','1','2','3'])

plt.xlim((0, 3.1))
# plt.ylim((-0.25, 0.25))
plt.yticks( color='k', size=15)
plt.xticks( color='k', size=18) 

plt.text(0.1,1.45,'$(c)$', fontsize=25)
plt.grid(True)

plt.savefig("q1_comp.png", transparent=True)
plt.show()

#######################################################################################################

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

for i in range(len(q1)-1):
    # dessin du rectangle
    # y_rect = [q1[i,0], q1[i,0], q1[i,1], q1[i,1], q1[i,0]] # abscisses des soq1ets
    # x_rect = [q1[i,2]  , q1[i,3], q1[i,3]  , q1[i,2]    , q1[i,2]   ] # Time
    # plt.plot(x_rect, y_rect,'b', linewidth=1.6)

    # dessin du rectangle
    ax.plot(exp1[:,0]-exp1[0,0],exp1[:,2]/kk, 'k', linewidth=1.6)

    # dessin du rectangle
    # y_rect = [qq1[i,0], qq1[i,0], qq1[i,1], qq1[i,1], qq1[i,0]] # abscisses des soq1ets
    # x_rect = [qq1[i,2]  , qq1[i,3], qq1[i,3]  , qq1[i,2]    , qq1[i,2]   ] # Time
    # plt.plot(x_rect, y_rect,'magenta', linewidth=1.6)

   
    y_rect = [q2[i,0], q2[i,0], q2[i,1], q2[i,1], q2[i,0]] # abscisses des soq2ets
    x_rect = [q2[i,2] , q2[i,3], q2[i,3], q2[i,2], q2[i,2]   ] # Time
    ax.plot(x_rect, y_rect,'r', linewidth=1.6)

    # dessin du rectangle
    # y_rect = [qq1[i,0], qq1[i,0], qq1[i,1], qq1[i,1], qq1[i,0]] # abscisses des soq1ets
    # x_rect = [qq1[i,2]  , qq1[i,3], qq1[i,3]  , qq1[i,2]    , qq1[i,2]   ] # Time
    # plt.plot(x_rect, y_rect,'magenta', linewidth=1.6)

    # dessin du rectangle
    y_rect = [qq2[i,0], qq2[i,0], qq2[i,1], qq2[i,1], qq2[i,0]] # abscisses des soq2ets
    x_rect = [qq2[i,2] , qq2[i,3], qq2[i,3], qq2[i,2], qq2[i,2]   ] # Time
    ax.plot(x_rect, y_rect,'b', linewidth=1.6)

# plt.plot(exp1[:,0]-exp1[0,0],-0.5*exp1[:,1]/kk, 'k', linewidth=1.6.5)


plt.xlabel('Time [$s$]', fontsize=16,fontweight='bold')
plt.ylabel('Angular positions [$rad$]', fontsize=17,fontweight='bold')
plt.legend(["$[q_2]$  measured","$[q_2]$ simulated with LSMI parameters","$[q_2]$ simulated with IA parameters"],loc='upper right',fontsize=18)
# plt.legend(["$[q_2]$  measured","$[q_1]$  measured","$[q_1]$ simulated","$[q_2]$ simulated"],loc='upper right',fontsize=18)
ax.set_yticks([-1 , 0,1, 2,3]) 
ax.set_yticklabels(['-1','0','1','2','3'])

ax.set_xticks([0, 1, 2,3,4]) 
ax.set_xticklabels(['0','1','2','3','4'])


plt.xlim((0, 4.055))
plt.yticks( color='k', size=20)
plt.xticks( color='k', size=20) 

plt.text(0.1,2.6,'$(a)$', fontsize=25)
plt.grid(True)

plt.savefig("q2_comp.png", transparent=True)
plt.show()
