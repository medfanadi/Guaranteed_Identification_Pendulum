
import numpy as np
from scipy.signal import butter, lfilter, freqz, firwin
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib import pyplot as plt
from math import *
from scipy import interpolate
from scipy import signal
import random

from scipy import spatial
from numpy import mean,pi,cos,sin,sqrt,tan,arctan,arctan2,tanh,arcsin,\
                    exp,dot,array,log,inf, eye, zeros, ones, inf,size,\
                    arange,reshape,concatenate,vstack,hstack,diag,median,sign,sum,meshgrid,cross,linspace,append,round


kk=pi/180
###################################################################################################
########     Acquision des données expérimentales sous formes des intervales  #####################
###################################################################################################
def Experiments() :
    # exp1 = np.loadtxt('/home/mohamed/Desktop/MOOC_Jaulin/IAMOOC_2/ch4/Iden_IA.txt')
    exp1 = np.loadtxt('/home/mohamed/Desktop/NMPC_Pendule/Parameters_Indentification/Data/Iden_IA.txt')

    const = len(exp1)
    # const=322
    # print(constt)
    T = np.zeros((const,2))
    q1 = np.zeros((const,2))
    q2 = np.zeros((const,2))
    dq1 = np.zeros((const,2))
    dq2 = np.zeros((const,2))
    ddq1 = np.zeros((const,2))
    ddq2 = np.zeros((const,2))
    tau1 = np.zeros((const,2))
    tau2 = np.zeros((const,2))

    X5= butter_lowpass_filter((exp1[:,8]/5)*kk, 0.038, 1/0.016, 8)
    ydq1 = derivateur2(X5);
    X5= butter_lowpass_filter((exp1[:,9]/4)*kk, 0.034, 1/0.016, 8)
    ydq2= derivateur2(X5);


    # plt.plot(exp1[:,0],ydq1,'b', linewidth = 3)
    # plt.plot(exp1[:,0],ydq2,'r', linewidth = 3)
    # plt.show()


    X5= butter_lowpass_filter((exp1[:,8]/5)*kk, 0.03, 1/0.016, 8)
    yddq1 = derivateur2(derivateur2(X5));
    X5= butter_lowpass_filter((exp1[:,9]/5)*kk, 0.03, 1/0.016, 8)
    yddq2= derivateur2(derivateur2(X5));

    # plt.plot(exp1[:,0],yddq1,'b', linewidth = 3)
    # plt.plot(exp1[:,0],yddq2,'r', linewidth = 3)
    # plt.show()

    TAU= butter_lowpass_filter(exp1[:,7], 0.05, 1/0.016, 8)
    for i in range(const-1):
        T[i,0] = exp1[i,0]-0.001
        T[i,1] = exp1[i,0]+0.001
        q1[i,0] = exp1[i,1]*kk-random.uniform(0, 10*pi/180)  
        q1[i,1] = exp1[i,1]*kk+random.uniform(0,10*pi/180)  
        q2[i,0] = exp1[i,2]*kk-random.uniform(0,10*pi/180)
        q2[i,1] = exp1[i,2]*kk-random.uniform(0,10*pi/180)
        dq1[i,0] = ydq1[i]-random.uniform(0,10.)  
        dq1[i,1] = ydq1[i]+random.uniform(0,10.)  
        dq2[i,0] = ydq2[i]-random.uniform(0,10.)  
        dq2[i,1] = ydq2[i]+random.uniform(0,10.)  
        ddq1[i,0] = yddq1[i]-random.uniform(0,15)  
        ddq1[i,1] = yddq1[i]+random.uniform(0,15)  
        ddq2[i,0] = yddq2[i]-random.uniform(0,15)  
        ddq2[i,1] = yddq2[i]+random.uniform(0,15)  
        tau1[i,0] = TAU[i]-random.uniform(0,3)  
        tau1[i,1] = TAU[i]+random.uniform(0,3)  
        tau2[i,0] = -random.uniform(0,3)  
        tau2[i,1] = random.uniform(0,3)  
    return T,q1,q2,dq1,dq2,ddq1,ddq2,tau1,tau2
def Drawing(T,q1,q2,dq1,dq2,ddq1,ddq2,tau1,tau2):

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
    for i in range(len(q1)-1):
        y_rect1  =  [q1[i,0],q1[i,1], q1[i,1], q1[i,0], q1[i,0]] # abscisses des sommets
        x_rect1  =  [T[i,0],T[i,1], T[i,1], T[i,0], T[i,0]] # ordonnees des sommets
        ax.plot(x_rect1, y_rect1,'b', linewidth = 3)

        y_rect2  =  [q2[i,0],q2[i,1], q2[i,1], q2[i,0], q2[i,0]] # abscisses des sommets
        x_rect2  =  [T[i,0],T[i,1], T[i,1], T[i,0], T[i,0]] # ordonnees des sommets
        ax.plot(x_rect2, y_rect2,'r', linewidth = 3)


    plt.xlabel('Time [$s$]', fontsize = 18,fontweight = 'bold')
    plt.ylabel('Angular positions [$rad$]', fontsize = 18,fontweight = 'bold')
    plt.legend(["$[q_1]$ measured","$[q_2]$ measured"],loc = 'upper right',fontsize = 19)
    plt.xlim((0, 14))
    plt.ylim((-2,2.1))
    ax.set_yticks([ -1.000, 0.000 , 1.000 , 2.000, 3.000 ]) 
    # ax.set_xticklabels(['zero','two','four','six'])
    plt.yticks( color = 'k', size = 17)
    plt.xticks( color = 'k', size = 17) 
    plt.grid(True)
    plt.text(1,2.45,'$(a)$', fontsize=32)
    plt.savefig("q.png", transparent = True)
    plt.show()

    for i in range(len(q1)-1):
        y_rect1  =  [dq1[i,0],dq1[i,1], dq1[i,1], dq1[i,0], dq1[i,0]] # abscisses des sommets
        x_rect1  =  [T[i,0],T[i,1], T[i,1], T[i,0], T[i,0]] # ordonnees des sommets
        plt.plot(x_rect1, y_rect1,'b', linewidth = 3)

        y_rect2  =  [dq2[i,0],dq2[i,1], dq2[i,1], dq2[i,0], dq2[i,0]] # abscisses des sommets
        x_rect2  =  [T[i,0],T[i,1], T[i,1], T[i,0], T[i,0]] # ordonnees des sommets
        plt.plot(x_rect2, y_rect2,'r', linewidth = 3)

    plt.xlabel('Time [$s$]', fontsize = 18,fontweight = 'bold')
    plt.ylabel('Angular Velocity [$rad.s^{-1}$]', fontsize = 18,fontweight = 'bold')
    plt.legend(["$[\\dot{q}_1]$ measured","$[\\dot{q}_2]$ measured"],loc = 'upper right',fontsize = 19)
    plt.xlim((0, 14))
    plt.ylim((-16, 24))
    plt.yticks( color = 'k', size = 17)
    plt.xticks( color = 'k', size = 17) 
    plt.grid(True)
    plt.text(1,20.5,'$(b)$', fontsize=32)
    plt.savefig("dq.png", transparent = True)
    plt.show()


    for i in range(len(q1)-1):
        y_rect1  =  [ddq1[i,0],ddq1[i,1], ddq1[i,1], ddq1[i,0], ddq1[i,0]] # abscisses des sommets
        x_rect1  =  [T[i,0],T[i,1], T[i,1], T[i,0], T[i,0]] # ordonnees des sommets
        plt.plot(x_rect1, y_rect1,'b', linewidth = 3)

        y_rect2  =  [ddq2[i,0],ddq2[i,1], ddq2[i,1], ddq2[i,0], ddq2[i,0]] # abscisses des sommets
        x_rect2  =  [T[i,0],T[i,1], T[i,1], T[i,0], T[i,0]] # ordonnees des sommets
        plt.plot(x_rect2, y_rect2,'r', linewidth = 3)

    plt.xlabel('Time [$s$]', fontsize = 16,fontweight = 'bold')
    plt.ylabel('Angular acceleration [$rad.s^{-2}$]', fontsize = 16,fontweight = 'bold')
    plt.legend(["$[\\ddot{q}_1]$ measured","$[\\ddot{q}_2]$ measured"],loc = 'upper right',fontsize = 19)
    plt.xlim((0.65, 14))
    plt.ylim((-110, 140))
    plt.yticks( color = 'k', size = 15)
    plt.xticks( color = 'k', size = 15) 
    plt.grid(True)
    plt.text(1,110,'$(c)$', fontsize=32)
    plt.savefig("ddq.png", transparent = True)
    plt.show()

    figg = plt.figure()
    axx = figg.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
    for i in range(len(q1)-1):
        y_rect1  =  [tau1[i,0],tau1[i,1], tau1[i,1], tau1[i,0], tau1[i,0]] # abscisses des sommets
        x_rect1  =  [T[i,0],T[i,1], T[i,1], T[i,0], T[i,0]] # ordonnees des sommets
        axx.plot(x_rect1, y_rect1,'b', linewidth = 3)

    # plt.plot(T[:,0],tau1[:,1],'b', linewidth = 3)
        # y_rect2  =  [ddq2[i,0],ddq2[i,1], ddq2[i,1], ddq2[i,0], ddq2[i,0]] # abscisses des sommets
        # x_rect2  =  [T[i,0],T[i,1], T[i,1], T[i,0], T[i,0]] # ordonnees des sommets
        # plt.plot(x_rect2, y_rect2,'r', linewidth = 3)

    plt.xlabel('Time [$s$]', fontsize = 18,fontweight = 'bold')
    plt.ylabel('Motor Torque [$Nm$]', fontsize = 15,fontweight = 'bold')
    plt.legend(["$[\\tau]$ measured"],loc = 'lower right',fontsize = 22)
    axx.set_yticks([0.13, 0.15 , 0.17, 0.19, 0.21]) 
    axx.set_yticklabels(['0.13','0.15','0.17','0.19','0.21'])
    plt.xlim((0, 14))
    plt.ylim((0.115, 0.21))
    plt.yticks( color = 'k', size = 14)
    plt.xticks( color = 'k', size = 15) 
    plt.grid(True)
    plt.text(1,0.2,'$(d)$', fontsize=32)
    plt.savefig("tau.png", transparent = True)
    plt.show()


def butter_lowpass_filter(data, cutOff, fs, order=6):
    b, a = signal.butter(order, cutOff)
    y =signal.filtfilt(b, a, data)
    return y

def firwin_filter(data, cutOff, fs, order=6):
    numtaps = 50
    a = signal.firwin(numtaps, cutOff, window = "hamming")
    b = [1.0]
    y =signal.filtfilt(b, a, data)
    return y

def derivateur (data):
    Fs=1/0.016 #fréquence d'échantillonage (hz)
    G = pi*Fs
    B = G*signal.remez(50,[0,0.9],[0,0.9],type='differentiator') #firpm Matlab
    y = signal.filtfilt(B,1,data);
    return y

def derivateur2 (data):
    Te=0.0016
    Fs=1/Te #fréquence d'échantillonage (hz)
    a=[Te]
    b=[1,-1]
    y = signal.filtfilt([-1.36563231092433,2.78569107455510,-3.40888532196508,4.76847704904398,-6.63220992694741,9.18901783379322,-12.8702153122317,18.7030989693578,-29.7992142870062,61.7649620395912,0,-61.7649620395912,29.7992142870062,-18.7030989693578,12.8702153122317,-9.18901783379322,6.63220992694741,-4.76847704904398,3.40888532196508,-2.78569107455510,1.36563231092433],1,data)
    return y