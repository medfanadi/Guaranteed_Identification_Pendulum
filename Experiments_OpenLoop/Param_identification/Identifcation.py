from pyibex import *
import pyibex
from vibes import *
import pyibex
import numpy as np
from math import *
import numpy as np
import matplotlib.pyplot as plt
import random
from Functions import *

kk = pi/180


if __name__  ==  '__main__' :


	T_e,q1_e,q2_e,dq1_e,dq2_e,ddq1_e,ddq2_e,tau1_e,tau2_e = Experiments()
	# Drawing(T_e,q1_e,q2_e,dq1_e,dq2_e,ddq1_e,ddq2_e,tau1_e,tau2_e)

	# # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
	# # Déclaration des constantes (Pendule inverse + Moteur)
	# # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
	mp = Interval(0.04); 		       			# masse du pendule [kg]
	ma = Interval(0.2);  		    			# masse du bras rotatif [kg]
	M = Interval(0.015);						# masse accrochée  [kg]
	la = Interval(0.1); 	     				# longueur du bras rotatif [m]
	lp = Interval(0.175);   					# longeur du pendule [m]
	r0 = Interval(0.05);						# longeur par rapport à la base 
	N = Interval(5.0);  						# Rapport de réduction engrenage  
	Ja = Interval(1834.0*1e-6); 				# inertie bras rotatif [kg.m^2] #Ja((1.0/3.0)*ma*pow(la,2)); 
	Jp = Interval(562.802*1e-6);		  		# inertie pendule [kg.m^2]   #Jp((1.0/12.0)*ma*pow(lp,2));
   	# Interval aa,bb;
   	# aa = (1.0/3.0)*ma*pow(la,2);
   	# bb = (1.0/12.0)*mp*pow(lp,2);
   	# cout << aa <<bb <<endl;
	g = Interval(9.81);  						# gravité

	fv1 = Interval(0.0619);  					# Viscous Colomb friction coefficient  joint 1
	fc1 = Interval(0.0);  					# Static Colomb friction coefficient  joint 1
	fv2 = Interval(0.0007);  					# Viscous Colomb friction coefficient  joint 2
	fc2 = Interval(0.0);  					# Static Colomb friction coefficient  joint 2


   	# Moteur courant continu DCX22LG - MDP

	Ke = Interval(1.0/((981.0*2.0*pi)/60.0));		#  Constante de vitesse [981 tr/min/V]
	Kc = Interval(9.73e-3);  					#  Constante de couple [9.73 mNm/A]
	L = Interval(0.0346e-3);					# Inductance [Heneri]
	f = Interval (0);							# coeficient frottement visqueux
	Rm = Interval(0.335);						# Résistance de l'induit (ohm)
	Jm = Interval(9.06e-7);					# Inertie Moteur [g.cm^2] 
	mu_m = Interval(0.906);  					# Rendement Moteur 
	Im = Interval(0.0818);         			# courant à vide
	Cr = Interval(Kc*Im);         			# Couple résistif à vide
	Nm = Interval(5.2);						# Rapport du réduction moteur 
	Rv = Interval(34.0*(2.0*pi/60)); 			#*2*pi)/60			# Pente vitesse/couple [tr/mn/mN]
	n0 = Interval(2250.0*(2.0*pi/60));   		#  Vitesse à vide 

   	# Paramètres dynmaiques


	mu1 = Interval(lp**2*(mp/4.0+M));						# Dynamic parameter
	mu2 = Interval(la**2*(mp+M)+Jm/(N**2)+Ja);		# Dynamic parameter
	mu3 = Interval(lp*la*(mp/2+M));							# Dynamic parameter
	mu4 = Interval(lp**2*(mp/4.0+M)+Jp);					# Dynamic parameter
	mug = Interval(lp*g*(mp/2+M));


	# q1 = Interval(1,1.2)
	# q2 = Interval(2,2)
	# dq1 = Interval(1,1.2)
	# dq2 = Interval(2,2)
	# ddq1 = Interval(1,1.2)
	# ddq2 = Interval(2,2)
	# tau1 = Interval(1,1.2)
	# tau2 = Interval(2,2)
	

	# T=[T_e[100,1],T_e[200,1],T_e[300,1],T_e[320,1]]
	# Y1= [Interval(tau1_e[100,0],tau1_e[100,1]),Interval(tau1_e[200,0],tau1_e[200,1]),Interval(tau1_e[300,0],tau1_e[300,1]),Interval(tau1_e[320,0],tau1_e[320,1])]
	# q1=[[q1_e[100,0],q1_e[100,1]], [q1_e[150,0],q1_e[150,1]],[q1_e[200,0],q1_e[200,1]],[q1_e[250,0],q1_e[250,1]], [q1_e[300,0],q1_e[300,1]], [q1_e[350,0],q1_e[350,1]], [q1_e[320,0],q1_e[320,1]],[q1_e[450,0],q1_e[450,1]]]
	# q2=[[q2_e[100,0],q2_e[100,1]], [q2_e[150,0],q2_e[150,1]],[q2_e[200,0],q2_e[200,1]],[q2_e[250,0],q2_e[250,1]], [q2_e[300,0],q2_e[300,1]], [q2_e[350,0],q2_e[350,1]], [q2_e[320,0],q2_e[320,1]],[q2_e[450,0],q2_e[450,1]]]
	# dq1=[[dq1_e[100,0],dq1_e[100,1]], [dq1_e[150,0],dq1_e[150,1]],[dq1_e[200,0],dq1_e[200,1]],[dq1_e[250,0],dq1_e[250,1]], [dq1_e[300,0],dq1_e[300,1]], [dq1_e[350,0],dq1_e[350,1]], [dq1_e[320,0],dq1_e[320,1]],[dq1_e[450,0],dq1_e[450,1]]]
	# dq2=[[dq2_e[100,0],dq2_e[100,1]], [dq2_e[150,0],dq2_e[150,1]],[dq2_e[200,0],dq2_e[200,1]],[dq2_e[250,0],dq2_e[250,1]], [dq2_e[300,0],dq2_e[300,1]], [dq2_e[350,0],dq2_e[350,1]], [dq2_e[320,0],dq2_e[320,1]],[dq2_e[450,0],dq2_e[450,1]]]
	# ddq1=[[ddq1_e[100,0],ddq1_e[100,1]], [ddq1_e[150,0],ddq1_e[150,1]],[ddq1_e[200,0],ddq1_e[200,1]],[ddq1_e[250,0],ddq1_e[250,1]], [ddq1_e[300,0],ddq1_e[300,1]], [ddq1_e[350,0],ddq1_e[350,1]], [ddq1_e[320,0],ddq1_e[320,1]],[ddq1_e[450,0],ddq1_e[450,1]]]
	# ddq2=[[ddq2_e[100,0],ddq2_e[100,1]], [ddq2_e[150,0],ddq2_e[150,1]],[ddq2_e[200,0],ddq2_e[200,1]],[ddq2_e[250,0],ddq2_e[250,1]], [ddq2_e[300,0],ddq2_e[300,1]], [ddq2_e[350,0],ddq2_e[350,1]], [ddq2_e[320,0],ddq2_e[320,1]],[ddq2_e[450,0],ddq2_e[450,1]]]
	
	T=[T_e[100,1],T_e[200,1],T_e[300,1],T_e[320,1]]
	Y1= [Interval(tau1_e[100,0],tau1_e[100,1]),Interval(tau1_e[200,0],tau1_e[200,1]),Interval(tau1_e[300,0],tau1_e[300,1]),Interval(tau1_e[400,0],tau1_e[400,1])]
	q1=[[q1_e[100,0],q1_e[100,1]], [q1_e[200,0],q1_e[200,1]], [q1_e[300,0],q1_e[300,1]],[q1_e[400,0],q1_e[400,1]]]
	q2=[[q2_e[100,0],q2_e[100,1]], [q2_e[200,0],q2_e[200,1]], [q2_e[300,0],q2_e[300,1]], [q2_e[400,0],q2_e[400,1]]]
	dq1=[[dq1_e[100,0],dq1_e[100,1]], [dq1_e[200,0],dq1_e[200,1]], [dq1_e[300,0],dq1_e[300,1]], [dq1_e[400,0],dq1_e[400,1]]]
	dq2=[[dq2_e[100,0],dq2_e[100,1]], [dq2_e[200,0],dq2_e[200,1]], [dq2_e[300,0],dq2_e[300,1]], [dq2_e[400,0],dq2_e[400,1]]]
	ddq1=[[ddq1_e[100,0],ddq1_e[100,1]], [ddq1_e[200,0],ddq1_e[200,1]], [ddq1_e[300,0],ddq1_e[300,1]], [ddq1_e[400,0],ddq1_e[400,1]]]
	ddq2=[[ddq2_e[100,0],ddq2_e[100,1]], [ddq2_e[200,0],ddq2_e[200,1]], [ddq2_e[300,0],ddq2_e[300,1]], [ddq2_e[400,0],ddq2_e[400,1]]]
	# tau1=[[6,12], [-2,-5], [-3,10], [3,4]]
	# tau2=[[6,12], [-2,-5], [-3,10], [3,4]]
	# Creation of a separator with 
	seps = [] # empty list of separator
	# iterate over Experimental data
	for (q1lb,q1ub),(q2lb,q2ub),(dq1lb,dq1ub),(dq2lb,dq2ub),(ddq1lb,ddq1ub),(ddq2lb,ddq2ub),y in zip(q1,q2,dq1,dq2,ddq1,ddq2,Y1): 
		# f11 = Function("p[7]","p[1]*[%f,%f]+p[0]*sin([%f,%f])*sin([%f,%f])*[%f,%f]+p[2]*cos([%f,%f])*[%f,%f]-p[2]*sin([%f,%f])*[%f,%f]*[%f,%f]+2*p[0]*cos([%f,%f])*sin([%f,%f])*[%f,%f]*[%f,%f]"%(ddq1lb,ddq1ub,q2lb,q2ub,q2lb,q2ub,ddq1lb,ddq1ub,q2lb,q2ub,ddq2lb,ddq2ub,q2lb,q2ub,dq2lb,dq2ub,dq2lb,dq2ub,q2lb,q2ub,q2lb,q2ub,dq1lb,dq1ub,dq2lb,dq2ub))
		# f22 = Function("p[7]","p[2]*cos([%f,%f])*[%f,%f]+p[3]*[%f,%f]-p[0]*cos([%f,%f])*sin([%f,%f])*[%f,%f]*[%f,%f]+p[4]*sin([%f,%f])"%(q2lb,q2ub,ddq1lb,ddq1ub,ddq2lb,ddq2ub,q2lb,q2ub,q2lb,q2ub,dq1lb,dq1ub,dq1lb,dq1ub,q2lb,q2ub))
		# f11 = Function("p[2]","[0.00238404, 0.00238404]*[%f,%f]+[0.000765625, 0.000765625]*sin([%f,%f])*sin([%f,%f])*[%f,%f]+[0.0006125, 0.0006125]*cos([%f,%f])*[%f,%f]-[0.0006125, 0.0006125]*sin([%f,%f])*[%f,%f]*[%f,%f]+2*[0.000765625, 0.000765625]*cos([%f,%f])*sin([%f,%f])*[%f,%f]*[%f,%f]+p[0]*[%f,%f]+2*p[1]*[%f,%f]"%(ddq1lb,ddq1ub,q2lb,q2ub,q2lb,q2ub,ddq1lb,ddq1ub,q2lb,q2ub,ddq2lb,ddq2ub,q2lb,q2ub,dq2lb,dq2ub,dq2lb,dq2ub,q2lb,q2ub,q2lb,q2ub,dq1lb,dq1ub,dq2lb,dq2ub,dq1lb,dq1ub,q2lb,q2ub))
		# f22 = Function("p[2]","[0.0006125, 0.0006125]*cos([%f,%f])*[%f,%f]+[0.000765625, 0.000765625]*[%f,%f]-[0.000765625, 0.000765625]*cos([%f,%f])*sin([%f,%f])*[%f,%f]*[%f,%f]+[0.0600862, 0.0600863]*sin([%f,%f]+p[1]*[%f,%f])"%(q2lb,q2ub,ddq1lb,ddq1ub,ddq2lb,ddq2ub,q2lb,q2ub,q2lb,q2ub,dq1lb,dq1ub,dq1lb,dq1ub,q2lb,q2ub,q2lb,q2ub))
		
		f11 = Function("p[2]","[0.002, 0.003]*[%f,%f]+[0.0000, 0.0008]*sin([%f,%f])*sin([%f,%f])*[%f,%f]+[0.0005, 0.0007]*cos([%f,%f])*[%f,%f]-[0.0006, 0.0007]*sin([%f,%f])*[%f,%f]*[%f,%f]+2*[0.0007, 0.0008]*cos([%f,%f])*sin([%f,%f])*[%f,%f]*[%f,%f]-0.8*p[0]*[%f,%f]+p[1]*[%f,%f]"%(ddq1lb,ddq1ub,q2lb,q2ub,q2lb,q2ub,ddq1lb,ddq1ub,q2lb,q2ub,ddq2lb,ddq2ub,q2lb,q2ub,dq2lb,dq2ub,dq2lb,dq2ub,q2lb,q2ub,q2lb,q2ub,dq1lb,dq1ub,dq2lb,dq2ub,dq1lb,dq1ub,q2lb,q2ub))
		f22 = Function("p[2]","[0.0006125, 0.0006125]*cos([%f,%f])*[%f,%f]+[0.000765625, 0.000765625]*[%f,%f]-[0.000765625, 0.000765625]*cos([%f,%f])*sin([%f,%f])*[%f,%f]*[%f,%f]+[0.0600862, 0.0600863]*sin([%f,%f]+p[1]*[%f,%f])"%(q2lb,q2ub,ddq1lb,ddq1ub,ddq2lb,ddq2ub,q2lb,q2ub,q2lb,q2ub,dq1lb,dq1ub,dq1lb,dq1ub,q2lb,q2ub,q2lb,q2ub))
		

		sep1 = SepFwdBwd(f11,y)
		sep2 = SepFwdBwd(f22,Interval(-1.45,1.45))
		seps.append(sep1)

	
	# create the separator using the QIntersection
	sep = SepQInterProjF(seps)
	sep.q =1

	# init drawing area
	vibes.beginDrawing()
	vibes.newFigure('Result')
	vibes.setFigureProperties({'x': 0, 'y': 0, 'width': 500, 'height': 500})

	#configure pySIVIA output
	params = {'color_in': 'red[red]', 'color_out': 'green[green]', 'color_maybe': 'yellow[yellow]', 'use_patch' : True}

	# create the initial box X0 = [-10, 10] x [-10, 10]
	X0 = IntervalVector([[-1.2, 1.], [-1.2, 1.]])  # '#888888[#DDDDDD]'

	# run SIVIA 
	# (res_in, res_out, res_y) = pySIVIA(X0, sep, 0.1)
	pySIVIA(X0,sep,0.04,**params)

	# vibes.drawAUV(robot[0], robot[1], 1, np.rad2deg(0.3))
	# for (x, y), d in zip(landmarks, dist):
	#     vibes.drawCircle(x,y, 0.1, "[k]")
	#     vibes.drawCircle(x,y, d.lb(), "k")
	#     vibes.drawCircle(x,y, d.ub(), "k")

	#equalize axis lenght 
	vibes.axisEqual()


	vibes.endDrawing()
	# # q1lb=0.1
	# # q1ub=0.5
	# f1 = Function("p[7]","p[1]*[4,5]+p[0]*sin([%f,%f])*sin([4,5])*[4,5]-p[2]*sin([4,5])*[4,5]*[4,5]+2*p[0]*cos([4,5])*sin([4,5])*[4,5]*[4,5]"%(q1lb,q1lb))
	# f2 = Function("x[3]", "x[2]+x[0] + 2*x[1]*[1,2]*[1,2]*sin([4,5])")
	# print(f2)
	# vibes.beginDrawing()
	# vibes.newFigure('one Ring PyIbex')
	# vibes.setFigureProperties({'x':200,'y':100,'width':800,'height':800})
	# ctc = CtcFwdBwd(f2,sqr(r))
	# sep1 = SepFwdBwd(f2,sqr(r))
	# sep2 = SepFwdBwd(f2,sqr(r2))
	# sep = sep1&sep2 #intersection
	# sepU = sep1|sep2 #union
	# # ctc = myCtc()
	# pySIVIA(X0,sepU,0.5)
	# print("hellllllllllllllo",X0)
	# vibes.endDrawing()     
