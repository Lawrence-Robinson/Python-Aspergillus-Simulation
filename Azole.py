import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.style as style




def AzoleModel(x,t):
	HighErgosterol = x[0]
	LowErgosterol = x[1]
	Cyp51ac = x[2]
	Cyp51i = x[3]
	SrbAac = x[4]
	SrbAi = x[5]
	AtrRac = x[6]
	AtrRi = x[7]
	HapXac = x[8]
	HapXi = x[9]
	O2h = x[10]
	O2l = x[11]
	Feh = x[12]
	Fel = x[13]
	SrbAt = x[14]
	SrbAr = x[15]
	Efficiency = x[16]
	Azole = x[17]
	Cdr1Bt = x[18]
	Cdr1Br = x[19]

	#initialisation of parameters
	k1 = 1
	k2 = 1 
	k3 = 1
	k4 = 1
	k5 = 1
	k6 = 1
	d1 = 1
	d2 = 1
	d3 = 1
	d4 = 1
	d5 = 1
	d6 = 1

	# Difrential Equations

	Azoled = k1*1 - d1*Azole*Cdr1Bt

	Ergosterold = k1*LowErgosterol*Cyp51ac*(Feh+O2h) - d1*HighErgosterol*(Fel+O2l)*(Azole)
	LErgosterold = d1*HighErgosterol*(Fel+O2l)*(Azole)- k1*LowErgosterol*Cyp51ac*(Feh+O2h)

	Cyp51acd = k2*Cyp51i*(SrbAac+AtrRac) - d2*Cyp51ac*(HapXac)
	Cyp51id = d2*Cyp51ac*(HapXac) - k2*Cyp51i*(SrbAac+AtrRac)

	SrbAacd = k2*SrbAi*(SrbAt) - d2*SrbAac*HighErgosterol
	SrbAid = d2*SrbAac*HighErgosterol - k2*SrbAi*(SrbAt)

	SrbAad = k2*SrbAr*O2l - d2*SrbAt*Feh
	SrbArd = d2*SrbAt*Feh - k2*SrbAr*O2l

	AtrRacd = k3*AtrRi*Fel - d3*AtrRac*Feh*HighErgosterol
	AtrRid = d3*AtrRac*Feh*HighErgosterol - k3*AtrRi*Fel

	HapXacd = k4*HapXi*Fel - d4*HapXac*Feh*LowErgosterol
	HapXid = d4*HapXac*Feh*LowErgosterol - k4*HapXi*Fel

	O2hd = k5*(SrbAt)*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*(SrbAt)*O2l

	Fehd = k6*(HapXac)*Fel - d6*Feh*Cyp51ac
	Feld = d6*Feh*Cyp51ac -k6*(HapXac)*Fel

	Cdr1Btd =k3*Cdr1Br*AtrRac - d3*Cdr1Bt*O2l
	Cdr1Brd = d3*Cdr1Bt*O2l - k3*AtrRac*Cdr1Br

	Efficiency = (O2h+Feh)/200

	return[Ergosterold, LErgosterold , Cyp51acd, Cyp51id, SrbAacd, SrbAid, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld, SrbAad, SrbArd, Efficiency, Azoled, Cdr1Btd, Cdr1Brd]

def AzoleModel2(x,t):
	HighErgosterol = x[0]
	LowErgosterol = x[1]
	Cyp51ac = x[2]
	Cyp51i = x[3]
	SrbAac = x[4]
	SrbAi = x[5]
	AtrRac = x[6]
	AtrRi = x[7]
	HapXac = x[8]
	HapXi = x[9]
	O2h = x[10]
	O2l = x[11]
	Feh = x[12]
	Fel = x[13]
	SrbAt = x[14]
	SrbAr = x[15]
	Efficiency = x[16]
	Azole = x[17]
	Cdr1Bt = x[18]
	Cdr1Br = x[19]

	#initialisation of parameters
	k1 = 1
	k2 = 1 
	k3 = 1
	k4 = 1
	k5 = 1
	k6 = 1
	d1 = 1
	d2 = 1
	d3 = 1
	d4 = 1
	d5 = 10
	d6 = 10

	# Difrential Equations

	Azoled = k1*10 - d1*Azole*Cdr1Bt

	Ergosterold = k1*LowErgosterol*Cyp51ac*(Feh+O2h) - d1*HighErgosterol*(Fel+O2l)*(Azole)
	LErgosterold = d1*HighErgosterol*(Fel+O2l)*(Azole)- k1*LowErgosterol*Cyp51ac*(Feh+O2h)

	Cyp51acd = k2*Cyp51i*(SrbAac+AtrRac) - d2*Cyp51ac*(HapXac)
	Cyp51id = d2*Cyp51ac*(HapXac) - k2*Cyp51i*(SrbAac+AtrRac)

	SrbAacd = k2*SrbAi*(SrbAt) - d2*SrbAac*HighErgosterol
	SrbAid = d2*SrbAac*HighErgosterol - k2*SrbAi*(SrbAt)

	SrbAad = k2*SrbAr*O2l - d2*SrbAt*Feh
	SrbArd = d2*SrbAt*Feh - k2*SrbAr*O2l

	AtrRacd = k3*AtrRi*Fel - d3*AtrRac*Feh*HighErgosterol
	AtrRid = d3*AtrRac*Feh*HighErgosterol - k3*AtrRi*Fel

	HapXacd = k4*HapXi*Fel - d4*HapXac*Feh*LowErgosterol
	HapXid = d4*HapXac*Feh*LowErgosterol - k4*HapXi*Fel

	O2hd = k5*(SrbAt)*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*(SrbAt)*O2l

	Fehd = k6*(HapXac)*Fel - d6*Feh*Cyp51ac
	Feld = d6*Feh*Cyp51ac -k6*(HapXac)*Fel

	Cdr1Btd =k3*Cdr1Br*AtrRac - d3*Cdr1Bt*O2l
	Cdr1Brd = d3*Cdr1Bt*O2l - k3*AtrRac*Cdr1Br

	Efficiency = (O2h+Feh)/200

	return[Ergosterold, LErgosterold , Cyp51acd, Cyp51id, SrbAacd, SrbAid, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld, SrbAad, SrbArd, Efficiency, Azoled, Cdr1Btd, Cdr1Brd]

def AzoleModel3(x,t):
	HighErgosterol = x[0]
	LowErgosterol = x[1]
	Cyp51ac = x[2]
	Cyp51i = x[3]
	SrbAac = x[4]
	SrbAi = x[5]
	AtrRac = x[6]
	AtrRi = x[7]
	HapXac = x[8]
	HapXi = x[9]
	O2h = x[10]
	O2l = x[11]
	Feh = x[12]
	Fel = x[13]
	SrbAt = x[14]
	SrbAr = x[15]
	Efficiency = x[16]
	Azole = x[17]
	Cdr1Bt = x[18]
	Cdr1Br = x[19]

	#initialisation of parameters
	k1 = 1
	k2 = 1 
	k3 = 1
	k4 = 1
	k5 = 1
	k6 = 1
	d1 = 1
	d2 = 1
	d3 = 1
	d4 = 1
	d5 = 100
	d6 = 100

	# Difrential Equations

	Azoled = k1*100 - d1*Azole*Cdr1Bt

	Ergosterold = k1*LowErgosterol*Cyp51ac*(Feh+O2h) - d1*HighErgosterol*(Fel+O2l)*(Azole)
	LErgosterold = d1*HighErgosterol*(Fel+O2l)*(Azole)- k1*LowErgosterol*Cyp51ac*(Feh+O2h)

	Cyp51acd = k2*Cyp51i*(SrbAac+AtrRac) - d2*Cyp51ac*(HapXac)
	Cyp51id = d2*Cyp51ac*(HapXac) - k2*Cyp51i*(SrbAac+AtrRac)

	SrbAacd = k2*SrbAi*(SrbAt) - d2*SrbAac*HighErgosterol
	SrbAid = d2*SrbAac*HighErgosterol - k2*SrbAi*(SrbAt)

	SrbAad = k2*SrbAr*O2l - d2*SrbAt*Feh
	SrbArd = d2*SrbAt*Feh - k2*SrbAr*O2l

	AtrRacd = k3*AtrRi*Fel - d3*AtrRac*Feh*HighErgosterol
	AtrRid = d3*AtrRac*Feh*HighErgosterol - k3*AtrRi*Fel

	HapXacd = k4*HapXi*Fel - d4*HapXac*Feh*LowErgosterol
	HapXid = d4*HapXac*Feh*LowErgosterol - k4*HapXi*Fel

	O2hd = k5*(SrbAt)*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*(SrbAt)*O2l

	Fehd = k6*(HapXac)*Fel - d6*Feh*Cyp51ac
	Feld = d6*Feh*Cyp51ac -k6*(HapXac)*Fel

	Cdr1Btd =k3*Cdr1Br*AtrRac - d3*Cdr1Bt*O2l
	Cdr1Brd = d3*Cdr1Bt*O2l - k3*AtrRac*Cdr1Br

	Efficiency = (O2h+Feh)/200

	return[Ergosterold, LErgosterold , Cyp51acd, Cyp51id, SrbAacd, SrbAid, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld, SrbAad, SrbArd, Efficiency, Azoled, Cdr1Btd, Cdr1Brd]

def AzoleModel4(x,t):
	HighErgosterol = x[0]
	LowErgosterol = x[1]
	Cyp51ac = x[2]
	Cyp51i = x[3]
	SrbAac = x[4]
	SrbAi = x[5]
	AtrRac = x[6]
	AtrRi = x[7]
	HapXac = x[8]
	HapXi = x[9]
	O2h = x[10]
	O2l = x[11]
	Feh = x[12]
	Fel = x[13]
	SrbAt = x[14]
	SrbAr = x[15]
	Efficiency = x[16]
	Azole = x[17]
	Cdr1Bt = x[18]
	Cdr1Br = x[19]

	#initialisation of parameters
	k1 = 1
	k2 = 1 
	k3 = 1
	k4 = 1
	k5 = 1
	k6 = 1
	d1 = 1
	d2 = 1
	d3 = 1
	d4 = 1
	d5 = 200
	d6 = 200

	# Difrential Equations

	Azoled = k1*200 - d1*Azole*Cdr1Bt

	Ergosterold = k1*LowErgosterol*Cyp51ac*(Feh+O2h) - d1*HighErgosterol*(Fel+O2l)*(Azole)
	LErgosterold = d1*HighErgosterol*(Fel+O2l)*(Azole)- k1*LowErgosterol*Cyp51ac*(Feh+O2h)

	Cyp51acd = k2*Cyp51i*(SrbAac+AtrRac) - d2*Cyp51ac*(HapXac)
	Cyp51id = d2*Cyp51ac*(HapXac) - k2*Cyp51i*(SrbAac+AtrRac)

	SrbAacd = k2*SrbAi*(SrbAt) - d2*SrbAac*HighErgosterol
	SrbAid = d2*SrbAac*HighErgosterol - k2*SrbAi*(SrbAt)

	SrbAad = k2*SrbAr*O2l - d2*SrbAt*Feh
	SrbArd = d2*SrbAt*Feh - k2*SrbAr*O2l

	AtrRacd = k3*AtrRi*Fel - d3*AtrRac*Feh*HighErgosterol
	AtrRid = d3*AtrRac*Feh*HighErgosterol - k3*AtrRi*Fel

	HapXacd = k4*HapXi*Fel - d4*HapXac*Feh*LowErgosterol
	HapXid = d4*HapXac*Feh*LowErgosterol - k4*HapXi*Fel

	O2hd = k5*(SrbAt)*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*(SrbAt)*O2l

	Fehd = k6*(HapXac)*Fel - d6*Feh*Cyp51ac
	Feld = d6*Feh*Cyp51ac -k6*(HapXac)*Fel

	Cdr1Btd =k3*Cdr1Br*AtrRac - d3*Cdr1Bt*O2l
	Cdr1Brd = d3*Cdr1Bt*O2l - k3*AtrRac*Cdr1Br

	Efficiency = (O2h+Feh)/200

	return[Ergosterold, LErgosterold , Cyp51acd, Cyp51id, SrbAacd, SrbAid, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld, SrbAad, SrbArd, Efficiency, Azoled, Cdr1Btd, Cdr1Brd]

def AzoleModel5(x,t):
	HighErgosterol = x[0]
	LowErgosterol = x[1]
	Cyp51ac = x[2]
	Cyp51i = x[3]
	SrbAac = x[4]
	SrbAi = x[5]
	AtrRac = x[6]
	AtrRi = x[7]
	HapXac = x[8]
	HapXi = x[9]
	O2h = x[10]
	O2l = x[11]
	Feh = x[12]
	Fel = x[13]
	SrbAt = x[14]
	SrbAr = x[15]
	Efficiency = x[16]
	Azole = x[17]
	Cdr1Bt = x[18]
	Cdr1Br = x[19]

	#initialisation of parameters
	k1 = 1
	k2 = 1 
	k3 = 1
	k4 = 1
	k5 = 1
	k6 = 1
	d1 = 1
	d2 = 1
	d3 = 1
	d4 = 1
	d5 = 500
	d6 = 500

	# Difrential Equations

	Azoled = k1*500 - d1*Azole*Cdr1Bt

	Ergosterold = k1*LowErgosterol*Cyp51ac*(Feh+O2h) - d1*HighErgosterol*(Fel+O2l)*(Azole)
	LErgosterold = d1*HighErgosterol*(Fel+O2l)*(Azole)- k1*LowErgosterol*Cyp51ac*(Feh+O2h)

	Cyp51acd = k2*Cyp51i*(SrbAac+AtrRac) - d2*Cyp51ac*(HapXac)
	Cyp51id = d2*Cyp51ac*(HapXac) - k2*Cyp51i*(SrbAac+AtrRac)

	SrbAacd = k2*SrbAi*(SrbAt) - d2*SrbAac*HighErgosterol
	SrbAid = d2*SrbAac*HighErgosterol - k2*SrbAi*(SrbAt)

	SrbAad = k2*SrbAr*O2l - d2*SrbAt*Feh
	SrbArd = d2*SrbAt*Feh - k2*SrbAr*O2l

	AtrRacd = k3*AtrRi*Fel - d3*AtrRac*Feh*HighErgosterol
	AtrRid = d3*AtrRac*Feh*HighErgosterol - k3*AtrRi*Fel

	HapXacd = k4*HapXi*Fel - d4*HapXac*Feh*LowErgosterol
	HapXid = d4*HapXac*Feh*LowErgosterol - k4*HapXi*Fel

	O2hd = k5*(SrbAt)*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*(SrbAt)*O2l

	Fehd = k6*(HapXac)*Fel - d6*Feh*Cyp51ac
	Feld = d6*Feh*Cyp51ac -k6*(HapXac)*Fel

	Cdr1Btd =k3*Cdr1Br*AtrRac - d3*Cdr1Bt*O2l
	Cdr1Brd = d3*Cdr1Bt*O2l - k3*AtrRac*Cdr1Br

	Efficiency = (O2h+Feh)/200

	return[Ergosterold, LErgosterold , Cyp51acd, Cyp51id, SrbAacd, SrbAid, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld, SrbAad, SrbArd, Efficiency, Azoled, Cdr1Btd, Cdr1Brd]

def AzoleModel6(x,t):
	HighErgosterol = x[0]
	LowErgosterol = x[1]
	Cyp51ac = x[2]
	Cyp51i = x[3]
	SrbAac = x[4]
	SrbAi = x[5]
	AtrRac = x[6]
	AtrRi = x[7]
	HapXac = x[8]
	HapXi = x[9]
	O2h = x[10]
	O2l = x[11]
	Feh = x[12]
	Fel = x[13]
	SrbAt = x[14]
	SrbAr = x[15]
	Efficiency = x[16]
	Azole = x[17]
	Cdr1Bt = x[18]
	Cdr1Br = x[19]

	#initialisation of parameters
	k1 = 1
	k2 = 1 
	k3 = 1
	k4 = 1
	k5 = 1
	k6 = 1
	d1 = 1
	d2 = 1
	d3 = 1
	d4 = 1
	d5 = 1000
	d6 = 1000

	# Difrential Equations

	Azoled = k1*1000 - d1*Azole*Cdr1Bt

	Ergosterold = k1*LowErgosterol*Cyp51ac*(Feh+O2h) - d1*HighErgosterol*(Fel+O2l)*(Azole)
	LErgosterold = d1*HighErgosterol*(Fel+O2l)*(Azole)- k1*LowErgosterol*Cyp51ac*(Feh+O2h)

	Cyp51acd = k2*Cyp51i*(SrbAac+AtrRac) - d2*Cyp51ac*(HapXac)
	Cyp51id = d2*Cyp51ac*(HapXac) - k2*Cyp51i*(SrbAac+AtrRac)

	SrbAacd = k2*SrbAi*(SrbAt) - d2*SrbAac*HighErgosterol
	SrbAid = d2*SrbAac*HighErgosterol - k2*SrbAi*(SrbAt)

	SrbAad = k2*SrbAr*O2l - d2*SrbAt*Feh
	SrbArd = d2*SrbAt*Feh - k2*SrbAr*O2l

	AtrRacd = k3*AtrRi*Fel - d3*AtrRac*Feh*HighErgosterol
	AtrRid = d3*AtrRac*Feh*HighErgosterol - k3*AtrRi*Fel

	HapXacd = k4*HapXi*Fel - d4*HapXac*Feh*LowErgosterol
	HapXid = d4*HapXac*Feh*LowErgosterol - k4*HapXi*Fel

	O2hd = k5*(SrbAt)*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*(SrbAt)*O2l

	Fehd = k6*(HapXac)*Fel - d6*Feh*Cyp51ac
	Feld = d6*Feh*Cyp51ac -k6*(HapXac)*Fel

	Cdr1Btd =k3*Cdr1Br*AtrRac - d3*Cdr1Bt*O2l
	Cdr1Brd = d3*Cdr1Bt*O2l - k3*AtrRac*Cdr1Br

	Efficiency = (O2h+Feh)/200

	return[Ergosterold, LErgosterold , Cyp51acd, Cyp51id, SrbAacd, SrbAid, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld, SrbAad, SrbArd, Efficiency, Azoled, Cdr1Btd, Cdr1Brd]





plt.subplot(3,3,1)
t = np.linspace(0, 0.19, 100)
x_init = [100,20,200,200,100,100,100,100,100,100,100,100,100,100,100,100,100,200,100,100]
#x_init = [100,1,200,200,100,100,60,60,200,200,100,1,100,1,100,100,0,100,60,60]
x1 = odeint(AzoleModel, x_init, t)
plt.title('A', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t, x1[:,[2,4,6,8,18]], '-', linewidth = 2)
plt.plot(t, x1[:,[17]], 'k--', linewidth=2 )
plt.plot(t, x1[:,[0]], 'r--', linewidth=2 )

plt.subplot(3,3,2)
t = np.linspace(0, 0.19, 100)
x_init = [100,20,200,200,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
#x_init = [100,1,200,200,100,100,60,60,200,200,100,1,100,1,100,100,0,100,60,60]
x1 = odeint(AzoleModel2, x_init, t)
plt.title('B', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t, x1[:,[2,4,6,8,18]], '-', linewidth = 2)
plt.plot(t, x1[:,[17]], 'k--', linewidth=2 )
plt.plot(t, x1[:,[0]], 'r--', linewidth=2 )

plt.subplot(3,3,3)
t = np.linspace(0, 0.19, 100)
x_init = [100,20,200,200,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
#x_init = [100,1,200,200,100,100,60,60,200,200,100,1,100,1,100,100,0,100,60,60]
x1 = odeint(AzoleModel3, x_init, t)
plt.title('C', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t, x1[:,[2,4,6,8,18]], '-', linewidth = 2)
plt.plot(t, x1[:,[17]], 'k--', linewidth=2 )
plt.plot(t, x1[:,[0]], 'r--', linewidth=2 )

plt.subplot(3,3,4)
t = np.linspace(0, 0.19, 100)
x_init = [100,20,200,200,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
#x_init = [100,1,200,200,100,100,60,60,200,200,100,1,100,1,100,100,0,100,60,60]
x1 = odeint(AzoleModel4, x_init, t)
plt.title('D', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t, x1[:,[2,4,6,8,18]], '-', linewidth = 2)
plt.plot(t, x1[:,[17]], 'k--', linewidth=2 )
plt.plot(t, x1[:,[0]], 'r--', linewidth=2 )

plt.subplot(3,3,5)
t = np.linspace(0, 0.19, 100)
x_init = [100,20,200,200,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
#x_init = [100,1,200,200,100,100,60,60,200,200,100,1,100,1,100,100,0,100,60,60]
x1 = odeint(AzoleModel5, x_init, t)
plt.title('E', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t, x1[:,[2,4,6,8,18]], '-', linewidth = 2)
plt.plot(t, x1[:,[17]], 'k--', linewidth=2 )
plt.plot(t, x1[:,[0]], 'r--', linewidth=2 )


plt.subplot(3,3,6)
t = np.linspace(0, 0.19, 100)
x_init = [100,20,200,200,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
#x_init = [100,1,200,200,100,100,60,60,200,200,100,1,100,1,100,100,0,100,60,60]
x1 = odeint(AzoleModel6, x_init, t)
plt.title('F', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t, x1[:,[2,4,6,8,18]], '-', linewidth = 2)
plt.plot(t, x1[:,[17]], 'k--', linewidth=2 )
plt.plot(t, x1[:,[0]], 'r--', linewidth=2 )
plt.legend(['Cyp51', 'SrbAac', 'AtrR', 'HapX', 'Cdr1B','Azole','Ergosterol' ],loc='upper center',prop={'size':12} ,bbox_to_anchor=(-0.7, -0.21),ncol=7, fancybox=False)

plt.show()



'''
def ErgModel(x,t):
	HighErgosterol = x[0]
	LowErgosterol = x[1]
	Cyp51ac = x[2]
	Cyp51i = x[3]
	SrbAac = x[4]
	SrbAi = x[5]
	AtrRac = x[6]
	AtrRi = x[7]
	HapXac = x[8]
	HapXi = x[9]
	O2h = x[10]
	O2l = x[11]
	Feh = x[12]
	Fel = x[13]
	SrbAt = x[14]
	SrbAr = x[15]
	Efficiency = x[16]

	#initialisation of parameters
	k1 = 1
	k2 = 1 
	k3 = 1
	k4 = 1
	k5 = 1
	k6 = 1
	d1 = 1
	d2 = 1
	d3 = 1
	d4 = 1
	d5 = 1
	d6 = 1

	# Difrential Equations

	Ergosterold = k1*Cyp51ac - d1*HighErgos(ter+O2h)ol
	LErgosterold =1

(+O2l)p51acd = k2*S(rbA*O2l)ac*AtrR- d2*Cyp51ac*(HapXac*2)
	Cyp(51i+O2h)d = 1

	SrbAacd = k2*SrbAt - d2*SrbAac*HighErgosterol
	SrbAid = 1

	SrbAad = k1*SrbAt - d1*SrbAt*O2h*Feh
	SrbArd = 1

	AtrRacd = k3*AtrRi - d3*AtrRac*(Feh*HighErgosterol)/100
	AtrRid = 1

	HapXacd = k4*HapXac - d4*HapXac*Feh
	HapXid = 1

	O2hd = k5*SrbAt - d5*O2h*Cyp51ac/2
	O2ld = 1

	Fehd = k6*HapXac - d6*Feh*Cyp51ac/2
	Feld = 1

	Efficiency = (O2h+Feh)/200

	return[Ergosterold, LErgosterold , Cyp51acd, Cyp51id, SrbAacd, SrbAid, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld, SrbAad, SrbArd,Efficiency]


plt.subplot(2,1,1)
t = np.linspace(0, 1, 100)
x_init=[100,0,100,100,100,100,100,100,100,100,100,1,100,1,100,100,0]
x1 = odeint(ErgModel, x_init, t)
plt.title('A', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t, x1[:,[0,2,4,6]], '-', linewidth = 2)
plt.legend(['Ergosterol', 'Cyp51ac', 'SrbAac','AtrRac'  ])

plt.subplot(2,1,2)
t = np.linspace(0, 1, 100)
x_init=[100,0,100,100,100,100,100,100,100,100,100,1,100,1,100,100,0]
x1 = odeint(ErgModel, x_init, t)
plt.title('B', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t, x1[:,[0,8,12]], '-', linewidth = 2)
plt.legend(['Ergosterol', 'HapX', 'Fe',])
plt.show()

t = np.linspace(0, 1, 100)
x_init = [100,10,100,100,100,100,30,30,30,30,100,10,100,10,100,100,0]
x1 = odeint(ErgModel, x_init, t)
plt.title('A', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t, x1[:,[0,2,4,6,8,10,12]], '-', linewidth = 2)
plt.legend(['Ergosterol', 'Cyp51ac', 'SrbAac', 'AtrRac', 'HapXac','O2', 'Fe' ])
#plt.plot(t, x1[:,[]], '--', linewidth = 2)
plt.show()




#1AM LAST NIGHT VERSION CONTROL, before your GRAND IDEA OF GRANDNESS
def ErgModel(x,t):
	HighErgosterol = x[0]
	LowErgosterol = x[1]
	Cyp51ac = x[2]
	Cyp51i = x[3]
	SrbAac = x[4]
	SrbAi = x[5]
	AtrRac = x[6]
	AtrRi = x[7]
	HapXac = x[8]
	HapXi = x[9]
	O2h = x[10]
	O2l = x[11]
	Feh = x[12]
	Fel = x[13]
	SrbAt = x[14]
	SrbAr = x[15]
	Efficiency = x[16]

	#initialisation of parameters
	k1 = 1
	k2 = 1 
	k3 = 1
	k4 = 1
	k5 = 1
	k6 = 1
	d1 = 1
	d2 = 1
	d3 = 1
	d4 = 1
	d5 = 1
	d6 = 1

	# Difrential Equations

	Ergosterold = k1*Cyp51ac - d1*HighErgos(ter+O2h)ol*Efficiency
	LErgos(ter+O2l)old (= 2l)1*HighErgosterol*Effincy - k1*Cyp51(ac
+O2h)
	Cyp51acd = k2*(SrbAac*AtrRac) - d2*Cyp51ac*(HapXac*10)
	Cyp51id = d2*Cyp51ac*(HapXac*10) - k2*(SrbAac*AtrRac)

	SrbAacd = k2*(SrbAt)*LowErgosterol - d2*SrbAac*(HighErgosterol)
	SrbAid = SrbAt/10

	SrbAad = k2*SrbAr*O2l - d2*SrbAt*O2h*Feh
	SrbArd = d2*SrbAt*O2h*Feh - k2*SrbAr*O2l

	AtrRacd = k3*AtrRi*LowErgosterol - d3*AtrRac*HighErgosterol
	AtrRid = d3*AtrRac*HighErgosterol - k3*(AtrRi)*LowErgosterol

	HapXacd = k4*HapXi - d4*HapXac*Feh
	HapXid = d4*HapXac*Feh - k4*HapXi

	O2hd = k5*(SrbAt)*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*(SrbAt)*O2l

	Fehd = k6*(HapXac)*Fel - d6*Feh*Cyp51ac/2
	Feld = d6*Feh*Cyp51ac/2 -k6*(HapXac)*Fel

	Efficiency = (O2h+Feh)/100

	
	

	return[Ergosterold, LErgosterold , Cyp51acd, Cyp51id, SrbAacd, SrbAid, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld, SrbAad, SrbArd,Efficiency]


plt.subplot(2,1,1)
t = np.linspace(0, 1, 100)
x_init=[10100,100,100,100,100,100,100,100,100,100,100,1,100,1,100,10100,100]
x1 = odeint(ErgModel, x_init, t)
plt.title('A', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t, x1[:,[0,2,4,6,8]], '-', linewidth = 2)
plt.legend(['Ergosterol', 'Cyp51ac', 'SrbAac','AtrRac', 'HapXac'  ])

plt.subplot(2,1,2)
t = np.linspace(0, 1, 100)
x_init=[10100,100,100,100,100,100,100,100,100,100,100,1,100,1,100,10100,100]
x1 = odeint(ErgModel, x_init, t)
plt.title('B', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t, x1[:,[0,8,12]], '-', linewidth = 2)
plt.legend(['Ergosterol', 'HapX', 'Fe',])
plt.show()

t = np.linspace(0, 10, 100)
x_init = [100,1,100,1,100,1,100,1,100,1,100,1,100,1,100,1,0]
x1 = odeint(ErgModel, x_init, t)
plt.title('A', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t, x1[:,[0,2,4,6,8,10,12]], '-', linewidth = 2)
plt.legend(['Ergosterol', 'Cyp51ac', 'SrbAac', 'AtrRac', 'HapXac','O2', 'Fe' ])
#plt.plot(t, x1[:,[]], '--', linewidth = 2)
plt.show()



'''

