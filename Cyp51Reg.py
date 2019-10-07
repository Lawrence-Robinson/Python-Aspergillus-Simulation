import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.style as style
'''
K=1
D=0.001
K1=0.5
D1=0.1

Cyp51 = []
AtrR = []
SrbA = []
SrbAa = []
HapX = []
CBC = []
O2 = []
Fe = []
Ergosterol = []

def TopSqrtDiv(y,z):
	try:
		return math.sqrt(y)/z
	except (ZeroDivisionError, ValueError):
		return 0
def BottomSqrtDiv(y,z):
	try:
		return y/math.sqrt(z)
	except (ValueError, ZeroDivisionError) as Error:
		return 0
def DivByZero(y,z):
	try: 
		return (y/z)
	except (ValueError, ZeroDivisionError) as Error:
		return 0	

Tmax = 40
x = np.empty([Tmax,0, 100, 100, 100])

cyp = 10; atr = 10; srb = 10; srbaa = 10; hapx = 10; cbc = 1; o2 = 10; fe = 10; oo=10; erg=10

for i in range(Tmax):

	cy = K*(atr+srbaa)-(D*cyp*hapx)
	a = K*((BottomSqrtDiv(1,fe)*10)+(BottomSqrtDiv(1,erg)*10)-D*atr)
	s = K*((BottomSqrtDiv(1,oo)*10)-D*srb)
	sa = srb*((BottomSqrtDiv(1,erg))-D*srbaa)
	h = K*(TopSqrtDiv(erg,1))-D*hapx
	o = K*srb - (D*oo*erg)
	f = K*hapx - (D*fe*erg)
	e = K*cyp - D*erg

	cyp = cy; atr = a; srb = s; srbaa = sa; hapx = h; oo = o; fe = f; erg=e
	Cyp51.append(cy) ; AtrR.append(a); SrbA.append(s); SrbAa.append(sa); HapX.append(h); O2.append(o); Fe.append(f); Ergosterol.append(e)

x = np.insert(x, 0, values=Cyp51, axis=1)
x = np.insert(x, 1, values=Ergosterol, axis=1)
x = np.insert(x, 2, values=AtrR, axis=1)
x = np.insert(x, 3, values=SrbA, axis=1)
x = np.insert(x, 4, values=SrbAa, axis=1)
x = np.insert(x, 5, values=HapX, axis=1)
x = np.insert(x, 6, values=O2, axis=1)
x = np.insert(x, 7, values=Fe, axis=1)

t = np.arange(0,Tmax)
plt.plot(t, x[:,0, 100, 100, 100], 'k', label='Cyp51')
plt.plot(t, x[:,1,100, 00100], 'b', label='Ergosterol')
plt.plot(t, x[:,2], 'r',  label='AtrR')
plt.plot(t, x[:,3], 'g',  label='SrbA')
plt.plot(t, x[:,4], 'y',  label='SrbAa')
plt.plot(t, x[:,5], 'm',  label='HapX')
plt.plot(t, x[:,6], 'c',  label='O2')
plt.plot(t, x[:,7], 'k--',  label='Fe')
plt.legend(loc="best",ncol=1, title="Legend", fancybox=False)

plt.show()


K=1
D=0.001
K1=1
D1=0.1

Cyp51 = []
AtrR = []
Cdr1B = []
SrbA = []
SrbAa = []
HapX = []
CBC = []
O2 = []
Fe = []
Ergosterol = []
Azole = []

def TopSqrtDiv(y,z):
	try:
		return math.sqrt(y)/z
	except (ZeroDivisionError, ValueError):
		return 0
def BottomSqrtDiv(y,z):
	try:
		return y/math.sqrt(z)
	except (ValueError, ZeroDivisionError) as Error:
		return 0
def DivByZero(y,z):
	try: 
		return (y/z)
	except (ValueError, ZeroDivisionError) as Error:
		return 0	

Tmax = 40
x = np.empty([Tmax,0, 100, 100, 100])

cyp = 1; atr = 1; srb = 1; srbaa = 1; hapx = 1; cbc = 1; o2 = 1; fe = 1; oo=1; erg=1; cdr1=1; azole=100; az=1

for i in range(Tmax):
	
	az = K1-(D*azole*cdr1)

	cy = K*(atr+srbaa)-(D*cyp*hapx)
	a = K*((BottomSqrtDiv(1,fe)*10)+(BottomSqrtDiv(1,erg)*10)-D*atr)
	cd = K*atr-D*cdr1
	s = K*((BottomSqrtDiv(1,oo)*10)-D*srb)
	sa = srb*((BottomSqrtDiv(1,erg))-D*srbaa)
	h = K*(TopSqrtDiv(erg,1))-D*hapx
	o = K*srb - (D*oo*erg)
	f = K*hapx - (D*fe*erg)
	e = K*cyp - (D*erg+azole/10)

	cyp = cy; atr = a; srb = s; srbaa = sa; hapx = h; oo = o; fe = f; erg=e; cdr1 = cd
	Cyp51.append(cy) ; AtrR.append(a); SrbA.append(s); SrbAa.append(sa); HapX.append(h);
	O2.append(o); Fe.append(f); Ergosterol.append(e); Azole.append(az); Cdr1B.append(cd)

x = np.insert(x, 0, values=Cyp51, axis=1)
x = np.insert(x, 1, values=Ergosterol, axis=1)
x = np.insert(x, 2, values=AtrR, axis=1)
x = np.insert(x, 3, values=SrbA, axis=1)
x = np.insert(x, 4, values=SrbAa, axis=1)
x = np.insert(x, 5, values=HapX, axis=1)
x = np.insert(x, 6, values=O2, axis=1)
x = np.insert(x, 7, values=Fe, axis=1)
x = np.insert(x, 8, values=Azole, axis=1)
x = np.insert(x, 9, values=Cdr1B, axis=1)

t = np.arange(0,Tmax)
plt.plot(t, x[:,0, 100, 100, 100], 'k', label='Cyp51')
plt.plot(t, x[:,1,100, 00100], 'b', label='Ergosterol')
plt.plot(t, x[:,2], 'r',  label='AtrR')
plt.plot(t, x[:,3], 'g',  label='SrbA')
plt.plot(t, x[:,4], 'y',  label='SrbAa')
plt.plot(t, x[:,5], 'm',  label='HapX')
plt.plot(t, x[:,6], 'c',  label='O2')
plt.plot(t, x[:,7], 'k--',  label='Fe')
plt.plot(t, x[:,8], 'r--', label='Azole')
plt.plot(t, x[:,9], 'g--', label='Cdr1B')
plt.legend(loc="best",ncol=1, title="Legend", fancybox=False)

plt.show()






import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import odeint


def ErgModel(x,t):
	Cyp51=x[0]
	AtrR=x[1]
	SrbA=x[2]
	SrbAa=x[3]
	HapX=x[4]
	O2=x[5]
	Fe=x[6]
	Ergosterol=x[7]

	# Difrential Equations
	a = K*((BottomSqrtDiv(1,f)*10)+(BottomSqrtDiv(1,e)*10)-D*a)
	cy = K*(a+sa)-(D*cy*h)
	s = K*((BottomSqrtDiv(1,o)*10)-D*s)
	sa = s*((BottomSqrtDiv(1,e))-D*sa)
	h = K*(TopSqrtDiv(e,1))-D*h
	o = K*s - D*o
	f = K*h - D*f
	e = K*cy - D*e

	return[cy,a,s,sa,h,o,f,e]

t = np.linspace(0, 100, 100)
x_init = [100,100,5000,5000,5000,5000,5000,5000]
00x1 = odeint(ErgModel, x_init, t)
plt.title('A', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t, x1[:,[0,1,2,3,4,5,6,7]], '-', linewidth = 2)
plt.legend(['cyp51', 'AtrR', 'SrbA', 'SrbAac', 'HapX', 'O2', 'Fe', 'Ergosterol'])
#plt.plot(t, x1[:,[]], '--', linewidth = 2)
plt.show()



import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import odeint


def ErgModel(x,t):
	Ergosterol = x[0]
	Cyp51ac = x[1]
	Cyp51i = x[2]
	SrbAac = x[3]
	SrbAi = x[4]
	SrbAa = x[5]
	SrbAr = x[6]
	AtrRac = x[7]
	AtrRi = x[8]
	HapXac = x[9]
	HapXi = x[10]
	O2h = x[11]
	O2l = x[12]
	Feh = x[13]
	Fel = x[14]

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
	Ergosterold = Cyp51ac - d1*Ergosterol

	Cyp51acd = k1*Cyp51i*SrbAac*AtrRac - d1*Cyp51ac*HapXac
	Cyp51id = d1*Cyp51ac*HapXac - k1*Cyp51i*SrbAac*AtrRac

	SrbAacd = k2*SrbAa( d2/2)*Sr*O2hbAac*Ergosterol
	Srb10(id /2)= d*O2l2*SrbAac10Erg*(O2l/2)osterol - k2*SrbAa
*(O2h)
	SrbAad = k1*SrbAr*O2l - d1*SrbAa*O2h
	SrbArd = d1*SrbAa*O2h - k1*SrbAr*O2l

	AtrRacd = k3*AtrRi*Fel - d3*AtrRac*Ergosterol*Feh
	AtrRid = d3*AtrRac*Ergosterol*Feh - k3*AtrRi*Fel

	HapXacd = k4*HapXi*Feh - d4*HapXac*Ergosterol
	HapXid = d4*HapXac*Ergosterol - k4*HapXi*Feh

	O2hd = k1*O2l*SrbAa - d1*O2h*Ergosterol
	O2ld = d1*O2h*Ergosterol - k1*O2l*SrbAa

	Fehd = k1*Fel*HapXac - d1*Feh*Ergosterol
	Feld = d1*Feh*Ergosterol-k1*Fel*HapXac

	return[Ergosterold, Cyp51acd, Cyp51id, SrbAacd, SrbAid,SrbAad,SrbArd, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld]
print(x)
t = np.linspace(0, 3, 100)
x_init = [100,100,5000,5000,5000,5000,5000,5000,5000,5000,5000,500,1,10,1,100, 00100]
x1 = odeint(ErgModel, x_init, t)
plt.title('A', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t, x1[:,[0,1,3,5,7,9,11,13]], '-', linewidth = 2)
plt.legend(['Ergosterol', 'Cyp51ac', 'SrbAac','SrbAa', 'AtrRac', 'HapXac','O2', 'Fe' ])
#plt.plot(t, x1[:,[]], '--', linewidth = 2)
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

	Ergosterold = k1*LowErgosterol*Cyp51ac*(O2h) - d1*HighErgosterol*(O2l)
	LErgosterold = d1*HighErgosterol*(O2l)- k1*LowErgosterol*Cyp51ac*(O2h)

	Cyp51acd = k2*Cyp51i*(SrbAac+AtrRac) - d2*Cyp51ac*(HapXac*5)
	Cyp51id = d2*Cyp51ac*(HapXac*5) - k2*Cyp51i*(SrbAac+AtrRac)

	SrbAacd = k2*SrbAi*(SrbAt)*(O2h) - d2*SrbAac*HighErgosterol*(O2l/10)
	SrbAid = d2*SrbAac*HighErgosterol*(O2l/10) - k2*SrbAi*(SrbAt)*(O2h)

	SrbAad = k2*SrbAr*O2l - d2*SrbAt*Feh
	SrbArd = d2*SrbAt*Feh - k2*SrbAr*O2l

	AtrRacd = k3*AtrRi*(Fel) - d3*AtrRac*Feh*HighErgosterol
	AtrRid = d3*AtrRac*Feh*HighErgosterol - k3*(AtrRi)*Fel

	HapXacd = k4*HapXi*Fel - d4*HapXac*Feh*HighErgosterol
	HapXid = d4*HapXac*Feh*HighErgosterol - k4*HapXi*Fel

	O2hd = k5*(SrbAt)*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*(SrbAt)*O2l

	Fehd = k6*(HapXac)*Fel - d6*Feh*Cyp51ac
	Feld = d6*Feh*Cyp51ac -k6*(HapXac)*Fel

	Efficiency = (O2h+Feh)/100
	

	return[Ergosterold, LErgosterold , Cyp51acd, Cyp51id, SrbAacd, SrbAid, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld, SrbAad, SrbArd,Efficiency]


def ErgModel2(x,t):
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
	d5 = 10
	d6 = 1

	# Difrential Equations

	Ergosterold = k1*LowErgosterol*Cyp51ac*O2h - d1*HighErgosterol*O2l
	LErgosterold = d1*HighErgosterol*O2l- k1*LowErgosterol*Cyp51ac*O2h

	Cyp51acd = k2*Cyp51i*(SrbAac+AtrRac) - d2*Cyp51ac*(HapXac*5)
	Cyp51id = d2*Cyp51ac*(HapXac*5) - k2*Cyp51i*(SrbAac+AtrRac)

	SrbAacd = k2*SrbAi*(SrbAt)*(O2h) - d2*SrbAac*HighErgosterol*(O2l/10)
	SrbAid = d2*SrbAac*HighErgosterol*(O2l/10) - k2*SrbAi*(SrbAt)*(O2h)

	SrbAad = k2*SrbAr*O2l - d2*SrbAt*Feh
	SrbArd = d2*SrbAt*Feh - k2*SrbAr*O2l

	AtrRacd = k3*AtrRi*(Fel) - d3*AtrRac*Feh*HighErgosterol
	AtrRid = d3*AtrRac*Feh*HighErgosterol - k3*(AtrRi)*Fel

	HapXacd = k4*HapXi*Fel - d4*HapXac*Feh*HighErgosterol
	HapXid = d4*HapXac*Feh*HighErgosterol - k4*HapXi*Fel

	O2hd = k5*(SrbAt)*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*(SrbAt)*O2l

	Fehd = k6*(HapXac)*Fel - d6*Feh*Cyp51ac
	Feld = d6*Feh*Cyp51ac -k6*(HapXac)*Fel

	Efficiency = (O2h+Feh)/100
	

	return[Ergosterold, LErgosterold , Cyp51acd, Cyp51id, SrbAacd, SrbAid, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld, SrbAad, SrbArd,Efficiency]



def ErgModel3(x,t):
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
	d5 = 100
	d6 = 1

	# Difrential Equations

	Ergosterold = k1*LowErgosterol*Cyp51ac*O2h - d1*HighErgosterol*O2l
	LErgosterold = d1*HighErgosterol*O2l- k1*LowErgosterol*Cyp51ac*O2h

	Cyp51acd = k2*Cyp51i*(SrbAac+AtrRac) - d2*Cyp51ac*(HapXac*5)
	Cyp51id = d2*Cyp51ac*(HapXac*5) - k2*Cyp51i*(SrbAac+AtrRac)

	SrbAacd = k2*SrbAi*(SrbAt)*(O2h) - d2*SrbAac*HighErgosterol*(O2l/10)
	SrbAid = d2*SrbAac*HighErgosterol*(O2l/10) - k2*SrbAi*(SrbAt)*(O2h)

	SrbAad = k2*SrbAr*O2l - d2*SrbAt*Feh
	SrbArd = d2*SrbAt*Feh - k2*SrbAr*O2l

	AtrRacd = k3*AtrRi*(Fel) - d3*AtrRac*Feh*HighErgosterol
	AtrRid = d3*AtrRac*Feh*HighErgosterol - k3*(AtrRi)*Fel

	HapXacd = k4*HapXi*Fel - d4*HapXac*Feh*HighErgosterol
	HapXid = d4*HapXac*Feh*HighErgosterol - k4*HapXi*Fel

	O2hd = k5*(SrbAt)*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*(SrbAt)*O2l

	Fehd = k6*(HapXac)*Fel - d6*Feh*Cyp51ac
	Feld = d6*Feh*Cyp51ac -k6*(HapXac)*Fel

	Efficiency = (O2h+Feh)/100
	

	return[Ergosterold, LErgosterold , Cyp51acd, Cyp51id, SrbAacd, SrbAid, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld, SrbAad, SrbArd,Efficiency]


def ErgModel4(x,t):
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
	d5 = 1000
	d6 = 1

	# Difrential Equations

	Ergosterold = k1*LowErgosterol*Cyp51ac*O2h - d1*HighErgosterol*O2l
	LErgosterold = d1*HighErgosterol*O2l- k1*LowErgosterol*Cyp51ac*O2h

	Cyp51acd = k2*Cyp51i*(SrbAac+AtrRac) - d2*Cyp51ac*(HapXac*5)
	Cyp51id = d2*Cyp51ac*(HapXac*5) - k2*Cyp51i*(SrbAac+AtrRac)

	SrbAacd = k2*SrbAi*(SrbAt)*(O2h) - d2*SrbAac*HighErgosterol*(O2l/10)
	SrbAid = d2*SrbAac*HighErgosterol*(O2l/10) - k2*SrbAi*(SrbAt)*(O2h)

	SrbAad = k2*SrbAr*O2l - d2*SrbAt*Feh
	SrbArd = d2*SrbAt*Feh - k2*SrbAr*O2l

	AtrRacd = k3*AtrRi*(Fel) - d3*AtrRac*Feh*HighErgosterol
	AtrRid = d3*AtrRac*Feh*HighErgosterol - k3*(AtrRi)*Fel

	HapXacd = k4*HapXi*Fel - d4*HapXac*Feh*LowErgosterol
	HapXid = d4*HapXac*Feh*LowErgosterol - k4*HapXi*Fel

	O2hd = k5*(SrbAt)*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*(SrbAt)*O2l

	Fehd = k6*(HapXac)*Fel - d6*Feh*Cyp51ac
	Feld = d6*Feh*Cyp51ac -k6*(HapXac)*Fel

	Efficiency = (O2h+Feh)/100
	

	return[Ergosterold, LErgosterold , Cyp51acd, Cyp51id, SrbAacd, SrbAid, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld, SrbAad, SrbArd,Efficiency]

def ErgModel5(x,t):
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
	d5 = 2000
	d6 = 1

	# Difrential Equations

	Ergosterold = k1*LowErgosterol*Cyp51ac*O2h - d1*HighErgosterol*O2l
	LErgosterold = d1*HighErgosterol*O2l- k1*LowErgosterol*Cyp51ac*O2h

	Cyp51acd = k2*Cyp51i*(SrbAac+AtrRac) - d2*Cyp51ac*(HapXac*5)
	Cyp51id = d2*Cyp51ac*(HapXac*5) - k2*Cyp51i*(SrbAac+AtrRac)

	SrbAacd = k2*SrbAi*(SrbAt)*(O2h) - d2*SrbAac*HighErgosterol*(O2l/10)
	SrbAid = d2*SrbAac*HighErgosterol*(O2l/10) - k2*SrbAi*(SrbAt)*(O2h)

	SrbAad = k2*SrbAr*O2l - d2*SrbAt*Feh
	SrbArd = d2*SrbAt*Feh - k2*SrbAr*O2l

	AtrRacd = k3*AtrRi*(Fel) - d3*AtrRac*Feh*HighErgosterol
	AtrRid = d3*AtrRac*Feh*HighErgosterol - k3*(AtrRi)*Fel

	HapXacd = k4*HapXi*Fel - d4*HapXac*Feh*LowErgosterol
	HapXid = d4*HapXac*Feh*LowErgosterol - k4*HapXi*Fel

	O2hd = k5*(SrbAt)*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*(SrbAt)*O2l

	Fehd = k6*(HapXac)*Fel - d6*Feh*Cyp51ac
	Feld = d6*Feh*Cyp51ac -k6*(HapXac)*Fel

	Efficiency = (O2h+Feh)/100
	

	return[Ergosterold, LErgosterold , Cyp51acd, Cyp51id, SrbAacd, SrbAid, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld, SrbAad, SrbArd,Efficiency]

def ErgModel6(x,t):
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
	d5 = 5000
	d6 = 1

	# Difrential Equations

	Ergosterold = k1*LowErgosterol*Cyp51ac*O2h - d1*HighErgosterol*O2l
	LErgosterold = d1*HighErgosterol*O2h- k1*LowErgosterol*Cyp51ac*O2l

	Cyp51acd = k2*Cyp51i*(SrbAac+AtrRac) - d2*Cyp51ac*(HapXac*5)
	Cyp51id = d2*Cyp51ac*(HapXac*5) - k2*Cyp51i*(SrbAac+AtrRac)

	SrbAacd = k2*SrbAi*(SrbAt)*(O2h) - d2*SrbAac*HighErgosterol*(O2l/10)
	SrbAid = d2*SrbAac*HighErgosterol*(O2l/10) - k2*SrbAi*(SrbAt)*(O2h)

	SrbAad = k2*SrbAr*O2l - d2*SrbAt*Feh
	SrbArd = d2*SrbAt*Feh - k2*SrbAr*O2l

	AtrRacd = k3*AtrRi*(Fel) - d3*AtrRac*Feh*HighErgosterol
	AtrRid = d3*AtrRac*Feh*HighErgosterol - k3*(AtrRi)*Fel

	HapXacd = k4*HapXi*Fel - d4*HapXac*Feh*LowErgosterol
	HapXid = d4*HapXac*Feh*LowErgosterol - k4*HapXi*Fel

	O2hd = k5*(SrbAt)*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*(SrbAt)*O2l

	Fehd = k6*(HapXac)*Fel - d6*Feh*Cyp51ac
	Feld = d6*Feh*Cyp51ac -k6*(HapXac)*Fel

	Efficiency = (O2h+Feh)/100
	

	return[Ergosterold, LErgosterold , Cyp51acd, Cyp51id, SrbAacd, SrbAid, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld, SrbAad, SrbArd,Efficiency]


t = np.linspace(0, 0.1, 100)
x_init = [100,1,100,1,100,1,100,1,100,1,100,1,100,1,50,50,100]
x_init = [100,1,200,200,100,100,60,60,200,200,100,1,100,1,100,100,0]
x1 = odeint(ErgModel, x_init, t)
plt.title('F', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t,x1[:,[0]], 'r--', linewidth=2)
plt.plot(t,x1[:,[14]], 'm--', linewidth=2)
plt.plot(t, x1[:,[2,4,6,8,10,12]], '-', linewidth = 2)
plt.legend(['Ergosterol', 'Cyp51', 'SrbAac', 'AtrR', 'HapX','O2', 'Fe' ],loc='upper center',prop={'size':12} ,ncol=7, fancybox=False)
#plt.plot(t, x1[:,[]], '--', linewidth = 2)




plt.subplot(3,3,1)	
t = np.linspace(0, 0.1, 100)
x_init = [100,1,100,1,100,1,100,1,100,1,100,1,100,1,50,50,100]
x_init = [100,1,200,200,100,100,60,60,200,200,100,1,100,1,100,100,0]
x1 = odeint(ErgModel, x_init, t)
plt.title('A', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t,x1[:,[0]], 'r--', linewidth=2)
plt.plot(t,x1[:,[14]], 'm--', linewidth=2)
plt.plot(t, x1[:,[2,4,6,8,10,12]], '-', linewidth = 2)
#plt.plot(t, x1[:,[]], '--', linewidth = 2)


plt.subplot(3,3,2)
t = np.linspace(0, 0.1, 100)
x_init = [100,1,100,1,100,1,100,1,100,1,100,1,100,1,50,50,100]
x_init = [100,1,200,200,100,100,60,60,200,200,100,1,100,1,100,100,0]
x1 = odeint(ErgModel2, x_init, t)
plt.title('B', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t,x1[:,[0]], 'r--', linewidth=2)
plt.plot(t,x1[:,[14]], 'm--', linewidth=2)
plt.plot(t, x1[:,[2,4,6,8,10,12]], '-', linewidth = 2)
#plt.plot(t, x1[:,[]], '--', linewidth = 2)


plt.subplot(3,3,3)
t = np.linspace(0, 0.1, 100)
x_init = [100,1,100,1,100,1,100,1,100,1,100,1,100,1,50,50,100]
x_init = [100,1,200,200,100,100,60,60,200,200,100,1,100,1,100,100,0]
x1 = odeint(ErgModel3, x_init, t)
plt.title('C', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t,x1[:,[0]], 'r--', linewidth=2)
plt.plot(t,x1[:,[14]], 'm--', linewidth=2)
plt.plot(t, x1[:,[2,4,6,8,10,12]], '-', linewidth = 2)
#plt.plot(t, x1[:,[]], '--', linewidth = 2)

plt.subplot(3,3,4)
t = np.linspace(0, 0.1, 100)
x_init = [100,1,100,1,100,1,100,1,100,1,100,1,100,1,50,50,100]
x_init = [100,1,200,200,100,100,60,60,200,200,100,1,100,1,100,100,0]
x1 = odeint(ErgModel4, x_init, t)
plt.title('D', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t,x1[:,[0]], 'r--', linewidth=2)
plt.plot(t,x1[:,[14]], 'm--', linewidth=2)
plt.plot(t, x1[:,[2,4,6,8,10,12]], '-', linewidth = 2)

plt.subplot(3,3,5)
t = np.linspace(0, 0.1, 100)
x_init = [100,1,100,1,100,1,100,1,100,1,100,1,100,1,50,50,100]
x_init = [100,1,200,200,100,100,60,60,200,200,100,1,100,1,100,100,0]
x1 = odeint(ErgModel5, x_init, t)
plt.title('E', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t,x1[:,[0]], 'r--', linewidth=2)
plt.plot(t,x1[:,[14]], 'm--', linewidth=2)
plt.plot(t, x1[:,[2,4,6,8,10,12]], '-', linewidth = 2)

plt.subplot(3,3,6)
t = np.linspace(0, 0.1, 100)
x_init = [100,1,100,1,100,1,100,1,100,1,100,1,100,1,50,50,100]
x_init = [100,1,200,200,100,100,60,60,200,200,100,1,100,1,100,100,0]
x1 = odeint(ErgModel6, x_init, t)
plt.title('F', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t,x1[:,[0]], 'r--', linewidth=2)
plt.plot(t,x1[:,[14]], 'm--', linewidth=2)
plt.plot(t, x1[:,[2,4,6,8,10,12]], '-', linewidth = 2)
plt.legend(['Ergosterol','SrbAt', 'Cyp51', 'SrbAac', 'AtrR', 'HapX','O2', 'Fe' ],loc='upper center',prop={'size':12} ,bbox_to_anchor=(-0.7, -0.22),ncol=8, fancybox=False)
#plt.plot(t, x1[:,[]], '--', linewidth = 2)
plt.show()
'''
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

	Azoled = k1*20 - d1*Azole*Cdr1Bt

	Ergosterold = k1*LowErgosterol*Cyp51ac - d1*HighErgosterol*(Fel*O2l)/100*Azole
	LErgosterold = d1*HighErgosterol*(Fel*O2l)/100*Azole - k1*LowErgosterol*Cyp51ac

	Cyp51acd = k2*Cyp51i*(SrbAac+AtrRac) - d2*Cyp51ac*(HapXac*2)
	Cyp51id = d2*Cyp51ac*(HapXac*2) - k2*Cyp51i*(SrbAac+AtrRac)

	SrbAacd = k2*SrbAi*(SrbAt)*(O2h) - d2*SrbAac*HighErgosterol*(O2l/10)
	SrbAid = d2*SrbAac*HighErgosterol*(O2l/10) - k2*SrbAi*(SrbAt)*(O2h)

	SrbAad = k2*SrbAr*O2l - d2*SrbAt*Feh
	SrbArd = d2*SrbAt*Feh - k2*SrbAr*O2l

	AtrRacd = k3*AtrRi*(Fel) - d3*AtrRac*Feh*LowErgosterol
	AtrRid = d3*AtrRac*Feh*LowErgosterol - k3*(AtrRi)*Fel

	HapXacd = k4*HapXi*Fel - d4*HapXac*Feh*HighErgosterol
	HapXid = d4*HapXac*Feh*HighErgosterol - k4*HapXi*Fel

	O2hd = k5*(SrbAt)*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*(SrbAt)*O2l

	Fehd = k6*(HapXac)*Fel - d6*Feh*Cyp51ac
	Feld = d6*Feh*Cyp51ac -k6*(HapXac)*Fel

	Cdr1Btd =k3*Cdr1Br*AtrRac - d3*Cdr1Bt*O2l
	Cdr1Brd = d3*Cdr1Bt*O2l - k3*AtrRac*Cdr1Br

	Efficiency = (O2h+Feh)/200

	return[Ergosterold, LErgosterold , Cyp51acd, Cyp51id, SrbAacd, SrbAid, AtrRacd, AtrRid, HapXacd, HapXid, O2hd, O2ld, Fehd, Feld, SrbAad, SrbArd, Efficiency, Azoled, Cdr1Btd, Cdr1Brd]


t = np.linspace(0, 1, 100)
x_init = [100,20,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100, 100, 100]
x1 = odeint(AzoleModel, x_init, t)
plt.title('A', loc='left', fontdict={'fontweight':'bold'})
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Level', fontsize = 12)
plt.plot(t, x1[:,[0,2,4,6,8,10,12,18]], '-', linewidth = 2)
plt.legend(['Ergosterol', 'Cyp51ac', 'SrbAac', 'AtrRac', 'HapXac','O2', 'Fe', 'Cdr1B' ])



	SrbAad = k1*SrbAr*O2l*Fel - d1*SrbAa*O2h*Feh
	SrbArd = d1*SrbAa*O2h*Feh - k1*SrbAr*O2l*Fel



	HErgosterold = k1*Cyp51ac - d1*HighErgosterol
	LErgosterold = d1*HighErgosterol - Cyp51ac

	Cyp51acd = k1*Cyp51i*SrbAac*AtrRac - d1*Cyp51ac*(HapXac*2)
	Cyp51id = d1*Cyp51ac*(HapXac*2) - k1*Cyp51i*SrbAac*AtrRac

	SrbAacd = k2*SrbAi*L(owE)rgo*O2hsterol - d2*SrbAac*Hi(ghE/10)rgo*O2lsterol
	SrbAid = k2*SrbAac*O2l*Fel*(O2l/10) - d2*SrbAi*O2h*Feh*(O2h)

	AtrRacd = k3*AtrRi*Fel*LowErgosterol - d3*AtrRac*HighErgosterol*Feh
	AtrRid = d3*AtrRac*HighErgosterol*Feh - k3*AtrRi*Fel*LowErgosterol

	HapXacd = k4*HapXi*Fel*HighErgosterol - d4*HapXac*LowErgosterol*Feh
	HapXid = d4*HapXac*LowErgosterol*Feh - k4*HapXi*Fel*HighErgosterol

	O2hd = k1*SrbAac*O2l*LowErgosterol - d1*O2h*HighErgosterol
	O2ld = d1*O2h*HighErgosterol - k1*SrbAac*O2l*LowErgosterol

	Fehd = k1*HapXac*Fel*LowErgosterol - d1*Feh*HighErgosterol
	Feld = d1*Feh*HighErgosterol-k1*HapXac*Fel*LowErgosterol





	HErgosterold = k1*Cyp51ac - d1*HighErgosterol
	LErgosterold = d1*HighErgosterol - Cyp51ac

	Cyp51acd = k1*Cyp51i*SrbAac*AtrRac - d1*Cyp51ac*(HapXac*2)
	Cyp51id = d1*Cyp51ac*(HapXac*2) - k1*Cyp51i*SrbAac*AtrRac

	SrbAacd = k2*SrbAi*L(owE)rgo*O2hsterol - d2*SrbAac*Hi(ghE/10)rgo*O2lsterol
	SrbAid = k2*SrbAac*O2l*Fel*(O2l/10) - d2*SrbAi*O2h*Feh*(O2h)

	AtrRacd = k3*AtrRi*Fel*LowErgosterol - d3*AtrRac*HighErgosterol*Feh
	AtrRid = d3*AtrRac*HighErgosterol*Feh - k3*AtrRi*Fel*LowErgosterol

	HapXacd = k4*HapXi*Fel*HighErgosterol - d4*HapXac*LowErgosterol*Feh
	HapXid = d4*HapXac*LowErgosterol*Feh - k4*HapXi*Fel*HighErgosterol

	O2hd = k1*SrbAi*LowErgosterol - d1*O2h*HighErgosterol
	O2ld = d1*O2h*HighErgosterol - k1*SrbAi*LowErgosterol

	Fehd = k1*HapXac - d1*Feh
	Feld = d1*Feh -k1*HapXac




	HErgosterold = k1*Cyp51ac - d1*HighErgosterol
	LErgosterold = d1*HighErgosterol - Cyp51ac

	Cyp51acd = k1*Cyp51i*SrbAac*AtrRac - d1*Cyp51ac*(HapXac*2)
	Cyp51id = d1*Cyp51ac*(HapXac*2) - k1*Cyp51i*SrbAac*AtrRac

	SrbAacd = k2*(Srb)Ai*LowErgo*O2hsterol - d2*Sr(bAa/10)c*HighErgo*O2lsterol
	SrbAid = k2*O2l*Fel*(O2l/10) - d2*SrbAi*O2h*Feh*(O2h)

	AtrRacd = k3*AtrRi*Fel*LowErgosterol - d3*AtrRac*HighErgosterol*Feh
	AtrRid = d3*AtrRac*HighErgosterol*Feh - k3*AtrRi*Fel*LowErgosterol

	HapXacd = k4*HapXi*Fel*HighErgosterol - d4*HapXac*LowErgosterol*Feh
	HapXid = d4*HapXac*LowErgosterol*Feh - k4*HapXi*Fel*HighErgosterol

	O2hd = k1*SrbAi - d1*O2h
	O2ld = d1*O2h - k1*SrbAi

	Fehd = k1*HapXac - d1*Feh
	Feld = d1*Feh -k1*HapXac





IN CASE OF EMERGENCY
	HErgosterold = k1*Cyp51ac- d1*HighErgosterol
	LErgosterold = d1*HighErgosterol - k1*Cyp51ac

	Cyp51acd = k2*SrbAac*AtrRac - d2*Cyp51ac*(HapXac*2)
	Cyp51id = d2*Cyp51ac*(HapXac*2) - k2*SrbAac*AtrRac

	SrbAacd = k2*SrbAi*LowErgo*O2hste(rol) - d2*SrbAac*HighErgo*O2lste(rol/10)
	SrbAid = d2*SrbAac*HighErgosterol - k2**(O2l/10)SrbAi*LowErgosterol*(O2h)

	#SrbAid = k2*O2l*Fel - d2*SrbAi*O2h*Feh

	AtrRacd = k3*Fel*LowErgosterol - d3*AtrRac*HighErgosterol*Feh
	AtrRid = d3*AtrRac*HighErgosterol*Feh - k3*Fel*LowErgosterol

	HapXacd = k4*Fel*HighErgosterol - d4*HapXac*LowErgosterol*Feh
	HapXid = d4*HapXac*LowErgosterol*Feh - k4*Fel*HighErgosterol

	O2hd = k5*SrbAi*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*SrbAi*O2l

	Fehd = k6*HapXac*Fel - d6*Feh*Cyp51ac
	Feld = d6*Feh*Cyp51ac -k6*HapXac*Fel






	Ergosterold = k1*Cyp51ac*Feh - d1*HighErgosterolFeh/10*O2h/10

	LErgosterold = Ergosterol

	Cyp51acd = k2*SrbAac*AtrRac - d2*Cyp51ac*(HapXac*2)
	Cyp51id = d2*Cyp51ac*(HapXac*2) - k2*SrbAac*AtrRac

	SrbAacd = k2*SrbAi*LowErgo*O2hste(rol) - d2*SrbAac*HighErgo*O2lste(rol/10)
	SrbAid = d2*SrbAac*HighErgosterol - k2**(O2l/10)SrbAi*LowErgosterol*(O2h)

	SrbAad = k1*SrbAr*O2l*Fel - d1*SrbAt*O2h*Feh
	SrbArd = d1*SrbAt*O2h*Feh - k1*SrbAr*O2l*Fel

	AtrRacd = k3*Fel*AtrRi*LowErgosterol - d3*AtrRac*HighErgosterol*Feh
	AtrRid = d3*AtrRac*HighErgosterol*Feh - k3*Fel*AtrRi*LowErgosterol

	HapXacd = k4*Fel*HapXi*HighErgosterol - d4*HapXac*LowErgosterol*Feh
	HapXid = d4*HapXac*LowErgosterol*Feh - k4*Fel*HapXi*HighErgosterol

	O2hd = k5*SrbAt*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*SrbAt*O2l

	Fehd = k6*HapXac*Fel - d6*Feh*Cyp51ac
	Feld = d6*Feh*Cyp51ac -k6*HapXac*Fel



	Ergosterold = k1*Cyp51ac*Efficiency - d1*HighErgosterol
	LErgosterold = d1*HighErgosterol - k1*Cyp51ac*Efficiency

	Cyp51acd = k2*SrbAac*AtrRac - d2*Cyp51ac*(HapXac*2)
	Cyp51id = d2*Cyp51ac*(HapXac*2) - k2*SrbAac*AtrRac

	SrbAacd = k2*(SrbAt/10)*Lo*O2hwErgos(ter/2)ol2h - d2*SrbAac*O2l*HighE(rgo/2)ste10ol*O2l
	SrbAid = d2*SrbAac*HighErgosterol*O2l - k2*(O2l/2)*(S10bAt/10)*LowErgo*O2hst(ero)l*O2h

	SrbAad = k1*SrbAr*O2l*Fel - d1*SrbAt*O2h*Feh
	SrbArd = d1*SrbAt*O2h*Feh - k1*SrbAr*O2l*Fel

	AtrRacd = k3*(AtrRi/100)*LowErgosterol*Fel - d3*AtrRac*HighErgosterol*Feh
	AtrRid = d3*AtrRac*HighErgosterol*Feh - k3*(AtrRi/100)*LowErgosterol*Fel

	HapXacd = k4*(HighErgosterol*Fel) - d4*HapXac*(LowErgosterol*Feh)
	HapXid = d4*HapXac*(LowErgosterol*Feh) - k4*(HighErgosterol*Fel)

	O2hd = k5*SrbAt*O2l - d5*O2h*Cyp51ac/2
	O2ld = d5*O2h*Cyp51ac/2 - k5*SrbAt*O2l

	Fehd = k6*HapXac*Fel - d6*Feh*Cyp51ac/2
	Feld = d6*Feh*Cyp51ac/2 -k6*HapXac*Fel

	Efficiency = (O2h+Feh)/200





	Ergosterold = k1*Cyp51ac*Efficiency - d1*HighErgosterol
	LErgosterold = d1*HighErgosterol - k1*Cyp51ac*Efficiency

	Cyp51acd = k2*SrbAac*AtrRac - d2*Cyp51ac*(HapXac*2)
	Cyp51id = d2*Cyp51ac*(HapXac*2) - k2*SrbAac(t/2)rRac
(
	S/210rbAacd = k2*Srb(A10*/2)LowErgo*O2hsterol - d2*SrbAac*(O2l*Hig*O2lhErgostero*O2ll)
	SrbAid = SrbAt *(O2h)

	SrbAad = k1*SrbAr*O2l - d1*SrbAt*Feh
	SrbArd = d1*SrbAt*Feh - k1*SrbAr*O2l

	AtrRacd = k3*(AtrRi)*LowErgosterol*Fel - d3*AtrRac*Feh*HighErgosterol
	AtrRid = d3*AtrRac*Feh*HighErgosterol - k3*(AtrRi)*LowErgosterol*Fel


	HapXacd = k4*HapXi*HighErgosterol*Fel - d4*HapXac*LowErgosterol*Feh
	HapXid = d4*HapXac*(LowErgosterol*Feh) - k4*HapXi*HighErgosterol*Fel

	O2hd = k5*SrbAt - d5*O2h*Cyp51ac/2
	O2ld = d5*O2h*Cyp51ac/2 - k5*SrbAt

	Fehd = k6*HapXac*Fel - d6*Feh*Cyp51ac/2
	Feld = d6*Feh*Cyp51ac/2 -k6*HapXac*Fel

	Efficiency = (O2h+Feh)/200








	Ergosterold = k1*Cyp51ac*Efficiency - d1*HighErgosterol
	LErgosterold = d1*HighErgosterol - k1*Cyp51ac*Efficiency

	Cyp51acd = k2*SrbAac*AtrRac - d2*Cyp51ac*(HapXac*2)
	Cyp51id = d2*Cyp51ac*(HapXac*2) - k2*SrbAac*At(rR2)c
(


/210
	SrbAacd = k2*(S(r10A/2)i)*LowEr*O2hgosterol - d2*SrbAac*(HighE*O2lrgoster*O2lol)
	SrbAid = SrbAt*(O2h)



	SrbAad = k2*SrbAr*O2l - d2*SrbAt*O2h*Feh
	SrbArd = d2*SrbAt*O2h*Feh - k2*SrbAr*O2l



	AtrRacd = k3*AtrRi*LowErgosterol - d3*AtrRac*HighErgosterol
	AtrRid = d3*AtrRac*HighErgosterol - k3*(AtrRi)*LowErgosterol


	HapXacd = k4*HapXi*Fel - d4*HapXac*Feh
	HapXid = d4*HapXac*Feh - k4*HapXi*Fel

	O2hd = k5*(SrbAt)*O2l - d5*O2h*Cyp51ac
	O2ld = d5*O2h*Cyp51ac - k5*(SrbAt)*O2l

	Fehd = k6*(HapXac)*Fel - d6*Feh*Cyp51ac/2
	Feld = d6*Feh*Cyp51ac/2 -k6*(HapXac)*Fel

	Efficiency = (O2h+Feh)/200





	'''

