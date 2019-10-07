import matplotlib.pyplot as plt 
import matplotlib
import math
import numpy as np
import matplotlib.style as style
import seaborn as sns

for name, hex in matplotlib.colors.cnames.items():
	colorname=[]
	colorid=[]
	colorname.append(name)
	colorid.append(hex)
zippedcolors = list(zip(colorname, colorid))
zippedcolors = sorted(zippedcolors, key=lambda x: x[1])

style.use('ggplot')

####################### ITZ graphs ########################

Label = ['0.125' ,'0.25', '0.5', '1', '2']

Cyp51 = [0.759, 0.902, 1.329, 1.196, 0.83956269]
Cyp51B = [0.486, 0.702, 0.803, 0.913, 0.683]
SrbA = [0.250, 0.102, 0.175, 0.154, -0.031]
AtrR = [0.493, 0.555, 0.918, 1.329, 1.589]
HapE = [-0.181, -0.235, 0.125, 0.422, 0.320]
HapB = [0.322, 0.501, -0.515, -0.820, -1.140]
HapC = [-0.058 ,-0.231, -0.260, -0.608, -0.580]
HapX = [-0.086, -0.173, -0.525, -0.728, -1.731]
Cdr1B = [1.100, 1.680, 1.911, 1.949, 1.504]

index=np.arange(len(Label))
palette=('blues_d')

fig, axs = plt.subplots(3, 3)
axs[0, 0].bar(Label, Cyp51)
axs[0, 0].axhline(y=0.0, linestyle='-')
axs[0, 0].set_title('Cyp51A')

axs[0, 1].bar(Label, Cyp51B)
axs[0, 1].axhline(y=0.0, linestyle='-')
axs[0, 1].set_title('Cyp51B')

axs[0, 2].bar(Label, SrbA)
axs[0, 2].axhline(y=0.0, linestyle='-')
axs[0, 2].set_title('SrbA')

axs[1, 0].bar(Label, HapE)
axs[1, 0].axhline(y=0.0, linestyle='-')
axs[1, 0].set_title('HapE')

axs[1, 1].bar(Label, HapB)
axs[1, 1].axhline(y=0.0, linestyle='-')
axs[1, 1].set_title('HapB')

axs[1, 2].bar(Label, HapC)
axs[1, 2].axhline(y=0.0, linestyle='-')
axs[1, 2].set_title('HapC')

axs[2, 0].bar(Label, HapX)
axs[2, 0].axhline(y=0.0, linestyle='-')
axs[2, 0].set_title('HapX')

axs[2, 1].bar(Label, Cdr1B)
axs[2, 1].axhline(y=0.0, linestyle='-')
axs[2, 1].set_title('Cdr1B')

axs[2, 2].bar(Label, AtrR )
axs[2, 2].axhline(y=0.0, linestyle='-')
axs[2, 2].set_title('AtrR')

for ax in axs.flat:
    ax.set(xlabel='µg/ml Itraconazole', ylabel='LFC Expression')
#for ax in axs.flat:
  # ax.label_outer()
plt.show()

###################### O2 Graphs ########################

Cyp51A = [8, 6.8, 7.5, 7.8]
SrbA = [6.5, 6.9, 6.3, 7.2]
AtrR = [5, 4.8, 5, 5.3]
HapB = [4.6, 5, 4.7, 5]
HapC = [6.3, 6.3, 6.3, 5.9]
HapE = [6.3, 5, 6.1, 5.7]
HapX = [4.6, 4, 4.4, 4.7]
Cdr1B = [5.7, 4.6, 5, 4.8]

Cyp51A = [290,110,180,210]
Cyp51B = [475,70,310,255]
SrbA = [90, 130,70,140]
AtrR = [33,25,30,38]
HapB = [23,32,25,33]
HapC = [58,28,46,39]
HapE = [90,30,75,50]
HapX = [23,15,20,26]
Cdr1B = [45,24,30,27]

Label= ['0h', '12h', '24h', '36h']

fig, axs = plt.subplots(3,3)
axs[0, 0].bar(Label, Cyp51A, color='steelblue')
axs[0, 0].axhline(y=0.0, linestyle='-')
axs[0, 0].set_title('Cyp51A')

axs[0, 1].bar(Label, Cyp51B, color='steelblue')
axs[0, 1].axhline(y=0.0, linestyle='-')
axs[0, 1].set_title('Cyp51B')

axs[0, 2].bar(Label, SrbA, color='steelblue')
axs[0, 2].axhline(y=0.0, linestyle='-')
axs[0, 2].set_title('SrbA')

axs[1, 0].bar(Label, HapE, color='steelblue')
axs[1, 0].axhline(y=0.0, linestyle='-')
axs[1, 0].set_title('HapE')

axs[1, 1].bar(Label, HapB, color='steelblue')
axs[1, 1].axhline(y=0.0, linestyle='-')
axs[1, 1].set_title('HapB')

axs[1, 2].bar(Label, HapC, color='steelblue')
axs[1, 2].axhline(y=0.0, linestyle='-')
axs[1, 2].set_title('HapC')

axs[2, 0].bar(Label, HapX, color='steelblue')
axs[2, 0].axhline(y=0.0, linestyle='-')
axs[2, 0].set_title('HapX')

axs[2, 1].bar(Label, Cdr1B, color='steelblue')
axs[2, 1].axhline(y=0.0, linestyle='-')
axs[2, 1].set_title('Cdr1B')

axs[2, 2].bar(Label, AtrR, color='steelblue' )
axs[2, 2].axhline(y=0.0, linestyle='-')
axs[2, 2].set_title('AtrR')

for ax in axs.flat:
    ax.set(xlabel='Time', ylabel='RPKM')
plt.show()

#################### Fe graphs #######################

style.use('ggplot')
Label= ['Cyp51A Ctrl', 'Cyp51A -Fe', 'Cyp51B Ctrl', 'Cyb51B -Fe', 'SrbA Ctrl', 'SrbA -Fe', 'AtrR Ctrl', 'AtrR -Fe', 'Cdr1B Ctrl', 'Cdr1B -Fe', 'HapB Ctrl', 'HapB -Fe', 'HapC Ctrl', 'HapC -Fe', 'HapE Ctrl', 'HapE -Fe', 'HapX Ctrl', 'HapX -Fe']
Data=[125,70,170,155,77,150,31,76,78,127,25,26,91,95,67,108,26,331,]

Cyp51A = [125,70]
Cyp51B = [170,155]
SrbA = [77,150]
AtrR = [31,76]
Cdr1B = [78,127]
HapB = [25,26]
HapC = [91,95]
HapE = [67,108]
HapX = [26,331]

Fe_starved = (70, 155, 150, 76, 127, 26, 95, 108, 331)
Ctrl = (125, 170, 77, 31, 78, 25, 91, 67, 26)

n_groups = 9

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width=0.35
opacity = 1

rects1 = plt.bar(index, Ctrl,bar_width,color='grey',label = 'Control')
rects2 = plt.bar(index+bar_width, Fe_starved,bar_width,label='Fe Starved')
plt.xlabel('Gene ID')
plt.ylabel('Fragments per Kilobase per Million (FKPM)')
plt.xticks(index+bar_width/2, ('Cyp51A', 'Cyp51B', 'SrbA', 'AtrR', 'Cdr1B', 'HapB', 'HapC', 'HapE', 'HapX'))
plt.title('Relative Gene Expression During Iron Starvation')
plt.legend()
plt.tight_layout()
plt.show()


