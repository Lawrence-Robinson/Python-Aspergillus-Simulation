import boolean2 as b2 
import matplotlib.pyplot as plt
from boolean2 import util

#########-------------######## Erg Regulation ########------------#########


model_definition='''
Cyp51=False
SrbA=False
SrbAac=False
AtrR=False
Ergosterol=False
O2=False
CBCac=False
HapX=False
Fe=False

Cyp51* = SrbAac or AtrR and not CBCac
SrbA* = not O2 and not Fe
SrbAac* = SrbA and O2 or not Ergosterol
AtrR* = not Ergosterol or not Fe
CBCac* = HapX
HapX* = Ergosterol or not Fe
O2* = SrbA 
Fe* = HapX 
Ergosterol* = Cyp51

'''


model = b2.Model(text=model_definition, mode='sync')
model.initialize()
model.iterate(steps=20)
for node in model.data:
    print node, model.data[node]
model.report_cycles()

image = list()
for node in model.data:
    image.append(model.data[node])
plt.yticks(range(0,9), model.data)
plt.imshow(image, cmap=plt.cm.get_cmap('RdYlGn'), interpolation='none')
#plt.title('Ergosterol Simulation')
plt.show(block=True)
#plt.legend(loc="best",ncol=1, tit


model_definition='''
Cyp51=False
SrbA=False
SrbAac=False
AtrR=False
Ergosterol=False
O2=False
CBCac=False
HapX=False
Fe=False
Cdr1B=False
Azole=True

Azole*=not Cdr1B
Cyp51* = SrbAac or AtrR and not CBCac
SrbA* = not O2 and not Fe
SrbAac* = SrbA and O2 or not Ergosterol
AtrR* = not Ergosterol and not Fe
Cdr1B* = AtrR and O2
CBCac* = HapX
HapX* = Ergosterol and not Fe
O2* = SrbA
Fe* = HapX 
Ergosterol* = Cyp51 and not Azole



'''
'''
Cyp51* = SrbAac or AtrR and not CBCac
SrbA* = not O2 and not Fe
SrbAac* = SrbA and O2 or not Ergosterol
AtrR* = not Ergosterol or not Fe
CBCac* = HapX
HapX* = Ergosterol or not Fe
O2* = SrbA 
Fe* = HapX 
Ergosterol* = Cyp51
'''

model = b2.Model(text=model_definition, mode='sync')
model.initialize()
model.iterate(steps=20)
for node in model.data:
    print node, model.data[node]
model.report_cycles()

image = list()
for node in model.data:
    image.append(model.data[node])
plt.yticks(range(0,11), model.data)
plt.imshow(image, cmap=plt.cm.get_cmap('RdYlGn'), interpolation='none')
#plt.title('Ergosterol Simulation')
plt.show(block=True)
#plt.legend(loc="best",ncol=1, tit




'''
model_definition=
Cyp51=Random
SrbA=Random
AtrR=Random
CBC=Random
Ergosterol=Random
O2=Random
Fe=False

Cyp51* = SrbA or AtrR and not CBC 
SrbA* = AtrR or not Ergosterol and not O2
AtrR* = not Ergosterol
CBC* = not Fe
Ergosterol* = Cyp51 and O2 or Fe
Fe* = False
O2* = Random



model = b2.Model(text=model_definition, mode='async')
model.initialize()
model.iterate(steps=30)
for node in model.data:
    print node, model.data[node]
model.report_cycles()

image = list()
for node in model.data:
    image.append(model.data[node])
plt.yticks(range(0,7), model.data)
plt.imshow(image, cmap=plt.cm.get_cmap('RdYlGn'), interpolation='none')
plt.title('Ergosterol Regulation without Azoles')
plt.show(block=True)
#plt.legend(loc="best",ncol=1, tit

coll = util.Collector()
for i in range(100):
    model = b2.Model(text=model_definition, mode='sync')
    model.initialize()
    model.iterate(steps=30)
    coll.collect(states=model.states, nodes=model.nodes)
avgs = coll.get_averages()

image = list()
for node in model.data:
    image.append(avgs[node])
plt.yticks(range(0,7), model.data)
plt.imshow(image, cmap=plt.cm.get_cmap('RdYlGn'), interpolation='none')
plt.title('100 simulation average')
plt.show(block = True)




############################# Azole Erg Regulation #############################


model_definition=
Azole=True
Cyp51=True
SrbA=True
AtrR=Random
O2=Random
Fe=Random
Cdr1B=Random
CBC=True
Ergosterol=True

Azole* = not Cdr1B
Cyp51* = SrbA or AtrR and not CBC
SrbA* = AtrR or not O2 or not Ergosterol
AtrR* = Azole and not Ergosterol
Cdr1B* = AtrR
CBC* = not Fe
Ergosterol* = Cyp51 and O2 or Fe and not Azole
O2* = Random
Fe* = Random


model = b2.Model(text=model_definition, mode='async')
model.initialize()
model.iterate(steps=20)
for node in model.data:
    print node, model.data[node]
model.report_cycles()

image = list()
for node in model.data:
    image.append(model.data[node])
plt.yticks(range(0,9), model.data)
plt.imshow(image, cmap=plt.cm.get_cmap('RdYlGn'), interpolation='none')
plt.title('complete model with stochasticity')
#plt.legend(loc="best",ncol=1, title="Legend", fancybox=False)
plt.show(block = True)

from boolean2 import util

coll = util.Collector()
for i in range(100):
    model = b2.Model(text=model_definition, mode='async')
    model.initialize()
    model.iterate(steps=20)
    coll.collect(states=model.states, nodes=model.nodes)
avgs = coll.get_averages()

image = list()
for node in model.data:
    image.append(avgs[node])
plt.yticks(range(0,9), model.data)
plt.imshow(image, cmap=plt.cm.get_cmap('RdYlGn'), interpolation='none')
plt.show(block = True)


######################### Natural Resistance ######################### 


model_definition=
Azole=True
Cyp51=True
SrbA=True
AtrR=Random
O2=Random
Fe=Random
Cdr1B=Random
CBC=True
Ergosterol=True

Azole* = not Cdr1B
Cyp51* = SrbA or AtrR and not CBC
SrbA* = AtrR or not O2 or not Ergosterol
AtrR* = Azole and not Ergosterol
Cdr1B* = AtrR
CBC* = not Fe
Ergosterol* = Cyp51 and O2 or Fe
O2* = Random
Fe* = Random


model = b2.Model(text=model_definition, mode='async')
model.initialize()
model.iterate(steps=20)
for node in model.data:
    print node, model.data[node]
model.report_cycles()

image = list()
for node in model.data:
    image.append(model.data[node])
plt.yticks(range(0,9), model.data)
plt.imshow(image, cmap=plt.cm.get_cmap('RdYlGn'), interpolation='none')
plt.title('complete model with stochasticity')
#plt.legend(loc="best",ncol=1, title="Legend", fancybox=False)
plt.show(block = True)

from boolean2 import util

coll = util.Collector()
for i in range(100):
    model = b2.Model(text=model_definition, mode='async')
    model.initialize()
    model.iterate(steps=20)
    coll.collect(states=model.states, nodes=model.nodes)
avgs = coll.get_averages()

image = list()
for node in model.data:
    image.append(avgs[node])
plt.yticks(range(0,9), model.data)
plt.imshow(image, cmap=plt.cm.get_cmap('RdYlGn'), interpolation='none')
plt.show(block = True)

'''

