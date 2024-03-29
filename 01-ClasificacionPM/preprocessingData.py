# Practica 2 de Redes de Neuronas artificiales.

import numpy as np
import random as rnd

# Funcion que coge los datos del archivo y los pasa en forma de matriz en numpy
def datainput():
    f = open('datosNubes.dat')
    data = np.loadtxt(f,dtype=float, delimiter=',',skiprows=1)
    np.array(data)
    f.close()
    return data


# Funcion cuyo objetivo es normalizar los valores de la matriz entre 0 y 1
def normalizacion(data):
    #columna1
    v0 = data[:,0]
    maximo1 = np.amax(v0)
    minimo1 = np.amin(v0)
    data[:,0] = (v0-minimo1)/(maximo1-minimo1)

    #columna2
    v1 = data[:,1]
    maximo2 = np.amax(v1)
    minimo2 = np.amin(v1)
    data[:,1] = (v1-minimo2)/(maximo2-minimo2)

    #columna3
    v2 = data[:,2]
    maximo3 = np.amax(v2)
    minimo3 = np.amin(v2)
    data[:,2] = (v2-minimo3)/(maximo3-minimo3)

    #columna4
    v3 = data[:,3]
    maximo4 = np.amax(v3)
    minimo4 = np.amin(v3)
    data[:,3] = (v3-minimo4)/(maximo4-minimo4)

    #columna5
    v4 = data[:,4]
    maximo5 = np.amax(v4)
    minimo5 = np.amin(v4)
    data[:,4] = (v4-minimo5)/(maximo5-minimo5)

    #columna6
    v5 = data[:,5]
    maximo6 = np.amax(v5)
    minimo6 = np.amin(v5)
    data[:,5] = (v5-minimo6)/(maximo6-minimo6)

    #columna7
    v6 = data[:,6]
    maximo7 = np.amax(v6)
    minimo7 = np.amin(v6)
    data[:,6] = (v6-minimo7)/(maximo7-minimo7)

    #columna8
    v7 = data[:,7]
    maximo8 = np.amax(v7)
    minimo8 = np.amin(v7)
    data[:,7] = (v7-minimo8)/(maximo8-minimo8)

    #columna9
    v8 = data[:,8]
    maximo9 = np.amax(v8)
    minimo9 = np.amin(v8)
    data[:,8] = (v8-minimo9)/(maximo9-minimo9)

    #columna10
    v9 = data[:,9]
    maximo10 = np.amax(v9)
    minimo10 = np.amin(v9)
    data[:,9] = (v9-minimo10)/(maximo10-minimo10)

    #columna11
    v10 = data[:,10]
    maximo11 = np.amax(v10)
    minimo11 = np.amin(v10)
    data[:,10] = (v10-minimo11)/(maximo11-minimo11)

    #columna12
    v11 = data[:,11]
    maximo12 = np.amax(v11)
    minimo12 = np.amin(v11)
    data[:,11] = (v11-minimo12)/(maximo12-minimo12)

########################################################################################################
# CODIGO DE PREPROCESADO DE DATOS 

# Obtenemos la matriz a partir de la entrada de datos del fichero
datos = datainput()

# Normalizamos la matriz de datos
normalizacion(datos)

np.random.shuffle(datos)

# Division en 4 partes P1, P2, P3, P4

p1 = np.empty(shape=(1,13))

p2 = np.empty(shape=(1,13))

p3 = np.empty(shape=(1,13))

p4 = np.empty(shape=(1,13))


# Contadores de cada tipo de imagen
contadorCieloD = 1
contadorMulinube = 1
contadorNube = 1


# Numero maximo de cada instancia por conjunto para guardar proporcion NO USADO AUN
numerCieloD = 12
numeroMultinube = 39
numeroNube1 = 128
numeroNube2 = 129

# Meter datos cielo despejado
for i in range(717):
    if(datos[i][12]==1):
        if(contadorCieloD<=12):
            p1 = np.vstack((p1,datos[i]))
            contadorCieloD = contadorCieloD+1
        if(12<contadorCieloD and contadorCieloD<=24):
            p2 = np.vstack((p2,datos[i]))
            contadorCieloD = contadorCieloD+1
        if(24<contadorCieloD and contadorCieloD<=36):
            p3 = np.vstack((p3,datos[i]))
            contadorCieloD = contadorCieloD+1
        if(36<contadorCieloD and contadorCieloD<=48):
            p4 = np.vstack((p4,datos[i]))
            contadorCieloD = contadorCieloD+1

# Meter datos multinube
for i in range(717):
    if(datos[i][12]==2):
        if(contadorMulinube<=39):
            p1 = np.vstack((p1,datos[i]))
            contadorMulinube = contadorMulinube+1
        if(39<contadorMulinube and contadorMulinube<=78):
            p2 = np.vstack((p2,datos[i]))
            contadorMulinube = contadorMulinube+1
        if(78<contadorMulinube and contadorMulinube<=117):
            p3 = np.vstack((p3,datos[i]))
            contadorMulinube = contadorMulinube+1
        if(117<contadorMulinube and contadorMulinube<=156):
            p4 = np.vstack((p4,datos[i]))
            contadorMulinube = contadorMulinube+1


# Meter datos nube
for i in range(717):
    if(datos[i][12]==3):
        if(contadorNube<=128):
            p1 = np.vstack((p1,datos[i]))
            contadorNube = contadorNube+1
        if(128<contadorNube and contadorNube<=256):
            p2 = np.vstack((p2,datos[i]))
            contadorNube = contadorNube+1
        if(256<contadorNube and contadorNube<=384):
            p3 = np.vstack((p3,datos[i]))
            contadorNube = contadorNube+1
        if(384<contadorNube and contadorNube<=513):
            p4 = np.vstack((p4,datos[i]))
            contadorNube = contadorNube+1


p1 = np.delete(p1,0,0)
p2 = np.delete(p2,0,0)
p3 = np.delete(p3,0,0)
p4 = np.delete(p4,0,0)

np.random.shuffle(p1)
np.random.shuffle(p2)
np.random.shuffle(p3)
np.random.shuffle(p4)


# Creamos los conjuntos de entrenamiento
p234 = p2
p134 = p1
p124 = p1
p123 = p1

p234 = np.vstack((p234,p3))
p234 = np.vstack((p234,p4))

p134 = np.vstack((p134,p3))
p134 = np.vstack((p134,p4))

p124 = np.vstack((p124,p2))
p124 = np.vstack((p124,p4))

p123 = np.vstack((p123,p2))
p123 = np.vstack((p123,p3))

# Aleatorizamos los conjuntos de entrenamiento

np.random.shuffle(p123)
np.random.shuffle(p234)
np.random.shuffle(p134)
np.random.shuffle(p124)



# TODO CREAR LOS FICHEROS DE LOS FOLDS DE TRAIN Y TEST COMO PONE EN LA PRESENTACION

"""
# Aleatorizacion de los datos (rotar las filas de la matriz)
np.random.shuffle(datos)

# Dividimos el conjunto de datos en 3 subconjuntos 
# Subconjunto de datos de entrenamiento
trainData = datos[:10200,]

# Subconjunto de datos de validacion
validationData = datos[10200:13600]

# Subconjunto de datos de test
testData = datos[13600:]

"""
####################################### SALIDA DE DATOS #################################################

def changer(cdata):
    for i in range(np.shape(cdata)[0]):
        if(cdata[i,12]==1):
            cdata[i,12]=10
        if(cdata[i,12]==2):
            cdata[i,12]=20
        if(cdata[i,12]==3):
            cdata[i,12]=30



changer(p1)
changer(p2)
changer(p3)
changer(p4)
changer(p123)
changer(p124)
changer(p234)
changer(p134)




# Fichero datos normalizados 
f1 = open("dataout.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f1, datos, delimiter=',', fmt='%f')
f1.close()






# Ficheros p1, p2, p3, p4
f2 = open("Test1.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f2, p1, delimiter=',', fmt='%f')
f2.close()

f2 = open("Test2.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f2, p2, delimiter=',', fmt='%f')
f2.close()

f2 = open("Test3.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f2, p3, delimiter=',', fmt='%f')
f2.close()

f2 = open("Test4.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f2, p4, delimiter=',', fmt='%f')
f2.close()


# Ficheros de entrenamiento
f2 = open("Train1.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f2, p234, delimiter=',', fmt='%f')
f2.close()

f2 = open("Train2.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f2, p134, delimiter=',', fmt='%f')
f2.close()

f2 = open("Train3.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f2, p124, delimiter=',', fmt='%f')
f2.close()

f2 = open("Train4.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f2, p123, delimiter=',', fmt='%f')
f2.close()







"""
# SALIDA DATOS DE ENTRENAMIENTO
# Abrimos el archivo de salida de datos 
f2 = open("trainData.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f2, trainData, delimiter=' , ', fmt='%f')
f2.close()


# SALIDA DATOS DE VALIDACION
# Abrimos el archivo de salida de datos 
f3 = open("validationData.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f3, validationData, delimiter=' , ', fmt='%f')
f3.close()


# SALIDA DATOS DE TEST
# Abrimos el archivo de salida de datos 
f4 = open("testData.txt", "w")
# Guardamos la matriz en su formato en el archivo de salida de texto
np.savetxt(f4, testData, delimiter=' , ', fmt='%f')
f4.close()

"""



