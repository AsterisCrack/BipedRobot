
import math
import numpy as np

#Establezco los valores predeterminados de la distribución del robot (en mm)
#Cada letra es la separación entre dos pintos de rotación, empezando por el más alto.
#"f" es la separación entre el punto de anclaje de las dos piernas
#"bodyCenter" es la altura del centro del cuerpo respectiva a el punto de anclaje de las dos piernas
#"minL" es la longitud mínima que puede tener la pierna del robot para que no haya colusiones
a = 33
b = 40
c = 62
d = 72
e = 52
f = 69
bodyCenter = 90
minL = 65
#La posición inicial del robot
X = [[0,sum([a,b,c,d,e]),0],[0,0,0]]
#El vector de los motores en la pos inicial
#Si es necesario ajustarlo se puede hacer ejecutando el programa "InitialDataTweaker.py"
initialData = [128, 0, 0, 88, 90, 128, 46, 84, 59, 174, 92, 89, 93, 135, 41, 94, 0, 0]

def getInitialData():
    return initialData
def getFullLength():
    return sum([a,b,c,d,e])
def Longitud(X, J2):
    #Calculamos la longitud de la pierna
    #Para ello calculo la posición del punto de rotacióndel tobillo respecto de la cadera
    #Luego calculo la ecuación de distancia en R3
    eAux = e*math.cos(J2)*math.cos(X[1][0])
    Xnuevo = (X[0][0])-eAux*math.sin(J2)
    Ynuevo = (X[0][1])-eAux*math.cos(J2)-a-b*math.cos(J2)
    Znuevo = (X[0][2])-eAux*math.sin(X[1][0])
    longitud = math.sqrt((Xnuevo)**2+(Ynuevo)**2+(Znuevo)**2)
    return longitud

def aplicarRotaciones(R, Xder, Xizq):
    #Esta función sirve para aplicar la rotación del cuerpo
    #Para ello desplazo los puntos de origen de las piernas mediante matrices de rotación
    Rx = math.radians(R[0])
    Ry = math.radians(R[1])
    Rz = math.radians(R[2])

    #Matriz de rotacion en X
    MatRx = [[1,0,0],[0,math.cos(Rx),-math.sin(Rx)],[0,math.sin(Rx),math.cos(Rx)]]
    #Matriz de rotacion en Y
    MatRy = [[math.cos(Ry),0,math.sin(Ry)],[0,1,0],[-math.sin(Ry),0,math.cos(Ry)]]
    #Matriz de rotacion en Z
    MatRz = [[math.cos(Rz),-math.sin(Rz),0],[math.sin(Rz),math.cos(Rz),0],[0,0,1]]

    #Los origenes iniciales son
    O1 = [f/2, -bodyCenter, 0]
    O2 = [-f/2, -bodyCenter, 0]

    #Rotamos los origenes
    O1transformed = np.matmul(MatRx,np.matmul(MatRy,np.matmul(MatRz, O1))).tolist()
    O2transformed = np.matmul(MatRx,np.matmul(MatRy,np.matmul(MatRz, O2))).tolist()
    XderTransformed = np.matmul(MatRx,np.matmul(MatRz, Xder)).tolist()
    XizqTransformed = np.matmul(MatRx,np.matmul(MatRz, Xizq)).tolist()
    
    #Restamos los origenes
    O1transformed = [O1transformed[0]-O1[0], O1transformed[1]-O1[1], O1transformed[2]-O1[2]]
    O2transformed = [O2transformed[0]-O2[0], O2transformed[1]-O2[1], O2transformed[2]-O2[2]]

    return O1transformed, O2transformed, XderTransformed, XizqTransformed



def getJPiernas(X):
    error = None
    #Primero sacamos la rotacion de J1 y la matriz de rotación para transformar P
    RotMat = [[math.cos(X[1][1]),0,math.sin(X[1][1])],
              [0,1,0],
              [-math.sin(X[1][1]),0,math.cos(X[1][1])]]
    P0 = [X[0]]
    #multiplico las matrices P0 y RotMat
    #P0 es una matriz de 3x1
    #RotMat es una matriz de 3x3
    P = [[0,0,0]]
    for i in range(len(P0)): 
        for j in range(len(RotMat[0])): 
            for k in range(len(RotMat)): 
                # resulted matrix 
                P[i][j] += P0[i][k] * RotMat[k][j] 
                
    #Sustituyo para calcularlo a la nueva posición
    X[0] = P[0]

    #Pruebo si es posible
    J2 = math.atan(((X[0][0]))/(((X[0][1])-math.cos(X[1][0])*e-a)))
    L = Longitud(X, J2)

    if L <= c+d and L >= minL:
        #Comenzamos sacando los motores para la posición deseada
        J4 = math.acos((-Longitud(X, J2)**2+c**2+d**2)/(2*c*d))

        eAux = e*math.cos(J2)*math.cos(X[1][0])
        Znuevo = (X[0][2])-eAux*math.sin(X[1][0])
        Ynuevo = (X[0][1])-eAux*math.cos(J2)-a-b*math.cos(J2)
        J3_0 = math.atan(Znuevo/Ynuevo)

        #Añadimos a J3 el ángulo del triángulo que forma con J4 y J5
        J3_1 = math.acos((-d**2+c**2+Longitud(X, J2)**2)/(2*c*Longitud(X, J2)))
        J3 = J3_0 + J3_1

        #Y las rotaciones
        J1 = X[1][1]
        J5 = X[1][0] - J3_0 +  math.pi-J3_1-J4 
        J6 = X[1][2] - J2 
    else:
        error = "Error, no es una configuración posible (longitud)"
        J1 = None
        J2 = None
        J3 = None
        J4 = None
        J5 = None
        J6 = None
    return [J1, J2, J3, J4, J5, J6], error
    
    
def getJPiernaDer(X):
    J = [initialData[11], initialData[12], initialData[13], initialData[14], initialData[15], initialData[8]]
    #X pasa a radianes
    for i in range(len(X[1])):
        X[1][i] = math.radians(X[1][i])

    NewJ, error = getJPiernas(X)    
    if error == None:
        #Al vector J le sumamos los valores de los motores en sus posiciones
        J[0] = J[0]+math.degrees(NewJ[0])
        J[1] = J[1]+math.degrees(NewJ[1])
        J[2] = J[2]-math.degrees(NewJ[2])
        J[3] = J[3]-math.degrees(NewJ[3])+180
        J[4] = J[4]+math.degrees(NewJ[4])
        J[5] = J[5]-math.degrees(NewJ[5])

        #Redondeo todos los valores de J
        for i in range(len(J)):
            J[i] = int(J[i])

        if any(i>180 for i in J) or any(i<0 for i in J):
            error = "Error, no es una configuración posible (angulo) (pierna derecha)"
            return error
        else:
            return J
    else:
        error += " (pierna derecha)"
        return error

def getJPiernaIzq(X): 
    J = [initialData[3], initialData[4], initialData[5], initialData[6], initialData[7], initialData[0]]
    #X pasa a radianes
    for i in range(len(X[1])):
        X[1][i] = math.radians(X[1][i])
    
    NewJ, error = getJPiernas(X)    
    if error == None:
        #Al vector J le sumamos los valores de los motores en sus posiciones
        J[0] = J[0]+math.degrees(NewJ[0])
        J[1] = J[1]+math.degrees(NewJ[1])
        J[2] = J[2]-math.degrees(NewJ[2])
        J[3] = J[3]-math.degrees(NewJ[3])+180
        J[4] = J[4]+math.degrees(NewJ[4])
        J[5] = J[5]-math.degrees(NewJ[5])

        #Redondeo todos los valores de J
        for i in range(len(J)):
            J[i] = int(J[i])

        if any(i>180 for i in J) or any(i<0 for i in J):
            error = "Error, no es una configuración posible (angulo) (pierna izquierda)"
            return error
        else:
            return J
    else:
        error += " (pierna izquierda)"
        return error
