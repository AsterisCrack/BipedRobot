
#Las animaciones vienen dadas por una variable valores y una variable tiempo
#Vienen escritos en un documento de texto de la forma "valores,tiempo" en cada linea. Cuando haya una linea vacia, se termina la animacion. 
#El nombre de la animacion es el nombre del documento de texto.

import IKcontrol as IK
import threading
import time
import copy
import IKserialCom as com

global canPlay
canPlay = True
global fps
fps = 15

def readAnimation(fileName):
    #Cada linea es un frame y aparece así: 
    #0.0:259:0.0;0.0:0.0:0.0;/-55.0:259:0.0;0.0:0.0:0.0;/0:0:0;0:0:0;/,0
    error = None
    try:
        with open(fileName, "r") as file:
            lines = file.readlines()
            XAnimacion = []
            Xfinal = []
            TiemposAnimacion = []
            for line in lines: #Frames
                values = line.split(",") #X y Tiempos
                TiemposAnimacion.append(float(values[1]))
                #Values[0] viene de la forma float:float:float;float:float:float/ etc 
                #Convierto ese string a una lista de listas de floats
                XAnimacion = values[0].split("/")
                XAnimacion.pop(-1)
                for i in range(len(XAnimacion)):
                    XAnimacion[i] = XAnimacion[i].split(";")
                    XAnimacion[i].pop(-1)
                    for j in range(len(XAnimacion[i])):
                        XAnimacion[i][j] = [float(x) for x in XAnimacion[i][j].split(":")]
                Xfinal.append(XAnimacion)

        return Xfinal, TiemposAnimacion, error
    except:
        error = "No se ha encontrado el archivo de animación"
        return None, None, error

#La siguiente función reproduce una animación desde su archivo no procesado. No se utiliza actualmente, pero puede ser útil.
def playAnimation2(XAnimacion, TiemposAnimacion, XInicial, SetSliders):
    ################
    ###IMPORTANTE### => La animacion se ejecuta en un thread aparte
    ################
    #En vez de enviar la posición y esperar, envío la posición en intervalos de tiempo iguales pero más rápìdo o lento según el tiempo
    #Con esto consigo un movimiento más uniforme y natural.
    #Si se requieren paradas se pueden agregar como frames iguales al anterior
    #La desventaja es posiblemente que el puerto serial no sea capaz de hacer muchas por segundo.
    #Sin duda con una conexión rápida será la opción deseada.

    global canPlay
    canPlay = True
    X = [XInicial]

    XAnimacion.insert(0, XInicial)

    XAux = copy.deepcopy(X[0])
    IK.calcularTodo(XAux) 

    for i in range(len(XAnimacion)-1):
        t = int(fps * float(TiemposAnimacion[i]))
        #DifX es la diferencia entre el valor actual de X y el valor i+1 de X
        DifX = [(XAnimacion[i+1][j][k][l] - XAnimacion[i][j][k][l])/t for j in range(len(XAnimacion[i])) for k in range(len(XAnimacion[i][j])) for l in range(len(XAnimacion[i][j][k]))]
        #DifX debe dividirse en t para que sea una velocidad constante
        X = XAnimacion[i]
        for _ in range(t):
            for j in range(len(XAnimacion[i])):
                for k in range(len(XAnimacion[i][j])):
                    for l in range(len(XAnimacion[i][j][k])):
                        X[j][k][l] = X[j][k][l] + DifX[j*6+k*3+l]
            if canPlay:
                XAux = copy.deepcopy(X)
                IK.calcularTodo(XAux)
                SetSliders(X)
                time.sleep(1/fps)
            else:
                break
    if not canPlay:
        print("Animación detenida")
    else:
        global ready
        ready = True
        print("Done")

#Al querer ejecutar una animación, se debe llamar a esta función. Depende del tipo se ejecutará con distintas funciones y argumentos.
#Esta función crea y ejecuta un thread aparte para que la animación se ejecute en segundo plano.
#Esto es necesario, por ejemplo, para poder pararla con el botón se "stop".
def startAnimation(XAnimacion, TiemposAnimacion, XInicial, SetSliders, tipo = "normal"):
    global canPlay
    canPlay = False
    if tipo == "normal":
        animationThread = threading.Thread(target=playAnimation2, args=(XAnimacion, TiemposAnimacion, XInicial, SetSliders))
    elif tipo == "ciclo":
        animationThread = threading.Thread(target=playWalkingCycle, args=(XAnimacion, TiemposAnimacion, XInicial, SetSliders))
    elif tipo == "procesada":
        animationThread = threading.Thread(target=playProcessedAnimation, args=(XAnimacion, XInicial, SetSliders))
    elif tipo == "ciclo_procesada":
        animationThread = threading.Thread(target=playProcessedAnimationCycle, args=(XAnimacion, XInicial, SetSliders))
    animationThread.start()

#Guarda la forma original de la animación en un archivo de texto.
def saveAnimation(fileName, XAnimacion, TiemposAnimacion):
    with open(fileName, "w") as file:
        for i in range(len(XAnimacion)): #Frames, cada linea
            strToWrite = ""
            for j in range(len(XAnimacion[i])): #Partes del cuerpo, cada /
                for k in range(len(XAnimacion[i][j])): #Pos y rotación, cada ;
                    strToWrite += str(XAnimacion[i][j][k][0]) + ":" + str(XAnimacion[i][j][k][1]) + ":" + str(XAnimacion[i][j][k][2]) + ";"
                strToWrite += "/"
            strToWrite += ","+str(TiemposAnimacion[i])+"\n"
            file.write(strToWrite)

#Guarda la forma procesada de la animación en un archivo de texto.
def saveProcessedAnimation(fileName, XAnimacion):
    with open("PROCESSED_"+fileName, "w") as file:
        for i in range(len(XAnimacion)): #Frames, cada linea
            strToWrite = str(XAnimacion[i])+"\n"
            file.write(strToWrite)

#Devuelve cada frame de la forma procesada de la animación.
def readProcessedAnimation(fileName):
    with open("PROCESSED_"+fileName, "r") as file:
        XAnimacion = []
        for line in file:
            XAnimacion.append(line.strip("\n"))
        return XAnimacion

#Similar a "PlayAnimation2", pero en vez de enviar los datos los guarda.
def processAnimation(XAnimacionRAW, TiemposAnimacionRAW):
    Xprocessed = [] 
    XAnimacion = copy.deepcopy(XAnimacionRAW)
    TiemposAnimacion = copy.deepcopy(TiemposAnimacionRAW)
    for i in range(len(XAnimacion)-1):
        t = int(fps * float(TiemposAnimacion[i]))
        #DifX es la diferencia entre el valor actual de X y el valor i+1 de X
        DifX = [(XAnimacion[i+1][j][k][l] - XAnimacion[i][j][k][l])/t for j in range(len(XAnimacion[i])) for k in range(len(XAnimacion[i][j])) for l in range(len(XAnimacion[i][j][k]))]
        #DifX debe dividirse en t para que sea una velocidad constante
        X = XAnimacion[i]
        for _ in range(t):
            for j in range(len(XAnimacion[i])):
                for k in range(len(XAnimacion[i][j])):
                    for l in range(len(XAnimacion[i][j][k])):
                        X[j][k][l] = X[j][k][l] + DifX[j*6+k*3+l]
            XAux = copy.deepcopy(X)
            data, valores = IK.calcularTodo(XAux, False)
            Xprocessed.append(data)

    return Xprocessed

#Crea un ciclo completo de animación y llama a la función anterior para procesarla. 
#Las animaciones cíclicas deben animarse hasta el primer frame espejo, sin incluirlo.
def processAnimationCycle(XAnimacionRAW, TiemposAnimacionRAW):
    Xprocessed = [] 
    XAnimacion = copy.deepcopy(XAnimacionRAW)
    TiemposAnimacion = copy.deepcopy(TiemposAnimacionRAW)
    Xaux = copy.deepcopy(XAnimacion)
    for i in range(len(TiemposAnimacion)):
        TiemposAnimacion.append(TiemposAnimacion[i])
    #Reversing the animation
    for i in range(len(Xaux)):
        a = Xaux[i][0]
        Xaux[i][0] = Xaux[i][1] 
        Xaux[i][1] = a
        #Volteamos la X de la posicion
        Xaux[i][0][0][0] = -Xaux[i][0][0][0]
        Xaux[i][1][0][0] = -Xaux[i][1][0][0]
        Xaux[i][2][0][0] = -Xaux[i][2][0][0]

        #Volteamos las rotaciones necesarias
        #cuerpo:
        for k in range(len(Xaux[i][2][1])):
            if int(Xaux[i][2][1][k]) != 0 and k != 1:
                Xaux[i][2][1][k] = -Xaux[i][2][1][k]
        #piernas
        for l in range(len(Xaux[i])-1):
            for k in range(len(Xaux[i][l][1])):
                if int(Xaux[i][l][1][k]) != 0 and k != 0:
                    Xaux[i][l][1][k] = -Xaux[i][l][1][k]     

    for i in range(len(XAnimacion)):
        XAnimacion.append(Xaux[i])
    XAux = copy.deepcopy(XAnimacion[0])
    XAnimacion.append(XAux)
    TiemposAnimacion.append(TiemposAnimacion[0])
    
    return processAnimation(Xprocessed, TiemposAnimacion)

#Actualmente se utiliza esta función para ejecutar animaciones
def playProcessedAnimation(Xprocessed):
    global canPlay
    canPlay = True
    for i in range(len(Xprocessed)):
        print(Xprocessed[i])
        print()
        com.write(str(Xprocessed[i]))
        time.sleep(1/fps)
        if not canPlay:
            break
    if not canPlay:
        print("Animación detenida")
    else:
        global ready
        ready = True
        print("Done")

def playProcessedAnimationCycle(Xprocessed):
    global canPlay
    canPlay = True
    while canPlay:
        for i in range(len(Xprocessed)):
            print(Xprocessed[i])
            print()
            com.write(str(Xprocessed[i]))
            time.sleep(1/fps)
            if not canPlay:
                break
    if not canPlay:
        print("Animación detenida")
    else:
        global ready
        ready = True
        print("Done")

#Funciona pero no se usa actualmente ya que no es ideal sobreescribir la animación.
def changeAnimationFrame(fileName, index, X, t): 
    XAnimacion, TiemposAnimacion, error = readAnimation(fileName)
    #Elimina el archivo ya existente
    open(fileName, "w").close()
    if error == None:
        #print("Longitud e indice:", len(XAnimacion), index)
        if len(XAnimacion) > index:
            XAnimacion[index] = X
            TiemposAnimacion[index] = t
        else:
            XAnimacion.append(X)
            TiemposAnimacion.append(t)
        saveAnimation(fileName, XAnimacion, TiemposAnimacion)
    return error

#Funciona y podría ser utilizada para modificar una animación, pero tiene el mismo problema que la anterior.
def addAnimationFrame(fileName, index, X, t):
    XAnimacion, TiemposAnimacion, error = readAnimation(fileName)
    #Elimina el archivo ya existente
    open(fileName, "w").close()
    if error == None:
        XAnimacion.insert(index, X)
        TiemposAnimacion.insert(index, t)
        saveAnimation(fileName, XAnimacion, TiemposAnimacion)
    return error

def stopAll():
    global canPlay
    canPlay = False

#Versión cíclica de "PlayAnimation2"
def playWalkingCycle(XAnimacion, TiemposAnimacion, XInicial, SetSliders):
    global canPlay
    canPlay = True
    X = [XInicial]
    XAnimacion.insert(0, XInicial)
    XAux = copy.deepcopy(X[0])
    IK.calcularTodo(XAux) 
    print(XAnimacion)
    started = False
    while canPlay:
        Xanim = copy.deepcopy(XAnimacion)
        Tanim = copy.deepcopy(TiemposAnimacion)
        for i in range(len(XAnimacion)-1):
            
            t = int(fps * Tanim[i])
            #DifX es la diferencia entre el valor actual de X y el valor i+1 de X
            DifX = [(Xanim[i+1][j][k][l] - Xanim[i][j][k][l])/t for j in range(len(Xanim[i])) for k in range(len(Xanim[i][j])) for l in range(len(Xanim[i][j][k]))]
            #DifX debe dividirse en t para que sea una velocidad constante
            X = Xanim[i]
            for _ in range(t):
                ini = time.time()
                for j in range(len(Xanim[i])):
                    for k in range(len(Xanim[i][j])):
                        for l in range(len(Xanim[i][j][k])):
                            X[j][k][l] = X[j][k][l] + DifX[j*6+k*3+l]
                if canPlay:
                    XAux = copy.deepcopy(X)
                    IK.calcularTodo(XAux)
                    SetSliders(X)
                    time.sleep(1/fps)
                else:
                    break
        if not started:
            started = True
            XAnimacion = XAnimacion[1:]
        #Reversing the animation
        for i in range(len(XAnimacion)):
            a = XAnimacion[i][0]
            XAnimacion[i][0] = XAnimacion[i][1] 
            XAnimacion[i][1] = a
            #Volteamos la X de la posicion
            XAnimacion[i][0][0][0] = -XAnimacion[i][0][0][0]
            XAnimacion[i][1][0][0] = -XAnimacion[i][1][0][0]
            XAnimacion[i][2][0][0] = -XAnimacion[i][2][0][0]
            #Volteamos las rotaciones necesarias
            #cuerpo:
            for k in range(len(XAnimacion[i][2][1])):
                if int(XAnimacion[i][2][1][k]) != 0 and k != 1:
                    XAnimacion[i][2][1][k] = -XAnimacion[i][2][1][k]
            #piernas
            for l in range(len(XAnimacion[i])-1):
                for k in range(len(XAnimacion[i][l][1])):
                    if int(XAnimacion[i][l][1][k]) != 0 and k != 0:
                        XAnimacion[i][l][1][k] = -XAnimacion[i][l][1][k]
    print("Animación detenida")

