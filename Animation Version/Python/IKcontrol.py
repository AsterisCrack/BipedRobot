
import IKfunciones as IK
import IKserialCom as com

initialData = IK.getInitialData()
currentData = initialData.copy()
global previousData
previousData = initialData.copy()

def getFullLength():
    return IK.getFullLength()

def IKcuerpo(valores, sendData=True):
    #Calculo el desplazamiento de los origenes debido a la rotacion del cuerpo
    if sendData:
        print("valores: ", valores)
    O1Transformed, O2Transformed, valores[0][0], valores[1][0]= IK.aplicarRotaciones(valores[2][1], valores[0][0], valores[1][0])
    valores[0][0][0] = valores[0][0][0] + O1Transformed[0]
    valores[1][0][0] = valores[1][0][0] + O2Transformed[0]
    valores[0][0][1] = valores[0][0][1] + O1Transformed[1]
    valores[1][0][1] = valores[1][0][1] + O2Transformed[1]
    valores[0][0][2] = valores[0][0][2] + O1Transformed[2] 
    valores[1][0][2] = valores[1][0][2] + O2Transformed[2]

    valores[0][1][0] = valores[0][1][0] + valores[2][1][0]
    valores[1][1][0] = valores[1][1][0] + valores[2][1][0]
    valores[0][1][2] = valores[0][1][2] - valores[2][1][2]
    valores[1][1][2] = valores[1][1][2] - valores[2][1][2]

    #Modifico valores para calcular las kinemáticas de posición del cuerpo automaticamente
    valores[0][0][0] = valores[0][0][0] - valores[2][0][0]
    valores[1][0][0] = valores[1][0][0] - valores[2][0][0]
    valores[0][0][1] = valores[0][0][1] + valores[2][0][1]
    valores[1][0][1] = valores[1][0][1] + valores[2][0][1]
    valores[0][0][2] = valores[0][0][2] - valores[2][0][2]
    valores[1][0][2] = valores[1][0][2] - valores[2][0][2]

    return valores

def calcularTodo(valores, sendData=True):
    valores = IKcuerpo(valores, sendData)
    #Obtengo los motores para la posición deseada
    JPiernaDer = IK.getJPiernaDer(valores[0])
    JPiernaIzq = IK.getJPiernaIzq(valores[1])
    if type(JPiernaDer) == list and type(JPiernaIzq) == list:
        finalData = []
        #Creo el vector completo
        for i in range(18):
            finalData.append(initialData[i])
        
        #Le agrego los valores de los motores
        finalData[11] = JPiernaDer[0]
        finalData[12] = JPiernaDer[1]
        finalData[13] = JPiernaDer[2]
        finalData[14] = JPiernaDer[3]
        finalData[15] = JPiernaDer[4]
        finalData[8]  = JPiernaDer[5]
        finalData[3]  = JPiernaIzq[0]
        finalData[4]  = JPiernaIzq[1]
        finalData[5]  = JPiernaIzq[2]
        finalData[6]  = JPiernaIzq[3]
        finalData[7]  = JPiernaIzq[4]
        finalData[0]  = JPiernaIzq[5]

        #Agrego los desfases de rotación provocados por la rotación del cuerpo:
        finalData[3] -= valores[2][1][1]
        finalData[11] -= valores[2][1][1]

        #Limpio los datos para poder enviarlos correctamente
        global currentData
        currentData = finalData

        finalData = str(finalData).strip('[]')
        if sendData:
            com.write(finalData)
            print(finalData)
            print("\n")
        return finalData, valores
    else:
        #Si hay un error lo devuelvo para que se imprima en pantalla en vez de los valores de los motores
        return JPiernaDer if type(JPiernaDer) == str else JPiernaIzq, None

def sendRawData(valores):
    com.write(str(valores).strip('[]'))
    #print(str(valores).strip('[]'))
    #print("\n")