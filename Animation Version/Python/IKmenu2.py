
#La interfaz consiste de 6 sliders por columna, tres columnas, y una cuarta con botones
#El esquema de color es modo oscuro

import tkinter as tk
import IKcontrol as IK
import IKserialCom
import AnimationHandler as animations
import sys
from pathvalidate import sanitize_filename
import copy
import os.path

labels = ["X", "Y", "Z", "Rx", "Ry", "Rz"]

#Altura inicial del robot. Es más baja que la altura de base para mejor estabilidad.
defaultHeight = -50
defaultSliderValues1 = [0,IK.getFullLength(),0,0,0,0]
defaultSliderValues2 = [0,IK.getFullLength(),0,0,0,0]
defaultSliderValues3 = [0,defaultHeight,0,0,0,0]
#Lista de animaciones que deben ser ejecutadas en modo cíclico
CYCLE_LIST = ["animaciones/walk2.txt", "animaciones/contento.txt","animaciones/fun.txt","animaciones/andar.txt"]

def popupNotification(title, msg, confirmFunction=None, cancelFunction=None, textInput=False):
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 4.0)
    root.configure(background="#121212")
    #titulo de la ventana
    root.title(title)
    #La ventana es de 800 x 500 y aparece centrada
    root.geometry("400x300+540+200")
    label = tk.Label(root, text=msg)
    label.pack(side="top", fill="x", pady=50)
    entry = tk.Entry(root, width=30, justify="center")

    if textInput:
        entry.pack(padx=30, pady=5)
        command = lambda:[confirmFunction(confirmFunctionGetEntry(entry, root))]
    else:
        command = lambda:[root.destroy(), confirmFunction()]
    B1 = tk.Button(root, text="Confirmar", command = command) if confirmFunction != None else tk.Button(root, text="Confirmar", command = root.destroy)
    B2 = tk.Button(root, text="Cancelar", command = lambda:[root.destroy(), cancelFunction()]) if cancelFunction != None else tk.Button(root, text="Cancelar", command = root.destroy)
    B1.pack(side="right", padx=30, pady=5)
    B2.pack(side="left", padx=30, pady=5)
    root.mainloop()

def confirmFunctionGetEntry(entry, root):
    entrada = entry.get()
    root.destroy()
    return entrada

def get_values(*_):
    valores = [sliderPiernaDerecha.get_slider_values(), sliderPiernaIzquierda.get_slider_values(), sliderCuerpo.get_slider_values()]
    valoresAux = copy.deepcopy(valores)
    #print(valores)
    resultado, nuevosValores = IK.calcularTodo(valoresAux)
    etiqueta3.config(text=resultado)
    etiqueta3.grid(row=9, column=0, pady=40, columnspan=4)
    return valores

def get_raw_values():
    return [sliderPiernaDerecha.get_slider_values(), sliderPiernaIzquierda.get_slider_values(), sliderCuerpo.get_slider_values()]

def get_values_auto(_=None, tipo=None):
    try:
        if automatico.get() == 1:
            get_values(tipo)
    except:
        pass

def SetSlidersVisually(valores):
    Sliders = [sliderPiernaDerecha, sliderPiernaIzquierda, sliderCuerpo]
    for i in range(len(Sliders)):
        Sliders[i].set_sliders(valores[i])

def exitApp(): 
    animations.stopAll()
    ventana.destroy()
    IKserialCom.exitApp()
    sys.exit()

def resetPosition():
    sliderPiernaDerecha.reset_sliders()
    sliderPiernaIzquierda.reset_sliders()
    sliderCuerpo.reset_sliders()
    get_values_auto(None, 0)

#La clase Sliders crea una columa de sliders, con sus respectivas posiciones iniciales y etiquetas
#Además, contiene algunas funciones necesarias para interactuar con ellas
class Sliders:
    def __init__(self, column, default_values, label_text, get_values_function):
        self.column = column
        self.default_values = default_values
        self.label_text = label_text
        self.get_values_function = get_values_function
        self.sliders = []
        #Creo los 6 sliders
        for i in range(6):
            self.sliders.append(tk.Scale(ventana, from_=-200, to=200, orient=tk.HORIZONTAL, label=labels[i]))
            self.sliders[i].config(bg="#121212", activebackground="#949494", troughcolor="#323232", highlightbackground="#121212")
            #El color del texto de los sliders es blanco
            self.sliders[i].config(fg="#FFFFFF")
            #slider length es 100 pixeles y sliderrelief es flat
            self.sliders[i].config(length=350, sliderrelief="flat")
            #el highlightthickness es 0
            self.sliders[i].config(highlightthickness=0, bd=0)
            #el comando de los sliders es get_values
            self.sliders[i].config(command=self.get_values_aux)
            if column != 2 and i == 1:
                self.sliders[i].config(from_=-50, to=IK.getFullLength()-defaultHeight)
            self.sliders[i].set(self.default_values[i])
        if column == 2:
            self.sliders[1].config(from_=-200, to=0)
        self.main_label = tk.Label(ventana, text=label_text)
        self.position_label = tk.Label(ventana, text="Posición")
        self.rotation_label = tk.Label(ventana, text="Rotación")

    def get_values_aux(self, valor=None):
        self.get_values_function(None, self.column)

    def place(self):
        #Initial row sirve para bajar toda la columna el número de filas que sean necesarias
        initialRow = 0
        extra_index = initialRow+2
        for slider in self.sliders:
            if self.sliders.index(slider) == 3:
                extra_index += 1
            slider.grid(row=self.sliders.index(slider)+extra_index, column=self.column, padx=15)
        self.main_label.grid(row=initialRow+0, column=self.column, pady=20)
        self.position_label.grid(row=initialRow+1, column=self.column, pady=10)
        self.rotation_label.grid(row=initialRow+5, column=self.column, pady=10)
    
    def get_slider_values(self):
        values = []
        for i in range(2):
            values.append([])
            for j in range(3):
                values[i].append(self.sliders[i*3+j].get())
        return values

    def reset_sliders(self):
        for slider in self.sliders:
            slider.set(self.default_values[self.sliders.index(slider)])

    def set_sliders(self, values):
        values = values[0]+values[1]
        for slider in self.sliders:
            slider.set(values[self.sliders.index(slider)])

#La clase NewAnimationButtons crea los botones del menú de creación de animaciones.
#Además, contiene algunas funciones necesarias para interactuar con ellos
class NewAnimationButtons:
    def __init__(self, ventana, placeOriginalButtons, placeOriginalButtonsInAnimationMenu, hideOriginalButtons, get_values, get_values_auto, Sliders):
        #Contador de frames, input de texto para tiempo en segundos, guardar frame, guardar animación, fotograma anterior y posterior, volver.
        self.SaveFrameButton = tk.Button(ventana, text="Guardar fotograma", command=self.saveFrame)
        self.SaveAnimationButton = tk.Button(ventana, text="Guardar animación", command=self.guardarAnimacion)
        self.PreviousFrameButton = tk.Button(ventana, text="<<", command=self.backFrame)
        self.NextFrameButton = tk.Button(ventana, text=">>", command=self.forewardFrame)
        self.BackButton = tk.Button(ventana, text="Volver", command=self.hideAll)
        self.TimeInput = tk.Entry(ventana, width=10, justify="center")
        self.TimeInput.insert(0,"1")
        self.FrameIndicator = tk.Label(ventana, text="Fotograma 0")
        #El resto de los botones se reutilizan de los ya existentes en el menú original

        self.ventana = ventana
        self.placeOriginalButtonsInAnimationMenu = placeOriginalButtonsInAnimationMenu
        self.placeOriginalButtons = placeOriginalButtons
        self.hideOriginalButtons = hideOriginalButtons
        self.get_values = get_values
        self.get_values_auto = get_values_auto
        self.Sliders = Sliders
        
    def placeAll(self):
        #Variables globales necesarias para guardar la animación
        self.CurrentFrame = 0
        self.Frames = []
        self.Times = []
        self.openedAnimation = None

        #Recoloco los botones originales necesaarios
        self.hideOriginalButtons()
        self.placeOriginalButtonsInAnimationMenu()
        #Coloco los nuevos botones
        initialRow = 1
        self.FrameIndicator.grid(row=initialRow+1, column=3, columnspan=2, pady=10)
        self.TimeInput.grid(row=initialRow+2, column=3, columnspan=2, pady=10)
        self.SaveFrameButton.grid(row=initialRow+3, column=3, columnspan=2, pady=10)
        self.PreviousFrameButton.grid(row=initialRow+2, column=3, columnspan=1, pady=10)
        self.NextFrameButton.grid(row=initialRow+2, column=4, columnspan=1, pady=10)
        self.SaveAnimationButton.grid(row=initialRow+6, column=3, columnspan=2, pady=10)
        self.BackButton.grid(row=initialRow+7, column=3, columnspan=2, pady=10)

    def hideAll(self):
        self.hideOriginalButtons()
        self.placeOriginalButtons()
        self.FrameIndicator.grid_remove()
        self.TimeInput.grid_remove()
        self.SaveFrameButton.grid_remove()
        self.PreviousFrameButton.grid_remove()
        self.NextFrameButton.grid_remove()
        self.SaveAnimationButton.grid_remove()
        self.BackButton.grid_remove()

    def updateFrame(self):
        #Cambio el fotograma actual
        self.FrameIndicator.config(text="Fotograma " + str(self.CurrentFrame))
        #Recoloco los sliders
        for i in range(3):
            if self.CurrentFrame == len(self.Frames):
                #Como este frame no está en la lista, coloco los sliders como el último frame guardado
                self.Sliders[i].set_sliders(self.Frames[-1][i])
            else:
                self.Sliders[i].set_sliders(self.Frames[self.CurrentFrame][i])
        self.get_values_auto()

    def backFrame(self):
        if self.CurrentFrame > 0:
            self.CurrentFrame -= 1
            self.updateFrame()
    def forewardFrame(self):
        if self.CurrentFrame < len(self.Times):
            self.CurrentFrame += 1
            self.updateFrame()

    def saveFrame(self):
        if self.CurrentFrame == len(self.Times):
            self.Times.append(self.TimeInput.get())
            self.Frames.append(self.get_values())
        else:
            self.Frames[self.CurrentFrame] =  self.get_values()
            self.Times[self.CurrentFrame] = self.TimeInput.get()
        self.CurrentFrame += 1
        self.updateFrame()

    def guardarAnimacion(self):
        #A la hora de guardar abro una nueva ventana para que el usuario pueda elegir el nombre de la animación 
        #O si desean sobreescribir (Si hay una ya existente abierta)
        if not self.openedAnimation:
            popupNotification("Archivo Animación", "Introduce nombre de animación:", self.guardarAnimConfirmFunct, None, True)
        else:
            popupNotification("Sobreescribir Animación", "Deseas sobreescribir el archivo "+self.openedAnimation+"?", self.guardarAnimConfirmFunct, None, False)
    
    def guardarAnimConfirmFunct(self, entrada=None):
        if not self.openedAnimation:
            #Limpio el nombre del archivo
            entrada = cleanFileName(entrada)
            if os.path.exists(entrada):
                popupNotification("Archivo Animación", "Archivo ya existente\nIntroduce un nombre de animación nuevo:", self.guardarAnimConfirmFunct, None, True)
                return
        else: 
            entrada = self.openedAnimation
    
        #Proceso y guardo la animación
        if entrada in CYCLE_LIST:
            Xprocesado = animations.processAnimationCycle(self.Frames, self.Times)
        else:
            Xprocesado = animations.processAnimation(self.Frames, self.Times)
        animations.saveAnimation(entrada, self.Frames, self.Times)
        animations.saveProcessedAnimation(entrada, Xprocesado)

    def openExistentAnim(self):
        popupNotification("Archivo Animación", "Introduce nombre de animación existente:", self.abrirAnimConfirmFunct, None, True)
    
    def abrirAnimConfirmFunct(self, entrada):
        entrada = cleanFileName(entrada)
        if not os.path.exists(entrada):
            popupNotification("Archivo Animación", "No se ha encontrado el archivo\nIntroduce nombre de animación existente:", self.abrirAnimConfirmFunct, None, True)
        else:
            self.placeAll()
            self.openedAnimation = entrada
            self.Frames, self.Times, error = animations.readAnimation(entrada)
            self.CurrentFrame = len(self.Times)
            self.updateFrame()
            if error != None:
                popupNotification("Archivo Animación", error+"\nIntroduce nombre de animación existente:", self.abrirAnimConfirmFunct, self.hideAll, True)

#La clase PlayAnimations crea los botones del menú reproducción de animaciones.
#Además, contiene algunas funciones necesarias para interactuar con ellos
class PlayAnimations:
    def __init__(self,ventana, placeOriginalButtons, placeOriginalButtonsInPlayMenu, hideOriginalButtons):
        self.ventana = ventana
        self.placeOriginalButtons = placeOriginalButtons
        self.placeOriginalButtonsInPlayMenu = placeOriginalButtonsInPlayMenu
        self.hideOriginalButtons = hideOriginalButtons

        self.DisponiblesLabel = tk.Label(self.ventana, text="Animaciones Disponibles")
        self.StopButton = tk.Button(self.ventana, text="Stop", command=self.stopAnimation)
        self.BackButton = tk.Button(self.ventana, text="Volver", command=self.hideAll) #Hecho
        self.scrollbar = tk.Scrollbar(self.ventana)
        self.listbox = tk.Listbox(self.ventana, yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.listbox.yview)
        #Le atribuimos el evento de clikarlo
        self.listbox.bind("<<ListboxSelect>>",self.reproducirAnimacion)

    def fillListbox(self):
        self.listbox.delete(0, "end")
        path = "animaciones/"
        files=[]
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path,file)):
                file = file.split(".")
                if file[1] == "txt" and not "PROCESSED" in file[0]:
                    files.append(file[0])
        for file in files:
            self.listbox.insert("end", file)
        
    def placeAll(self):
        #Rellenamos la lista
        self.fillListbox()
        self.hideOriginalButtons()
        self.placeOriginalButtonsInPlayMenu()
        initialRow = 1
        self.listbox.grid(row=initialRow+0,column=3,padx=20, pady=0, columnspan=2, rowspan=5)
        self.StopButton.grid(row=initialRow+5,column=3,padx=20, pady=0, columnspan=2)
        self.BackButton.grid(row=initialRow+8,column=3,padx=20, pady=0, columnspan=2)

    def hideAll(self):
        self.hideOriginalButtons() 
        self.placeOriginalButtons()
        self.scrollbar.grid_remove()
        self.listbox.grid_remove()
        self.BackButton.grid_remove()
    
    def reproducirAnimacion(self, *_):
        animacion = self.listbox.curselection()
        animacion = "animaciones/"+self.listbox.get(animacion)+".txt"
        Xprocesado = animations.readProcessedAnimation(animacion)
        Xinicial = get_raw_values()
        if animacion in CYCLE_LIST:
            animations.startAnimation(Xprocesado, None, Xinicial, SetSlidersVisually,"ciclo_procesada")
        else:
            animations.startAnimation(Xprocesado, None, Xinicial, SetSlidersVisually,"procesada")
    
    def stopAnimation(self):
        animations.stopAll()


def placeOriginalButtons():
    initialRow = 2
    autoEnviar.grid(row=initialRow,column=3,padx=20, pady=10, columnspan=2)
    boton.grid(row=initialRow+1,column=3,padx=20, pady=10, columnspan=2)
    boton3.grid(row=initialRow+2,column=3,padx=20, pady=10, columnspan=2)
    boton4.grid(row=initialRow+3,column=3,padx=20, pady=10, columnspan=2)
    boton5.grid(row=initialRow+4,column=3,padx=20, pady=10, columnspan=2)
    boton6.grid(row=initialRow+5,column=3,padx=20, pady=10, columnspan=2)
    boton2.grid(row=initialRow+6,column=3,padx=20, pady=10, columnspan=2)

def placeOriginalButtonsInAnimationMenu():
    initialRow = 1
    autoEnviar.grid(row=initialRow+0,column=3,padx=20, pady=10, columnspan=2)
    boton.grid(row=initialRow+4,column=3,padx=20, pady=10, columnspan=2)
    boton3.grid(row=initialRow+5,column=3,padx=20, pady=10, columnspan=2)
    boton2.grid(row=initialRow+8,column=3,padx=20, pady=10, columnspan=2)

def placeOriginalButtonsInPlayMenu():
    initialRow = 1
    autoEnviar.grid(row=initialRow-1,column=3,padx=20, pady=0, columnspan=2)
    boton.grid(row=initialRow+7,column=3,padx=20, pady=0, columnspan=2)
    boton3.grid(row=initialRow+6,column=3,padx=20, pady=0, columnspan=2)
    boton2.grid(row=initialRow+9,column=3,padx=20, pady=0, columnspan=2)

def hideOriginalButtons():
    autoEnviar.grid_remove()
    boton.grid_remove()
    boton3.grid_remove()
    boton4.grid_remove()
    boton5.grid_remove()
    boton6.grid_remove()
    boton2.grid_remove()

def cleanFileName(entrada):
    sanitize_filename(entrada)
    entrada = entrada.split(".")
    if "PROCESSED" in entrada[0]:
        entrada[0].replace("PROCESSED_","")
    entrada = "animaciones/"+entrada[0]+".txt"   
    return entrada

ventana = tk.Tk()
ventana.tk.call('tk', 'scaling', 2.0)
ventana.configure(background="#121212")
#titulo de la ventana
ventana.title("IK pierna V0")
#La ventana es de 800 x 500 y aparece centrada
ventana.geometry("1500x768+0+0")

#Creo los sliders
sliderPiernaDerecha = Sliders(0, defaultSliderValues1, "Pierna Derecha", get_values_auto)
sliderPiernaDerecha.place()
sliderPiernaIzquierda = Sliders(1, defaultSliderValues2, "Pierna Izquierda", get_values_auto)
sliderPiernaIzquierda.place()
sliderCuerpo = Sliders(2, defaultSliderValues3, "Cuerpo", get_values_auto)
sliderCuerpo.place()

#creo los botones de animaciones
newAnimationButtons = NewAnimationButtons(ventana, placeOriginalButtons, placeOriginalButtonsInAnimationMenu, hideOriginalButtons, get_values, get_values_auto, [sliderPiernaDerecha, sliderPiernaIzquierda, sliderCuerpo])
playAnimationButtons = PlayAnimations(ventana, placeOriginalButtons, placeOriginalButtonsInPlayMenu, hideOriginalButtons)

#Mete los valores de los sliders en una lista
valores = []
if not 'etiqueta3' in locals():
    etiqueta3 = tk.Label(ventana, text="")

#Crea un checkbox para activar el modo automatico
#Creo una variable booleana para el checkbox
automatico = tk.IntVar()
autoEnviar = tk.Checkbutton(ventana, text="Actualizar automáticamente", variable=automatico)
automatico.set(1)

boton = tk.Button(ventana, text="Calcular", command=lambda: get_values(0))
boton3 = tk.Button(ventana, text="Reiniciar", command=resetPosition)
boton4 = tk.Button(ventana, text="Nueva Animación", command=newAnimationButtons.placeAll)
boton5 = tk.Button(ventana, text="Editar Animación", command=newAnimationButtons.openExistentAnim)
boton6 = tk.Button(ventana, text="Reproducir Animación", command=playAnimationButtons.placeAll)
boton2 = tk.Button(ventana, text="Salir", command=lambda: popupNotification("Cerrando", "Cerrando...", exitApp))

#Coloco los botones en la ventana
placeOriginalButtons()

ventana.mainloop()
