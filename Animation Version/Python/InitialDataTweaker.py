#This script is simmilar to IK menu but simpler.
#Consists of a tkinter window with 12 sliders corresponding to each servo.
#The sliders go from 0 to 180 degrees.
#They start in their correct position.

import tkinter as tk
import IKcontrol as IK
import IKfunciones

def sendValues(*_):
    valores = []
    for i in sliders:
        valores.append(i.get())
    print("Valores: ", valores)
    IK.sendRawData(valores)

#Arduino initial data 27, 180, 90, 87, 90, 109, 86, 109, 62, 180, 90, 87, 93, 114, 83, 118, 45, 135
initialData = IKfunciones.getInitialData()

ventana = tk.Tk()
ventana.title("Initial Data Tweaker")
ventana.tk.call('tk', 'scaling', 1.5)
ventana.configure(background="#121212")
ventana.geometry("800x1080+400+100")

#Creo los sliders
sliders = []
for i in range(len(initialData)):
    slider = tk.Scale(ventana, from_=0, to=180, orient=tk.HORIZONTAL, label=str(i))
    slider.set(initialData[i])
    slider.config(command=sendValues)
    slider.config(highlightthickness=0, bd=0, length=450, sliderrelief="flat", fg="#FFFFFF", bg="#121212", activebackground="#949494", troughcolor="#323232", highlightbackground="#121212")
    slider.pack()
    sliders.append(slider)

ventana.mainloop()
