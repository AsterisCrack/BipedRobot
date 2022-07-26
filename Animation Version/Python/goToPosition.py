#A tkinter menu consisting on a input box and a button.
#When you press the button you take the data from the input box and send it over to the serial
#This script is used to move the robot to a certain position, useful to test animations frame by frame.
import tkinter as tk
import IKserialCom as com

ventana = tk.Tk()
ventana.tk.call('tk', 'scaling', 2.0)
ventana.configure(background="#121212")
ventana.title("Go to position")
ventana.geometry("400x200")

entrada = tk.Entry(ventana, width=30)
entrada.pack(pady=10)

enviar = tk.Button(ventana, text="Send", command=lambda: com.write(entrada.get()))
enviar.pack(pady=10)

ventana.mainloop()