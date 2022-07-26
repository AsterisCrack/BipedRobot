
import serial
import time

try:
    esp32 = serial.Serial(port='COM3', baudrate=115200, timeout=0.1)
    print("Conectado al ESP32 mediante USB")
    espConnection = True
except:
    try:
        esp32 = serial.Serial(port='COM4', baudrate=115200, timeout=0.2)
        print("Conectado al ESP32 mediante Bluetooth")
        espConnection = True
    except:
        print("No se ha conectado al ESP32")
        espConnection = False
        
seHaSalido = False
def write(x):
    if not seHaSalido:
        if espConnection:
            esp32.write((str(x)+"\n").encode())
            time.sleep(0.05)           
        else:
            print("No se ha conectado al ESP32")

def exitApp():
    if espConnection:
        global seHaSalido
        seHaSalido = True
        esp32.close()
    print("\nSaliendo de la aplicación...")
        