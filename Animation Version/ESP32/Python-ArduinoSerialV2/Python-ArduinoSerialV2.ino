/* Made for ESP32 board. It might work with other boards.
 * Receives data from the COM port
 * Then processes it and writes it to the PWM controller board
 * Made by: Pablo Gómez Martínez (Asteris)
 * Version: 25/7/2022
 */
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver board1 = Adafruit_PWMServoDriver(0x40);
#define SERVOMIN  105
#define SERVOMAX  610 

const int MPU_addr=0x68; 
const int MPU_addr2=0x69;

String inputString = "";  
bool stringComplete = false;

int InstantMovement [18];

String receivedPos;
unsigned long timer;

String receivedString = "";

void setup() {
  Wire.begin();
  Wire.beginTransmission(MPU_addr); 
  Wire.beginTransmission(MPU_addr2);
  Wire.write(0x6B); Wire.write(0); 
  Wire.endTransmission(true); 
  Serial.begin(115200);
  Serial.flush();
  board1.begin();
  board1.setPWMFreq(60);  
  inputString.reserve(200); 
  Serial.println("Calibrating");
  calibrate(MPU_addr);
  calibrate(MPU_addr2);
  delay(1000);
}

void loop() {
  if (stringComplete) {
    inputString.trim();
    processCmd(inputString);
    inputString = "";
    receivedString = "";
    stringComplete = false;
  }
} 

void serialEventRun(void) {
  if (Serial.available()) serialEvent();
}
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    }
    else {
      inputString += inChar;
    }
  }
}

//Example: 27, 180, 90, 87, 90, 109, 86, 109, 62, 180, 90, 87, 93, 114, 83, 118, 45, 135, 
void processCmd(String cmd){
  if (cmd) {
      unsigned int data_num = 0;
    while(cmd.indexOf(",")!=-1){
      InstantMovement[ data_num ] = cmd.substring(0,cmd.indexOf(",")).toInt();
      data_num++;
      cmd = cmd.substring(cmd.indexOf(",")+1);
    }
    InstantMovement[ data_num ] = cmd.toInt();
    String returnPos = "";
    for (int i = 0; i <16; i++){
      while (InstantMovement[i] > 180) InstantMovement[i] = InstantMovement[i] - 181;
      //Serial.println(InstantMovement[i]);
      board1.setPWM(i, 0, map(InstantMovement[i],0, 180, SERVOMIN,SERVOMAX));
      returnPos += String(InstantMovement[i]) + ", ";
    }
  }
}
