# BipedRobot
## An ESP32 + Python controlled biped robot powered by servo motors.

This is a long term project of mine that has an indefinite goal. 
For now, I have created a program in python that allows to create and reproduce animations. Most robots like this are animated motor by motor, but this one can be animated like in modern 3D animations, by end effector placement with Inverse Kinematics.
To execute it, put all python files in a folder and execute IKmenu2.py
You might need to install some libraries for it to work.

There are plenty of configuration option in the program so feel free to change any code you like.
I am no professional and there might be some errors or bad coding.

### The Build
If you wish to build the robot, for this first version I used an ESP32 and an Adafruit 16-Channel 12-bit PWM/Servo Driver as main boards. My motors are x10 RDS3225 servos and x2 Mg996r servos for the hips. You can access all the 3D models which you will need to print for the build. The 3D model "Pie" is just one foot, so you will need to print another mirrored version. You will also need x2 skateboard bearings to strengthen the hips. You might as well want to install 1 to 3 MPU6050 IMU sensors for future versions, although they have no use for now.

The servos are attached to the PWM servo board like this:

![servosArray](https://user-images.githubusercontent.com/94953985/181059536-53a6e7fc-049e-45b9-a8f9-d8276e9c9a49.jpg)

To power everything I use a 500 Watt computer power supply, modified to be easier to use and to just access the 5V channel.
I will evenctually post my circuit and other things here, so stay tunned for that!

### The Programm
![Captura de pantalla (243)](https://user-images.githubusercontent.com/94953985/181065606-bc7bb7a0-8083-4943-b7bf-8ec96f9f9c3f.png)

The first step is deciding weather you want the robot to work wirelessly or by cable. Install whatever program you decide to your ESP32 board. But first, you might want to check for pin configuartion just in case you haven't placed them as mine.
Next, create a folder and place all Python files in it. You need to have two folders inside it as well named "animaciones" and "PROCESSED_animaciones". These folders contain the saved animations. I have left a simple walking animation inside.
You might want to adjust some values before using it. To change default servo values execute InitialDataTweaker.py and change them until happy. Then, copy the resulting value form the console to the variable named "initialData" in line 22 in IKfunciones.py
In IKfunciones.py there are also some values you might want to change, which are the distances between the servos, very important for the Inverse Kinematics.

### Inverse Kinematics
The Inverse Kinematics calculations were done by myself and you can check them in IKcontrol.py and IKfunciones.py
They are not the typical Inverse Kinematics for robotics due to the strange motor configuration. I didn't  find any solution so I did it for myself. If you ran into here for the same reason, feel free to take my code and calculations. 
I might also evenctually upload a formal demonstration of the math if anyone is interested.
