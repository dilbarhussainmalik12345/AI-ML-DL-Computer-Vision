import cv2
from picamera import PiCamera
from darkflow.net.build import TFNet
import numpy as np
import time
import serial
import RPi.GPIO as GPIO      
import os

def setup(pin):
	global BuzzerPin
	BuzzerPin = pin
	GPIO.setmode(GPIO.BOARD)       # Numbers GPIOs by physical location
	GPIO.setup(BuzzerPin, GPIO.OUT)
	GPIO.output(BuzzerPin, GPIO.HIGH)
	
def on():
	GPIO.output(BuzzerPin, GPIO.LOW)

def off():
	GPIO.output(BuzzerPin, GPIO.HIGH)
	
port = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=1)

option = {
    'model': 'cfg/yolotiny.cfg',
    'load': 10000,			#adjust load according to the ckpt yolotiny files
    'threshold': 0.2,		#adjust threshold according to the accuracy required
    ##'gpu': 1.0
}

Buzzer = 11

tfnet = TFNet(option)
capture = cv2.VideoCapture('t1.jpg')
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
while 1:
    camera = PiCamera(resolution=(1280,720), framerate=15)
    camera.rotation = 180
    time.sleep(2)
    camera.capture('t1.jpg')
    image1 = cv2.imread('t1.jpg')
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('t1.jpg', gray)  
    capture = cv2.VideoCapture('t1.jpg')
    camera.close()
    time.sleep(2)                  
    flag = 0
    while (capture.isOpened()):
        stime = time.time()
        ret, frame = capture.read()
        if ret:
            results = tfnet.return_predict(frame)
            for color, result in zip(colors, results):
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                frame = cv2.rectangle(frame, tl, br, color, 7)
                frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                flag = flag + 1
            if (flag > 0):
                print("leopard")
                port.write(b'AT\r\n')						#Disable the Echo
                rcv = port.read(10)
                print (rcv, " ", 2)
                time.sleep(1)

                port.write(b'AT+CMGF=1\r\n')				#Select Message format as Text mode 
                rcv = port.read(10)
                print (rcv, " ", 3)
                time.sleep(1)

                port.write(b'AT+CSMP=17,167,0,0\r\n')		#New SMS Message Indications
                rcv = port.read(10)
                print (rcv, " ", 4)
                time.sleep(1)
															#Sending a message to a particular Number
                port.write(b'AT+CMGS="1111111111"\r\n')		#Enter a phone number where messages can be sent to
                rcv = port.read(10)
                print (rcv)
                time.sleep(1)

                port.write(b'C1 Leopard Detected\x1A\r\n')  #Enter Message to be sent
                rcv = port.read(10)

                setup(Buzzer)
                on()
                time.sleep(3)
                off()

                        
            cv2.imshow('frame', frame)
            print('FPS {:.1f}'.format(1 / (time.time() - stime)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            capture.release()
            cv2.destroyAllWindows()
            break