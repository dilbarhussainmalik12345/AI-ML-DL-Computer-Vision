import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option = {
    'model': 'cfg/yolotiny.cfg',
    'load': 10000,		#adjust load according to the ckpt yolotiny files
    'threshold': 0.2,	#adjust threshold according to the accuracy required
    'gpu': 1.0			#adjust GPU usage depending on how much you want your CPU to handle
}

tfnet = TFNet(option)

capture = cv2.VideoCapture(0)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

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
            print('leopard')
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break