import cv2
import numpy as np
from picamera2 import Picamera2


# Lower red range
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])

# Upper red range
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

#cap = cv2.VideoCapture(0)

picam2=Picamera2()
picam2.configure(picam2.create_preview_configuration(main={'format':'XRGB8888'}))
picam2.start()

while True:
   
    frame=picam2.capture_array()
     
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    

    

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2  # Combine both masks

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("result",result)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Ignore small areas (noise)
            x, y, w, h = cv2.boundingRect(contour)
            print(x,y,w,h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green box

    cv2.imshow("imgwithboundingbox", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()




