import os
import cv2
from matplotlib import pyplot as plt

'''
For some reason, Jupyter Notebook/SageMaker doesn't like the cv.imshow() method. It seems to freeze the notebook and kill the kernel. I found a solution online that basically uses the matplotlib module's pyplot function to display the desired image(s) on a graph.
'''

test_video_name = 'sock_example.avi'

cap = cv2.VideoCapture(test_video_name)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    # the image is loaded in BGR by default, want to convert it to RGB
    plt.imshow(cv2.cvtColor(fgmask, cv2.COLOR_BGR2RGB))
    plt.show()

# the code below is what you could use if cv2.imshow() function works in your python IDE
#     cv2.imshow('frame', fgmask)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
        
cap.release()
cv2.destroyAllWindows()