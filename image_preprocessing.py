from PIL import Image
import cv2
import numpy as np

img = cv2.imread('input.png')

im2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
res = cv2.resize(img,None,fx=2,fy=2, interpolation = cv2.INTER_LANCZOS4)
ret,thresh = cv2.threshold(im2,127,255,cv2.THRESH_BINARY_INV)
cv2.imshow('second', thresh)
cv2.imwrite('binarization.png',thresh)

#dilation
kernel = np.ones((12,3))
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilated', img_dilation)
cv2.imwrite('dilation.png',img_dilation)
# find contours
im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = img[y:y + h, x:x + w]

    # show ROI
    # cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(img, (x, y), (x + w, y + h), (254,0,254), 1)
    # cv2.waitKey(0)

    #if w > 15 and h > 15:

    cv2.imwrite('{}.png'.format(i), roi)



cv2.imshow('marked areas',img)
cv2.imwrite('img_contouring.png',img)
cv2.waitKey(0)

"""
# text extraction logic begins from here..........
Img = Image.open('unix.jpeg')
text = pytesseract.image_to_string(Img)
print(text)

"""
