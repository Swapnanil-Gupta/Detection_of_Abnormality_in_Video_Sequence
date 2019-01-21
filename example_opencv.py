# importing packages
import cv2
import imutils

# reading image
img = cv2.imread('img/clouds.jpg')

# rgb to grayscale
grayscaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# getting the image dimensions
(h, w, d) = img.shape
print('width = {}, height = {}, depth = {}'.format(w, h, d))

# accessing individual pixels
(B, G, R) = img[200, 300]
print('R = {}, G = {}, B = {}'.format(R, G, B))

# region of interest(roi)/cropping
roi = img[100:200, 200:300]

# resizing image without keeping aspect ratio
resizedImg = cv2.resize(img, (200, 200))

# resizing image keeping aspect ratio
resizedImgAsp = imutils.resize(img, 200)

# rotating image
rotatedImg = imutils.rotate(img, -45)

# blurring an image
blurredImg = cv2.GaussianBlur(img, (11, 11), 0)

# drawing rectangle
copiedImg = img.copy()  # copying an image
cv2.rectangle(copiedImg, (100, 100), (200, 200), (0, 255, 0), 2)

# drawing circle
anotherCopy = img.copy()
cv2.circle(anotherCopy, (100, 100), 50, (0, 0, 255), 1)  # negative thickness for filled circle

# drawing a line
oneMoreCopyImg = img.copy()
cv2.line(oneMoreCopyImg, (100, 100), (250, 280), (0, 0, 255), 3)

# writing text
againAnotherCopyImg = img.copy()
cv2.putText(againAnotherCopyImg, 'Hello OpenCV', (100, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 1)

# showing image
cv2.imshow('clouds', img)
cv2.imshow('gray clouds', grayscaleImg)
cv2.imshow('ROI', roi)
cv2.imshow('resized', resizedImg)
cv2.imshow('resized aspect ratio', resizedImgAsp)
cv2.imshow('rotated', rotatedImg)
cv2.imshow('blurred', blurredImg)
cv2.imshow('rectangle', copiedImg)
cv2.imshow('circle', anotherCopy)
cv2.imshow('line', oneMoreCopyImg)
cv2.imshow('text', againAnotherCopyImg)

# wait for keypress
cv2.waitKey(0)
cv2.destroyAllWindows()