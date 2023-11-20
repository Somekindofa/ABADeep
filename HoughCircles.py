#%%
import cv2
import numpy as np

def detect_face(image):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces



#%%
# Load an image
image = cv2.imread('personne qui regarde tout droit.jpg')

#%%
faces = detect_face(image)



#%%
# Assuming only one face is detected
(x, y, w, h) = faces[0]

# Extract the region of interest (ROI) - the face
face_roi = image[y:y+h, x:x+w]



#%%
# Apply a contour extraction method (e.g., Canny edge detection)
edges = cv2.Canny(face_roi, 30, 100)

# print out the edges
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Now we need a pipeline to detect the eyes. 
# For that, the literature suggests to use the Hough transform.
# The Hough transform is a feature extraction method 
# that is used to detect simple shapes such as circles, lines, etc.
# First, we need to convert the image to grayscale
gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

#%%
# Then, we apply the Hough transform to detect circles
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

#%%
# Assuming only one circle is detected
(x, y, r) = circles[0][0].astype("int")

#%%
# Draw the circle on the original image
cv2.circle(face_roi, (x, y), r, (0, 255, 0), 4)

#%%
# Show the image
cv2.imshow('Image', face_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()


    

