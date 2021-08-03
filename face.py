import cv2

# récuperer le flux video

cap=cv2.VideoCapture(0)

# Tester le flux vidéo

if not (cap.isOpened()):
    print("impossible d'avoir le flux vidéo")

# tester si le flux vidéo est charger 

face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# charger le classificateur xml requis

# lire chaque image du flux vidéo

while True:
    #lecture des images
    _,image=cap.read()
    # Convertion à niveau de gris
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # detection des visages
    faces=face.detectMultiScale(image_gray,1.3,5)

    # chaque image est repéré par un rectangle
    for x,y,  width,height in faces:
        cv2.rectangle(image,(x,y),(x+width,y+height),color=(255,0,0),thickness=1)

    cv2.imshow('Tialao face detection', image)

    if cv2.waitKey(1)== ord('q'):
        break

# on libere les ressources
capture.release()
cv2.destroyAllWindows()










