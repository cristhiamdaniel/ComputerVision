import cv2

'''
Programa que detecta rostros en una imagen
y a traves del control de barra deslizante, 
otorga diferentes niveles de borrosidad a los otros
que esten dentro de una imagen,
ademas se puede mostrar la imagen a color o en escala de grises
'''

def nothing(x):
    pass

# Leer la imagen de entrada
img = cv2.imread('rostros.jpeg')

# Llamar el clasificador de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Especificar el nombre de la ventana
cv2.namedWindow('rostro')

# Crear los trackbars para el control de la imagen
cv2.createTrackbar('Blur','rostro',0,15,nothing)
cv2.createTrackbar('Gray','rostro',0,1,nothing)

# Obtener las posicioens de las barras de seguimiento
while True:
    val = cv2.getTrackbarPos('Blur','rostro')
    grayVal = cv2.getTrackbarPos('Gray','rostro')

    # Condicion de escalado de grises o colores
    if grayVal == 1:
        imagenN = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    else:
        imagenN = img.copy()

    # Detectar rostros y difuminarlos a diferentes niveles
    faces = face_cascade.detectMultiScale(imagenN, 1.1, 5)

    for (x,y,w,h) in faces:
        if val > 0:
            imagenN[y:y+h, x:x+w] = cv2.blur(imagenN[y:y+h, x:x+w], (val,val))

    # Mostrar la imagen
    cv2.imshow('rostro', imagenN)
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()