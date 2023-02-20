import cv2

class DetectorRostros:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detectar(self, nombre_imagen):
        # cargar la imagen
        img = cv2.imread(nombre_imagen)
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detectar rostros
        faces = self.face_cascade.detectMultiScale(gray,
                                              scaleFactor=1.1,
                                              minNeighbors=5,
                                              minSize=(30, 30),
                                              maxSize=(200, 200))
        # Dibujar rectangulos alrededor de los rostros
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # return la imagen
        return img

def main():
    detector = DetectorRostros()
    nombre = input("Introduce el nombre de la imagen: ")
    imagen = detector.detectar(nombre)
    cv2.imshow('rostro', imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()