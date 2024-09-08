import cv2
import argparse
import time


# Funzione per identificare la forma basata sui contorni
def detect_shape(contour):
    # Approssima il contorno a un poligono
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Riconosce la forma in base al numero di vertici
    if len(approx) == 3:
        return "Triangolo"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspectRatio = w / float(h)
        # Se il rapporto di aspetto è vicino a 1, è un quadrato
        if 0.95 <= aspectRatio <= 1.05:
            return "Quadrato"
        else:
            return "Rettangolo"
    elif len(approx) > 5:
        return "Cerchio"
    else:
        return "Forma sconosciuta"


# Funzione principale per processare l'immagine
def process_image(image_path):
    # Carica l'immagine
    image = cv2.imread(image_path)
    if image is None:
        print("Error")
        return

    # Pre-elaborazione: conversione in scala di grigi, sfocatura e rilevamento dei bordi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Trova i contorni
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Per ogni contorno, rileva la forma
    for contour in contours:
        shape = detect_shape(contour)
        # Ottieni il rettangolo delimitatore e disegna il contorno
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        # Posiziona il nome della forma rilevata sull'immagine
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(image, shape, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostra l'immagine con le forme rilevate
    cv2.imshow("Shape Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Gestione della linea di comando
if __name__ == "__main__":
    # Crea il parser degli argomenti
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Image path")

    # Parsing degli argomenti
    args = parser.parse_args()

    # Esegui l'analisi sull'immagine fornita
    process_image(args.image)


    cv2.imshow("Shape Detection", image)
    cv2.waitKey(0)  # Attendi che l'utente prema un tasto
    cv2.destroyAllWindows()
    time.sleep(1)
