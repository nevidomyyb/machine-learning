import cv2
import requests
import numpy as np
from PIL import Image, ImageTk
from tkinter import Tk, Label

# url = "https://miro.medium.com/v2/resize:fit:746/1*5aL4dPHXvvaDnAQxmMRLug.png"
url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"

response = requests.get(url)
image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
imagem_colorida = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
if imagem_colorida is None:
    print("Falha!")
else:
    imagem_cinza = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2GRAY)
    _, imagem_binaria = cv2.threshold(imagem_cinza, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    #Converter para um tipo que o Tkinter pode usar.
    # imagem_cinza_rgb = cv2.cvtColor(imagem_cinza, cv2.COLOR_GRAY2RGB)
    # imagem_binaria_rgb = cv2.cvtColor(imagem_binaria, cv2.COLOR_GRAY2RGB)
    
    cv2.imwrite('imagem_cinza.jpg', imagem_cinza)
    cv2.imwrite('imagem_binaria.jpg', imagem_binaria)
    
    # root = Tk()
    # root.title("Imagens")
    
    # imagem_cinza_tk = ImageTk.PhotoImage(image=Image.fromarray(imagem_binaria_rgb))
    # label_cinza = Label(root, image=imagem_cinza_tk)
    # label_cinza.pack(side="left")
    
    # imagem_binaria_tk = ImageTk.PhotoImage(image=Image.fromarray(imagem_binaria_rgb))
    # label_binaria = Label(root, image=imagem_binaria_tk)
    # label_binaria.pack(side="right")
    
    # root.mainloop()
    