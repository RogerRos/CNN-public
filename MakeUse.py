from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("C:/Users/roger/OneDrive/Documents/TDR/Red neuronal convulcional/model_IA-0.3")

img_path = "C:/Users/roger/OneDrive/Documents/TDR/red neuronal convulcional (aplicacio)/FOTO/download.jpg"
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255

prediction = model.predict(img_array)
if prediction[0][0] < 0.5:
    print("-------------------------------------------------------")
    print("--------------------[ es un cotxe ]--------------------")
    print("-------------------------------------------------------")
else:
    print("-------------------------------------------------------")
    print("--------------------[ es una moto ]--------------------")
    print("-------------------------------------------------------")
