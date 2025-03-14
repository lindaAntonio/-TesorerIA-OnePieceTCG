import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predecir_carta(model_path, image_path, class_indices):
    model = load_model(model_path)

    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_label = list(class_indices.keys())[class_idx]

    return class_label

#if __name__ == "__main__":
    # Clase de ejemplo
 #   class_indices = {'Carta_A': 0, 'Carta_B': 1, 'Carta_C': 2}  # Reemplazar con los reales
  #  prediccion = predecir_carta("modelo_cartas.h5", "imagen_prueba", class_indices)
   # print(f"La carta predicha es: {prediccion}")
