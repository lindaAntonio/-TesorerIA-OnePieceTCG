import pandas as pd

from predecir import predecir_carta
from buscar_excel import buscar_carta_en_excel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():
    modelo = "modelo_cartas.h5"
    imagen = "dataset_OPTCG\imagen_prueba.png"
    excel_path = "BD_OPTCG.xlsx"
    
    df =  pd.read_excel(excel_path)
    model = load_model(modelo)

    datagen = ImageDataGenerator(rescale=1.0/255)
    train_generator = datagen.flow_from_directory(
        "cards",  # Directorio del dataset
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    class_indices = train_generator.class_indices

    # Predicción
    carta_predicha = predecir_carta(modelo, imagen, class_indices)
    print(f"Carta predicha: {carta_predicha}")

    # Buscar en el Excel
    resultado = buscar_carta_en_excel(carta_predicha, excel_path)
    if resultado is not None:
        print("Información encontrada:")
        print(resultado)
    else:
        print("La carta no se encuentra en la base de datos.")

if __name__ == "__main__":
    main()
