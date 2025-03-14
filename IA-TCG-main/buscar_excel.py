import pandas as pd

def buscar_carta_en_excel(nombre_carta, excel_path):
    # Leer el archivo Excel
    df = pd.read_excel(excel_path)
    
    # Buscar la carta por el nombre exacto
    result = df[df['Nombre'] == nombre_carta]
    
    # Si la carta se encuentra, devolver el resultado, sino None
    if not result.empty:
        return result
    else:
        print(f"Error: La carta '{nombre_carta}' no se encuentra en la base de datos.")
        return None

if __name__ == "__main__":
    # Test para verificar si la carta está en el Excel
    resultado = buscar_carta_en_excel("Roronoa_Zoro", "BD_OPTCG.xlsx")  # Aquí puedes probar con un nombre real
    if resultado is not None:
        print("Carta encontrada:")
        print(resultado)
    else:
        print("No se encontró la carta.")
