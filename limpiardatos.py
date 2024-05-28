import pandas as pd
from datetime import datetime

# Leer el CSV
df = pd.read_csv('mediciones.csv')

# Imprimir las primeras filas del DataFrame para verificar que se leyeron correctamente
print("Datos originales:")
print(df.head())

# Filtrar las filas que corresponden a las mediciones de pH
df_ph = df[df['tipo_med'] == 4097]  # Suponiendo que el código '4097' corresponde a mediciones de pH

# Imprimir las primeras filas del DataFrame filtrado
print("\nMediciones de pH:")
print(df_ph.head())

# Convertir la columna 'fecha' a formato datetime
df_ph['fecha'] = pd.to_datetime(df_ph['fecha'])

# Imprimir las primeras filas del DataFrame con fechas convertidas
print("\nMediciones de pH con fechas convertidas:")
print(df_ph.head())

# Extraer la hora de las mediciones
df_ph['hora'] = df_ph['fecha'].dt.hour

# Imprimir las primeras filas del DataFrame con la hora extraída
print("\nMediciones de pH con hora extraída:")
print(df_ph.head())

# Etiquetar las mediciones como 'día' o 'noche'
# Supongamos que consideramos día de 6 AM a 6 PM
df_ph['es_dia'] = df_ph['hora'].apply(lambda x: 1 if 6 <= x < 18 else 0)

# Imprimir las primeras filas del DataFrame con la etiqueta de día/noche
print("\nMediciones de pH con etiquetas de día/noche:")
print(df_ph.head())

# Seleccionar solo las columnas relevantes
df_ph = df_ph[['valor', 'es_dia']]
df_ph.columns = ['ph', 'es_dia']

# Imprimir las primeras filas del DataFrame final
print("\nDataFrame final:")
print(df_ph.head())

# Guardar el DataFrame final en un nuevo archivo CSV

df_ph.to_csv('mediciones_ph.csv', index=False)
print("Datos guardados en mediciones_ph.csv")

