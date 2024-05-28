import requests
import csv
from datetime import datetime, timedelta

# Credenciales del sensor
user = '93I2S5UCP1ISEF4F'
password = '6552EBDADED14014B18359DB4C3B6D4B3984D0781C2545B6A33727A4BBA1E46E'

# URL y parámetros de la API
url = 'https://sensecap.seeed.cc/openapi/list_telemetry_data'
params = {
    'device_eui': '2CF7F1C051100031',
    'channel_index': 1,
    'time_start': int((datetime.now() - timedelta(days=30)).timestamp() * 1000),
    'time_end': int(datetime.now().timestamp() * 1000),
}

# Hacer la solicitud GET
response = requests.get(url, params=params, auth=(user, password))

# Imprimir la respuesta de la API (para depuración)
print("Respuesta de la API:")
print(response.text)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    try:
        data = response.json()

        # Nombre del archivo CSV
        csv_filename = 'mediciones.csv'

        # Escribir los datos en un archivo CSV
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Escribir el encabezado del CSV
            writer.writerow(['dispositivo', 'fecha', 'tipo_med', 'valor'])

            # Asumiendo que data es la lista de mediciones obtenidas de la API
            dispositivo = '2CF7F1C051100031'
            
            # Obtener la lista de tipos de medición
            tipos_med = [item[1] for item in data['data']['list'][0]]

            # Iterar sobre los canales y las mediciones dentro de cada canal
            for i, measurements in enumerate(data['data']['list'][1]):
                tipo_med = tipos_med[i]
                for measurement in measurements:
                    valor = measurement[0]
                    fecha_iso = measurement[1]
                    fecha = datetime.fromisoformat(fecha_iso.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')

                    # Escribir una fila en el CSV
                    writer.writerow([dispositivo, fecha, tipo_med, valor])

        print(f"Datos guardados en {csv_filename}")
    except KeyError as e:
        print(f"Error de clave: {e}")
        print("Respuesta JSON:", json.dumps(data, indent=2))
    except Exception as e:
        print(f"Ocurrió un error: {e}")
else:
    print(f"Error en la solicitud: {response.status_code}")
    print(response.text)
