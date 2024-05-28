import streamlit as st
import joblib
import pandas as pd

# Cargar el modelo entrenado y el scaler
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Crear la interfaz de usuario de Streamlit
st.title('Predicción de pH')

st.write('Introduce las características para predecir si es de día o de noche:')

# Crear un formulario para ingresar los datos
ph = st.number_input('pH', min_value=0.0, max_value=50.0, step=0.01)

# Botón para realizar la predicción
if st.button('Predecir'):
    features = pd.DataFrame({'ph': [ph]})
    
    # Estandarizar las características usando el scaler cargado
    features_scaled = scaler.transform(features)
    
    # Realizar la predicción
    prediction = model.predict(features_scaled)
    
    if prediction == 1:
        st.write('Es de día')
    else:
        st.write('Es de noche')

# Ejecute Streamlit desde la línea de comandos con `streamlit run app.py`
