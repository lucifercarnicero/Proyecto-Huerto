import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Leer el DataFrame final
df = pd.read_csv('./mediciones_ph.csv')

# Separar las características (X) y la variable objetivo (y)
X = df[['ph']]  # Aquí podrías agregar más características si tienes
y = df['es_dia']

# Estandarizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir el modelo de Random Forest
rf_model = RandomForestClassifier(random_state=42)

# Definir los hiperparámetros a ajustar
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Configurar la búsqueda de hiperparámetros
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Entrenar el modelo con búsqueda de hiperparámetros
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo encontrado por GridSearchCV
best_rf_model = grid_search.best_estimator_

# Predecir en el conjunto de prueba
y_pred = best_rf_model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print("Best Random Forest Accuracy:", accuracy)  # Aquí se imprime la precisión
print("\nBest Random Forest Classification Report:")
print(classification_report(y_test, y_pred))

# Mostrar los mejores hiperparámetros
print("Best Hyperparameters:", grid_search.best_params_)

# Guardar el modelo entrenado y el scaler
joblib.dump(best_rf_model, 'best_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
