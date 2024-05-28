import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# Definir los modelos base con los mejores hiperparámetros
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]
}

param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}

rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_gb = GridSearchCV(gb, param_grid_gb, cv=5, scoring='accuracy')

grid_rf.fit(X_train, y_train)
grid_gb.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
best_gb = grid_gb.best_estimator_

# Definir el meta-modelo
meta_model = LogisticRegression()

# Definir el StackingClassifier
estimators = [
    ('rf', best_rf),
    ('gb', best_gb)
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=5
)

# Entrenar el StackingClassifier
stacking_clf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_stacking = stacking_clf.predict(X_test)

# Evaluar el StackingClassifier
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print("Stacking Classifier Accuracy:", accuracy_stacking)
print("\nStacking Classifier Classification Report:")
print(classification_report(y_test, y_pred_stacking))

# Generar y mostrar la matriz de confusión
cm = confusion_matrix(y_test, y_pred_stacking)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=stacking_clf.classes_)
disp.plot()
plt.title('Matriz de Confusión')
plt.show()
