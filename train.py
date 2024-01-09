import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")

# Plot histogram
sns.histplot(pd.DataFrame(y_train))
plt.title("Balanced data")
plt.savefig("balanced_data.png")

# Crea el modelo de regresión logística
model = LogisticRegression(C=1,l1_ratio=0,max_iter=100,penalty='l1',solver='saga',random_state=42,)
model.fit(X_train, y_train)

# Evalúa el rendimiento del mejor modelo en el conjunto de prueba
test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy with Best Model: {test_accuracy}")
print(f"Best Parameters: {model.get_params()}")

# Obtiene informe de matrices de confusión
conf_matrix = confusion_matrix(y_test, model.predict(X_test), labels=model.classes_)

# Guarda los resultados en un archivo de texto
with open("metrics.txt", "w") as f:
    f.write(f"Test Accuracy: {test_accuracy}\n")
    f.write(f"Best Parameters: {model.get_params()}")

# Crear el contenido del README1.md
readme_content = f"## Métricas del Modelo\n\nTest Accuracy: {test_accuracy}\nBest Parameters: {model.get_params()}"

# Escribir el contenido en un nuevo archivo README.md
with open("README1.md", "w") as readme_file:
    readme_file.write(readme_content)

# Visualiza la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues, values_format=".4g")  # Visualiza la matriz de confusión
plt.savefig("confusion_matrix.png")
plt.title("Confusion Matrix")
plt.show()




