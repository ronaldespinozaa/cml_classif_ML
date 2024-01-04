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
model = LogisticRegression(max_iter=1000, random_state=42, penalty='elasticnet', solver='saga')

# Defina el espacio de búsqueda de hiperparámetros
param_grid = {
    'max_iter': [500, 1000, 2000, 3000],
    'C': [0.001, 0.01, 0.1, 1, 10],  # Parámetro de regularización
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # Ratio de la penalización L1 en Elastic Net
}

# Crea un objeto GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, )

# Ajusta el modelo con Grid Search
grid_search.fit(X_train, y_train)

# Obtiene el mejor modelo
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evalúa el rendimiento del mejor modelo en el conjunto de prueba
test_accuracy = best_model.score(X_test, y_test)
print(f"Test Accuracy with Best Model: {test_accuracy}")
print(f"Best Parameters: {best_params}")

# Obtiene informes de clasificación y matrices de confusión
classification_rep = classification_report(y_test, best_model.predict(X_test), labels=best_model.classes_, output_dict=True,digits=2)
conf_matrix = confusion_matrix(y_test, best_model.predict(X_test), labels=best_model.classes_)

# Imprime informe de clasificación
# .iloc[:-1, :] to exclude support
sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True,)
plt.title("Classification Report")
plt.savefig("classification_report.png")  # Guarda el informe de clasificación()

# Guarda los resultados en un archivo de texto
with open("metrics.txt", "w") as f:
    f.write(f"Test Accuracy: {test_accuracy}\n")
    f.write(f"Best Parameters: {best_params}")
    # f.write(f"Precision label_0: {round(classification_rep['0.0']['precision'],3)}\n")
    # f.write(f"Precision label_1: {round(classification_rep['1.0']['precision'],3)}\n")
    # f.write(f"Recall label_0: {round(classification_rep['0.0']['recall'],3)}\n")
    # f.write(f"Recall label_1: {round(classification_rep['1.0']['recall'],3)}\n")



# Visualiza la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=best_model.classes_)
disp.plot(cmap=plt.cm.Blues, values_format=".4g")  # Visualiza la matriz de confusión
plt.savefig("confusion_matrix.png")
plt.title("Confusion Matrix")
plt.show()



