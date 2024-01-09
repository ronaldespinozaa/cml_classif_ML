import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
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

# Create the model
model = RandomForestClassifier(n_estimators=500,criterion= 'entropy',max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features=None,random_state=42,n_jobs=-1)
model.fit(X_train, y_train)

# Get results   
feature_importance = model.feature_importances_
params = model.get_params()

# Evaluate model
test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy with Best Model: {test_accuracy}")
print(f"Best Parameters: {params}")

# Get confusion matrix
conf_matrix = confusion_matrix(y_test,model.predict(X_test), labels=model.classes_)

# Saving results
with open("metrics.txt", "w") as f:
    f.write(f"Test Accuracy: {test_accuracy}\n")
    f.write(f"Best Parameters: {params}")

# Visualizate the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues, values_format=".4g")  # Visualiza la matriz de confusión
plt.savefig("confusion_matrix.png")
plt.title("Confusion Matrix")
plt.show()

# Crear el contenido del README1.md
readme_content = f"## Métricas del Modelo\n\nTest Accuracy: {test_accuracy}\nBest Parameters: {best_params}"

# Escribir el contenido en un nuevo archivo README.md
with open("README1.md", "w") as readme_file:
    readme_file.write(readme_content)

# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(model.feature_importances_, index=pd.DataFrame(X_train,).columns).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar()
plt.savefig("feature_importance.png")
plt.show();




