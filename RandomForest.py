import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
###n_estimators = 100, max_features = 'sqrt', random_state = 42.
print (sys.version)
#Download file with coronary data

df = pd.read_csv(r"C:\Users\gta3_\Desktop\Master_ing_infor\TFM\CODIGO\heart_disease_health_indicators_BRFSS2015.csv")

# Handling missing values
data = df.fillna(df.mean())

# Creating the Random Forest Classifier model
X = data.drop(['HeartDiseaseorAttack'], axis=1)
y = data['HeartDiseaseorAttack']
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X, y)


# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Predict the vales of the test set
y_pred = rfc.predict(X_test)

# Show the confusion matrix and the classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

# Define los parámetros que deseas ajustar y los valores que deseas probar
param_grid = {
    'n_estimators': [5, 15, 25, 45, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Crea un clasificador de bosques aleatorios con el estado aleatorio fijo
rfc_rs = RandomForestClassifier(random_state=42)

# Realiza la búsqueda aleatoria
print("busqueda")
random_search = RandomizedSearchCV(
    estimator=rfc_rs,
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    random_state=42,
    n_jobs=-1
)
print("busqueda realizada")
# Entrena el modelo con la búsqueda aleatoria
print("Entrenamiento")
random_search.fit(X_train, y_train)

# Evalúa el modelo en el conjunto de pruebas
print("Reporte")
y_pred = random_search.predict(X_test)
print(classification_report(y_test, y_pred))

matrix = confusion_matrix(y_test, y_pred)

print("\n The best estimator across ALL searched params:\n", random_search.best_estimator_)
print("\n The best score across ALL searched params:\n", random_search.best_score_)
print("\n The best parameters across ALL searched params:\n", random_search.best_params_) #{'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 10, 'bootstrap': False} 
from mlxtend.plotting import plot_confusion_matrix
plot_confusion_matrix(conf_mat=matrix)
print(matrix)
print(classification_report(y_test, y_pred))
plt.show()