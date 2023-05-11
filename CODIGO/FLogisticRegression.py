import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# Cargar los datos desde un archivo o DataFrame
data = pd.read_csv(r"C:\Users\gta3_\Desktop\Master_ing_infor\TFM\CODIGO\heart_disease_health_indicators_BRFSS2015.csv")

x, y = data.drop('HeartDiseaseorAttack', axis=1), data['HeartDiseaseorAttack']
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(x, y)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)


# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Guardar el escalador de características para la web
joblib.dump(scaler, 'lRScaler.joblib')


# Crear un modelo de regresión logística
model = LogisticRegression()

# Ajustar el modelo a los datos de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred = model.predict(X_test)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

print(f'Precisión del modelo de regresión logística: {accuracy:.4f}')

from sklearn import metrics

cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

joblib.dump(model, 'logisticReggression.pkl')