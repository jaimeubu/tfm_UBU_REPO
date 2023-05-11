import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# Cargar los datos desde un archivo o DataFrame
data = pd.read_csv(r"C:\Users\gta3_\Desktop\Master_ing_infor\TFM\CODIGO\heart_disease_health_indicators_BRFSS2015.csv")

x, y = data.drop('HeartDiseaseorAttack', axis=1), data['HeartDiseaseorAttack']
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(x, y)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Dividir los datos en un conjunto de entrenamiento y prueba
#X_train, X_test, y_train, y_test = train_test_split(data.drop('HeartDiseaseorAttack', axis=1), 
#                                                    data['HeartDiseaseorAttack'], 
#                                                    test_size=0.3, 
#                                                    random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Guardar el escalador de características para la web
joblib.dump(scaler, 'svmScaler.joblib')

# Crear un modelo SVM
model = SVC(kernel='linear', C =10, gamma= 0.1)

# Ajustar el modelo a los datos de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred = model.predict(X_test)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

print(f'Precisión del modelo SVM: {accuracy:.4f}')

matrix = confusion_matrix(y_test, y_pred)
print(matrix)

joblib.dump(model, 'svm.pkl')