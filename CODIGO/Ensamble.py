import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

# Cargar los datos desde un archivo o DataFrame
data = pd.read_csv(r"C:\Users\gta3_\Desktop\Master_ing_infor\TFM\CODIGO\heart_disease_health_indicators_BRFSS2015.csv")

x, y = data.drop('HeartDiseaseorAttack', axis=1), data['HeartDiseaseorAttack']
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(x, y)

# 1. split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Guardar el escalador de características para la web
joblib.dump(scaler, 'web/eScaler.joblib')

# 2. Entrenamiento de los modelos individuales
svm_model = SVC(kernel='linear', C =10, gamma= 0.1, probability=True)
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, 'web/svm.pkl')

rf_model = RandomForestClassifier(n_estimators= 200, min_samples_split= 2, min_samples_leaf= 1, max_depth = 10, bootstrap = False)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'web/randomForest.pkl')

logreg_model = LogisticRegression(C= 0.01, penalty= 'l1', solver= 'saga')
logreg_model.fit(X_train, y_train)
joblib.dump(logreg_model, 'web/logisticReggression.pkl')

nn_model = MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=575, activation='tanh', alpha=0.05, learning_rate='adaptive', solver='sgd')
nn_model.fit(X_train, y_train)
joblib.dump(nn_model, 'web/neuralNetwork.pkl')

# 3. Crear el ensamble combinando los modelos
ensemble_model = VotingClassifier(
    estimators=[('svm', svm_model), ('rf', rf_model), ('logreg', logreg_model), ('nn', nn_model)],
    voting='soft'
)
ensemble_model.fit(X_train, y_train)
joblib.dump(ensemble_model, 'web/ensamble.pkl')

# 4. Predecir los resultados
y_pred = ensemble_model.predict(X_test)

# Un único valor binario, un umbral de probabilidad
threshold = 0.5
y_pred_binary = ensemble_model.predict_proba(X_test)[:, 1] > threshold
y_prob = y_pred_binary

# Evaluar la precisión del modelo Ensamble
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Imprimir las métricas
print("Ensamble Accuracy:", accuracy)
print("Ensamble Precision:", precision)
print("Ensamble Recall:", recall)
print("Ensamble F1-score:", f1)
print("Ensamble ROC AUC score:", roc_auc)

# Realizar predicciones en los datos de prueba
y_pred = svm_model.predict(X_test)
y_prob = svm_model.predict_proba(X_test)[:, 1]

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Imprimir las métricas
print("SVM Accuracy:", accuracy)
print("SVM Precision:", precision)
print("SVM Recall:", recall)
print("SVM F1-score:", f1)
print("SVM ROC AUC score:", roc_auc)


# Realizar predicciones en los datos de prueba
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Imprimir las métricas
print("RF Accuracy:", accuracy)
print("RF Precision:", precision)
print("RF Recall:", recall)
print("RF F1-score:", f1)
print("RF ROC AUC score:", roc_auc)




# Realizar predicciones en los datos de prueba
y_pred = logreg_model.predict(X_test)
y_prob = logreg_model.predict_proba(X_test)[:, 1]

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Imprimir las métricas
print("LOGREG Accuracy:", accuracy)
print("LOGREG Precision:", precision)
print("LOGREG Recall:", recall)
print("LOGREG F1-score:", f1)
print("LOGREG ROC AUC score:", roc_auc)




# Realizar predicciones en los datos de prueba
y_pred = nn_model.predict(X_test)
y_prob = nn_model.predict_proba(X_test)[:, 1]

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Imprimir las métricas
print("NN Accuracy:", accuracy)
print("NN Precision:", precision)
print("NN Recall:", recall)
print("NN F1-score:", f1)
print("NN ROC AUC score:", roc_auc)