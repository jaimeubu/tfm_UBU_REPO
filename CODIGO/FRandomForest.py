import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import joblib
###n_estimators = 100, max_features = 'sqrt', random_state = 42.
#Download file with coronary data
df = pd.read_csv(r"C:\Users\gta3_\Desktop\Master_ing_infor\TFM\CODIGO\heart_disease_health_indicators_BRFSS2015.csv")

# Handling missing values
data = df.fillna(df.mean())

# Creating the Random Forest Classifier model
x = data.drop(['HeartDiseaseorAttack'], axis=1)
y = data['HeartDiseaseorAttack']

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(x, y)

# split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

# Creates a random forest clasificator with {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 10, 'bootstrap': False} 
rfc = RandomForestClassifier(n_estimators= 200, min_samples_split= 2, min_samples_leaf= 1, max_depth = 10, bootstrap = False)

# Trains the model
rfc.fit(x_train, y_train)

# Evaluate the model with the test data
y_pred = rfc.predict(x_test)
print(classification_report(y_test, y_pred))

matrix = confusion_matrix(y_test, y_pred)
print(matrix)

joblib.dump(rfc, 'randomForest.pkl')