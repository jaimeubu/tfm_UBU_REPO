import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
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

# Defines the parameters (hyperparameters) to be set and the values to be tested
param_grid = {
    'n_estimators': [5, 15, 25, 45, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Creates a random forest clasificator with the random state at 42
rfc_rs = RandomForestClassifier(random_state=42)

# Performs the random search
print("Searching")
random_search = RandomizedSearchCV(
    estimator=rfc_rs,
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    random_state=42,
    n_jobs=-1
)
# Trains the model
print("Training")
random_search.fit(x_train, y_train)

# Evaluate the model with the test data
print("Report")
y_pred = random_search.predict(x_test)
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