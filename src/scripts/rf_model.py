import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import joblib
import pickle
import cloudpickle

# Load the dataset
df = pd.read_csv(r"H:\Text Mining\Tamil poems\all_aif.csv", encoding='utf-8-sig')
print(df.info())
print(df.head())

#Encode the target column
le = LabelEncoder()
y = le.fit_transform(df["author"])

#Features and target
x = df.drop(columns = ["author","poem_name"], axis=1)
print(x.columns)

#Feature scaling
scaler = StandardScaler()
x = scaler.fit_transform(x)


print(x.shape)
print(y.shape)

#Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)

#Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    n_jobs=-1,
    random_state=42
)
model.fit(x_train, y_train)

#Prediction
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_test_pred)
class_report = classification_report(y_test,y_test_pred, target_names=le.classes_)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
"""
n_esti = [1000, 2000, 3000, 4000, 5000]
max_depths = [4, 5, 6, 7, 8, 9, 10, 20, 30, None]
crit = ['gini', 'entropy', 'log_loss']

for i in n_esti:
    for j in max_depths:
        for k in crit:
            model = RandomForestClassifier(
            n_estimators=i,
            criterion=k,
            n_jobs=-1,
            max_depth=j,
            random_state=42
        )
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f"{i}, {j}, {k}")
        print(f"Train : {train_acc}, Test : {test_acc}")"""
'''        
import joblib

joblib.dump(model, r'H:\Text Mining\Tamil poems\Webpage\new_models\AIF\jl\aif_rf_model.joblib')
joblib.dump(scaler, r'H:\Text Mining\Tamil poems\Webpage\new_models\AIF\jl\aif_rf_scaler.joblib')
joblib.dump(le, r'H:\Text Mining\Tamil poems\Webpage\new_models\AIF\jl\aif_rf_le.joblib')


with open(r'H:\Text Mining\Tamil poems\Webpage\new_models\AIF\pic\aif_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open(r'H:\Text Mining\Tamil poems\Webpage\new_models\AIF\pic\aif_rf_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open(r'H:\Text Mining\Tamil poems\Webpage\new_models\AIF\pic\aif_rf_le.pkl', 'wb') as f:
    pickle.dump(le, f)


with open(r'H:\Text Mining\Tamil poems\Webpage\new_models\AIF\clo_pic\aif_rf_model.cpkl', 'wb') as f:
    cloudpickle.dump(model, f)
with open(r'H:\Text Mining\Tamil poems\Webpage\new_models\AIF\clo_pic\aif_rf_scaler.cpkl', 'wb') as f:
    cloudpickle.dump(scaler, f)
with open(r'H:\Text Mining\Tamil poems\Webpage\new_models\AIF\clo_pic\aif_rf_le.cpkl', 'wb') as f:
    cloudpickle.dump(le, f)
'''
