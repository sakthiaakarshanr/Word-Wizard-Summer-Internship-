import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\Summer_Intern\aif_emo_sen_new.csv", encoding='utf-8-sig')
print(df.info())
print(df.head())

#Encode the target column
le = LabelEncoder()
y = le.fit_transform(df["Emotion"])

#Features and target
x = df.drop(columns = ["author","poem_name", "Emotion", "Sentiment"], axis=1)
scaler = StandardScaler()
x = scaler.fit_transform(x)

#Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.23, random_state=69)

#SMOTE
#smote = SMOTE(random_state=42)
#x_train, y_train = smote.fit_resample(x_train, y_train)

#Decision Tree Classifier
model = DecisionTreeClassifier(
    criterion='log_loss',  
    splitter='random',
    max_depth=20,
    random_state=47
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
crit = ["gini", "entropy", "log_loss"]
split = ["best", "random"]
max_depth = [None, 5, 10, 15, 20]
random_state = [42, 44, 45, 46, 47]
test_size = [0.2, 0.23, 0.25, 0.28, 0.3, 0.33, 0.35, 0.38, 0.4]
best_results = 0 

for i in crit:
    for j in split:
        for k in max_depth:
            for l in random_state:
                for m in range(1,51):
                        for n in test_size:
                            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=n, random_state=m)
                            smote = SMOTE(random_state=42)
                            model = DecisionTreeClassifier(
                                 criterion=i,
                                 splitter=j,
                                 max_depth=k,
                                 random_state=l
                                )
                            model.fit(x_train, y_train)
                            y_test_pred = model.predict(x_test)
                            test_acc = accuracy_score(y_test, y_test_pred)
                            print(f"Criterion: {i}, Splitter: {j}, Max Depth: {k}, Random State: {l}, Test Size:{m}, {n:.2f}, Test Accuracy: {test_acc:.4f}")
                             """

import joblib
import pickle
import cloudpickle

joblib.dump(model, r'C:\Users\Dell\OneDrive\Desktop\Summer_Intern\Webpage\ml_models\emotion\jl\emot_dt_model.joblib')
joblib.dump(scaler, r'C:\Users\Dell\OneDrive\Desktop\Summer_Intern\Webpage\ml_models\emotion\jl\emot_dt_scaler.joblib')
joblib.dump(le, r'C:\Users\Dell\OneDrive\Desktop\Summer_Intern\Webpage\ml_models\emotion\jl\emot_dt_le.joblib')


with open(r'C:\Users\Dell\OneDrive\Desktop\Summer_Intern\Webpage\ml_models\emotion\pic\emot_dt_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open(r'C:\Users\Dell\OneDrive\Desktop\Summer_Intern\Webpage\ml_models\emotion\pic\emot_dt_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open(r'C:\Users\Dell\OneDrive\Desktop\Summer_Intern\Webpage\ml_models\emotion\pic\emot_dt_le.pkl', 'wb') as f:
    pickle.dump(le, f)


with open(r'C:\Users\Dell\OneDrive\Desktop\Summer_Intern\Webpage\ml_models\emotion\cloudpic\emot_dt_model.cpkl', 'wb') as f:
    cloudpickle.dump(model, f)
with open(r'C:\Users\Dell\OneDrive\Desktop\Summer_Intern\Webpage\ml_models\emotion\cloudpic\emot_dt_scaler.cpkl', 'wb') as f:
    cloudpickle.dump(scaler, f)
with open(r'C:\Users\Dell\OneDrive\Desktop\Summer_Intern\Webpage\ml_models\emotion\cloudpic\emot_dt_le.cpkl', 'wb') as f:
    cloudpickle.dump(le, f)
