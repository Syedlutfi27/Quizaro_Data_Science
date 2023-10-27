import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('credit_customers.csv')

data.dropna(inplace=True)

data.drop_duplicates(inplace=True)

le = LabelEncoder()
categorical_columns = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker']
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

X = data.drop(columns=['class'])
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
svm_linear_pred = svm_linear.predict(X_test)
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
svm_rbf_pred = svm_rbf.predict(X_test)

def evaluate_model(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    print("\n")

evaluate_model("Logistic Regression", y_test, lr_pred)
evaluate_model("KNN Classification", y_test, knn_pred)
evaluate_model("SVM (Linear Kernel)", y_test, svm_linear_pred)
evaluate_model("SVM (RBF Kernel)", y_test, svm_rbf_pred)


best_model = max([
    ("Logistic Regression", lr_pred),
    ("KNN Classification", knn_pred),
    ("SVM (Linear Kernel)", svm_linear_pred),
    ("SVM (RBF Kernel)", svm_rbf_pred)],
    key=lambda x: accuracy_score(y_test, x[1]))


print(f"Model with the Best Accuracy: {best_model[0]}")
