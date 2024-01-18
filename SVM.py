from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

X = data_encoded.drop(columns=['diabetes'])
y = data_encoded['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = svm.SVC()

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

result_df = pd.DataFrame({'Classification Report': [report], 'Confusion Matrix': [conf_matrix]})

result_df.to_csv('svm_result.csv', index=False)
