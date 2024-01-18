from sklearn import svm
from sklearn.model_selection import train_test_split

X = data_encoded.drop(columns=['diabetes'])
y = data_encoded['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = svm.SVC()

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
