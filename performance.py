from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

report = classification_report(y_test, y_pred)
print(report)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)