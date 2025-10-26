import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_pred_classes = [np.argmax(row) for row in y_pred]
    print(classification_report(y, y_pred_classes))
    cm = confusion_matrix(y, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.show()
