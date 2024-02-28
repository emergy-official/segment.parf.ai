import joblib  
from sklearn.metrics import accuracy_score, classification_report  
from sklearn.model_selection import train_test_split  

from sklearn.datasets import load_iris  
  
# Load the saved model  
model_filename = 'iris_model.pkl'  
loaded_model = joblib.load(model_filename)  
  
# Load dataset  
iris = load_iris()  
X, y = iris.data, iris.target  
  
# Split dataset  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  

# Predict the outcomes for the test dataset  
# y_pred = loaded_model.predict(X_test)  
  
# # Calculate the accuracy  
# accuracy = accuracy_score(y_test, y_pred)  
# print(f"Accuracy: {accuracy:.2f}")  
  
# # Generate a classification report  
# report = classification_report(y_test, y_pred)  
# print("\nClassification Report:\n", report)  

# print(X_test[0])
model = tf.keras.models.load_model("model.keras", custom_objects={'dice_coef': dice_coef, 'iou': iou})

y_pred = loaded_model.predict([[6.5, 3.2, 5.1, 2.]])  
print(y_pred)
