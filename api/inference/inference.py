import os
import joblib
import io  
import base64  
import json
# Import necessary libraries  
import tensorflow as tf
from tensorflow.keras import backend as K  
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def dice_coef(y_true, y_pred, smooth=1e-6):  
    y_true_f = K.flatten(y_true)  
    y_pred_f = K.flatten(y_pred)  
    intersection = K.sum(y_true_f * y_pred_f)  
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)  
  
def iou(y_true, y_pred, smooth=1e-6):  
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])  
    union = K.sum(y_true,[1,2,3]) + K.sum(y_pred,[1,2,3]) - intersection  
    return K.mean((intersection + smooth) / (union + smooth), axis=0)  

def read_image(base64_str):  
    img_bytes = base64.b64decode(base64_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)      
    origin_x = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    original_height, original_width = origin_x.shape[:2] 
      
    x = cv2.resize(origin_x, (512, 256))  
    x = x/255.0  
    x = x.astype(np.float32)  
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x, (original_height, original_width)
    
def model_inference_with_display(model, base64_image):  
    # Load and preprocess the image  
    image, origin_size = read_image(base64_image) 
    image_to_predict = np.expand_dims(image, axis=0)
    prediction = model.predict(image_to_predict)
    predicted_mask = tf.argmax(prediction, axis=-1)[0].numpy()  
    
    predicted_mask = tf.image.resize(predicted_mask[..., tf.newaxis], origin_size, method='nearest').numpy().astype(np.uint8)[:, :, 0]
    
    
    # Ensure it's a numpy array
    predicted_mask_normalized = predicted_mask.astype('float32') / predicted_mask.max()  
    colored_mask = plt.get_cmap('viridis')(predicted_mask_normalized)[:, :, :3]  # Exclude alpha channel  
    colored_mask = (colored_mask * 255).astype(np.uint8)  # Convert back to 8-bit format  
    
   # Convert numpy array back to PIL image  
    mask_image = Image.fromarray(colored_mask)  
    # Convert the PIL Image to a bytes object in PNG format  
    buffered = io.BytesIO()  
    mask_image.save(buffered, format="PNG")  
    image_png_bytes = buffered.getvalue()  
      
    # Encode bytes to base64 string  
    base64_image = base64.b64encode(image_png_bytes).decode("utf-8")
    
     # Initialize a dictionary to hold polygons for each label  
    labels_polygons = {  
        'void': [],   
        'flat': [],            
        'construction': [],    
        'object': [],         
        'nature': [],         
        'sky': [],         
        'human': [],            
        'vehicle': []          
    }
    
    for label, value in labels_polygons.items():  
        label_index = list(labels_polygons.keys()).index(label)  # Get the index of the label  
        mask = (predicted_mask == label_index).astype(np.uint8) * 255  # Create a mask for current label  
          
        # Find contours  
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  
          
        for contour in contours:  
            approx = contour.flatten().tolist()  # Without approximation  
            labels_polygons[label].append(approx)  
      
    # Convert the dictionary to a JSON object  
    return {"base64_image": base64_image, "polygons_json": labels_polygons}
    
# model = tf.keras.models.load_model("model.keras", custom_objects={'dice_coef': dice_coef, 'iou': iou})
# model_inference_with_display(model, "test.png")


def load_model():
    """
    Load the model from the specified directory.
    """
    # return joblib.load(model_path)
    return tf.keras.models.load_model("model.keras", custom_objects={'dice_coef': dice_coef, 'iou': iou})


def predict(model, file_stream):
    """
    Generate predictions for the incoming request using the model.
    """
    predictions = model_inference_with_display(model, file_stream)
    return {"predictions": predictions}