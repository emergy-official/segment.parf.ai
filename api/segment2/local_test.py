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

def read_image(x):  
    x = cv2.imread(x, cv2.IMREAD_COLOR)  
    x = cv2.resize(x, (512, 256))  
    x = x/255.0  
    x = x.astype(np.float32)  
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x  
    
def model_inference_with_display(model, image_path):  
    # Load and preprocess the image  
    image = read_image(image_path)  
    image_to_predict = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image_to_predict)  
    predicted_mask = tf.argmax(prediction, axis=-1)[0].numpy()  # Ensure it's a numpy array  
    
    predicted_mask_normalized = predicted_mask.astype('float32') / predicted_mask.max()  
    colored_mask = plt.get_cmap('viridis')(predicted_mask_normalized)[:, :, :3]  # Exclude alpha channel  
    colored_mask = (colored_mask * 255).astype(np.uint8)  # Convert back to 8-bit format  
  
    # Convert numpy array back to PIL image and save  
    mask_image = Image.fromarray(colored_mask)  
    mask_image.save("predicted_colored_mask.png")
    
    # Ensure the original image is also scaled to [0, 255]  
    original_image_uint8 = (image * 255).astype(np.uint8)  

    # Convert numpy arrays to PIL Images for blending  
    original_pil = Image.fromarray(original_image_uint8)  
    mask_pil = Image.fromarray(colored_mask)  

    # Blend with specified opacity  
    combined_image = Image.blend(original_pil, mask_pil, alpha=0.5)  
    
    # Save the combined image  
    combined_image.save("combined_image.png")
    
model = tf.keras.models.load_model("model.keras", custom_objects={'dice_coef': dice_coef, 'iou': iou})
model_inference_with_display(model, "test.png")
# Define a custom function for padding sequences to a fixed length  
# def pad_sequences(sequence, maxlen, value=0):  
#     """  
#     Pads sequences to ensure they all have the same length.  
      
#     Args:  
#         sequence (list of list of ints): The sequences to pad.  
#         maxlen (int): Desired length of each sequence.  
#         value (int): Padding value. Sequences will be padded with this value.  
          
#     Returns:  
#         np.array: Array of padded sequences.  
#     """
#     return np.array([np.pad(s[:maxlen], (max(0, maxlen-len(s)), 0), 'constant', constant_values=value)  
#                      if len(s) < maxlen else s[:maxlen] for s in sequence])  
  
# # Function to predict the segment of a comment  
# def predict(comment, model, vector_model):  
#     """  
#     Predicts the segment for the given text.  
      
#     Args:  
#         text_to_predict (list of str): List containing text to predict.  
          
#     Returns:  
#         float: The predicted segment represented as a float.  
#     """
#     # Tokenize the comment text  
#     sequence = vector_model.texts_to_sequences([comment])  
#     # Pad the resulted sequence  
#     padded_sequence = pad_sequences(sequence, maxlen=int(params["input_length"]))  
  
#     # Predict the segment  
#     prediction = model.predict(padded_sequence)  
#     segment = prediction.astype(float)[0][0]  
#     return segment.item()  
  
# Example usage of the predict function with three different comments  
# print(predict("I am so sad, this is very bad news, terrible!", model, tokenizer_w2vec))  
# print(predict("I am so happy, this is very good news, congrats!", model, tokenizer_w2vec))  
# print(predict("Our newsfeed is full of sadness today as this absolutely #devastating news broke.", model, tokenizer_w2vec))  