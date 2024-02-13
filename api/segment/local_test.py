# Import necessary libraries  
import pickle  
import json  
import numpy as np  
  
# Load the model saved in the previous run from a file  
with open("./artifacts/model.pkl", "rb") as file:    
    model = pickle.load(file)  
  
# Load the tokenizer saved in the previous run from a file  
with open("./artifacts/tokenizer.pkl", "rb") as file:    
    tokenizer_w2vec = pickle.load(file)  
  
# Load the parameters such as `input_length` used in training the model, stored in a JSON file  
params = json.load(open("./artifacts/params.json", 'r', encoding='utf-8'))  
  
# Define a custom function for padding sequences to a fixed length  
def pad_sequences(sequence, maxlen, value=0):  
    """  
    Pads sequences to ensure they all have the same length.  
      
    Args:  
        sequence (list of list of ints): The sequences to pad.  
        maxlen (int): Desired length of each sequence.  
        value (int): Padding value. Sequences will be padded with this value.  
          
    Returns:  
        np.array: Array of padded sequences.  
    """
    return np.array([np.pad(s[:maxlen], (max(0, maxlen-len(s)), 0), 'constant', constant_values=value)  
                     if len(s) < maxlen else s[:maxlen] for s in sequence])  
  
# Function to predict the segment of a comment  
def predict(comment, model, vector_model):  
    """  
    Predicts the segment for the given text.  
      
    Args:  
        text_to_predict (list of str): List containing text to predict.  
          
    Returns:  
        float: The predicted segment represented as a float.  
    """
    # Tokenize the comment text  
    sequence = vector_model.texts_to_sequences([comment])  
    # Pad the resulted sequence  
    padded_sequence = pad_sequences(sequence, maxlen=int(params["input_length"]))  
  
    # Predict the segment  
    prediction = model.predict(padded_sequence)  
    segment = prediction.astype(float)[0][0]  
    return segment.item()  
  
# Example usage of the predict function with three different comments  
print(predict("I am so sad, this is very bad news, terrible!", model, tokenizer_w2vec))  
print(predict("I am so happy, this is very good news, congrats!", model, tokenizer_w2vec))  
print(predict("Our newsfeed is full of sadness today as this absolutely #devastating news broke.", model, tokenizer_w2vec))  