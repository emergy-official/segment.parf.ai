# Import necessary libraries  
import pickle  
import json  
import numpy as np  
from tensorflow import keras  
  
# Load the tokenizer object from a saved .pkl file  
with open("./artifacts/tokenizer.pkl", "rb") as file:    
    tokenizer = pickle.load(file)  
    
# Load a previously saved model from a .pkl file  
with open("./artifacts/model.pkl", "rb") as file:    
    model = pickle.load(file)  
      
# Load model parameters from a JSON file  
params = json.load(open("./artifacts/params.json", 'r', encoding='utf-8'))  
  
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
    # The operation results in each sequence being padded with the defined value  
    # to ensure uniform length, which is crucial for models expecting fixed input shapes.  
    return np.array([np.pad(s[:maxlen], (max(0, maxlen-len(s)), 0), 'constant', constant_values=value)   
                     if len(s) < maxlen else s[:maxlen] for s in sequence])    
  
def predict(text_to_predict):  
    """  
    Predicts the segment for the given text.  
      
    Args:  
        text_to_predict (list of str): List containing text to predict.  
          
    Returns:  
        float: The predicted segment represented as a float.  
    """  
    # Transform text to a sequence of integers using the loaded tokenizer  
    sequence = tokenizer.texts_to_sequences(text_to_predict)  
    # Pad these sequences to ensure uniform length  
    padded_sequence = pad_sequences(sequence, maxlen=int(params["input_length"]))  
    # Predict segment using the loaded model  
    prediction = model.predict(padded_sequence)  
    # Extract the segment value from the prediction  
    segment = prediction.astype(float)[0][0]  
    # Convert to native Python datatype and return  
    return segment.item()  
    
def handler(event, context):  
    """  
    AWS Lambda handler function for processing events.  
      
    Args:  
        event: Event data (dictionary).  
        context: Runtime information (object).  
          
    Returns:  
        dict: The function response including status code and body.  
    """  
    # Load the text from the event's body  
    body = json.loads(event['body'])  
    text = body['text']  
    # Log the text to predict  
    print("Predicting...", text)  
    # Return the HTTP response with the predicted segment  
    return {  
        'statusCode': 200,  
        'body': json.dumps({'segment': predict([text])})  
    }  