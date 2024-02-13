# test_lambda_function.py  
import unittest  
from lambda_function import handler  
import json  
  
class TestLambdaFunction(unittest.TestCase):  
  
    def test_sentiment_negative(self):  
        event = {'body': '{"text": "I am so sad, this is very bad news, terrible!"}'}  
        result = handler(event, None)  
        self.assertEqual(result['statusCode'], 200)  
        sentiment = json.loads(result['body'])['sentiment']
        self.assertLess(sentiment, 0.5)  
          
    def test_sentiment_neutral(self):  
        event = {'body': '{"text": "I am so happy this is great news, congrats!"}'}  
        result = handler(event, None)  
        self.assertEqual(result['statusCode'], 200)  
        sentiment = json.loads(result['body'])['sentiment']  
        self.assertGreater(sentiment, 0.5)  
  
if __name__ == '__main__':  
    unittest.main()  