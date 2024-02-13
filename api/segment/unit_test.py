# test_lambda_function.py  
import unittest  
from lambda_function import handler  
import json  
  
class TestLambdaFunction(unittest.TestCase):  
  
    def test_segment_negative(self):  
        event = {'body': '{"text": "I am so sad, this is very bad news, terrible!"}'}  
        result = handler(event, None)  
        self.assertEqual(result['statusCode'], 200)  
        segment = json.loads(result['body'])['segment']
        self.assertLess(segment, 0.5)  
          
    def test_segment_neutral(self):  
        event = {'body': '{"text": "I am so happy this is great news, congrats!"}'}  
        result = handler(event, None)  
        self.assertEqual(result['statusCode'], 200)  
        segment = json.loads(result['body'])['segment']  
        self.assertGreater(segment, 0.5)  
  
if __name__ == '__main__':  
    unittest.main()  