// index.ts  
import { getFeedbacks } from './getFeedbacks';
import { returnData } from './helper';
import { submitFeedback } from './submitFeedback';

// Incoming lambda request
export async function handler(event: any, _: any) {
  console.log("INPUT", event);

  // POST method to submit the feedback
  if (event.httpMethod == "POST") {
    const params = JSON.parse(event.body);
    return submitFeedback(params)
  } else if (event.httpMethod == "GET") {
    // GET method to retrieve the latest feedback
    return getFeedbacks()
  }

  // Something else that should not happen.
  return returnData({
    message: "Nothing to say"
  })
}  