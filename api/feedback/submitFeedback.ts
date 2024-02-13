
import {
  BatchWriteItemCommand,
} from "@aws-sdk/client-dynamodb";
import { dynamoDBClient, wait, returnData } from "./helper"

type SUBMIT_FEEDBACK_INPUT = {
  text: string,
  segment: number,
  feedback: boolean
}

// Create a new feedback to DynamoDB
export const submitFeedback = async ({ text, segment, feedback }: SUBMIT_FEEDBACK_INPUT) => {
  const input:any = {
    RequestItems: {
      ["segment-feedback"]: [
        {
          PutRequest: {
            Item: {
              partition: {
                N: new Date().getFullYear().toString(),
              },
              timestamp: {
                N: new Date().getTime().toString(),
              },
              text: {
                S: text || "",
              },
              segment: {
                N: segment.toString() || "",
              },
              feedback: {
                BOOL: feedback,
              },
            },
          },
        },
      ],
    },
  };

  const command = new BatchWriteItemCommand(input);
  const max_retry = 3;
  let response;

  for (let retry = 1; retry <= max_retry; retry++) {
    try {
      await dynamoDBClient.send(command);
      console.log("Done adding feedback");
      response = { success: true };
      break;
    } catch (error: any) {
      console.log(`BatchWriteItem failed on retry ${retry}.`);
      await wait(2000); // Wait for 2 seconds before retrying
      if (retry >= max_retry) {
        response = { success: false, message: error?.message };
      }
    }
  }
  try {
  } catch (error: any) {
    response = { success: false, message: error?.message };
  }

  return returnData(response)
}