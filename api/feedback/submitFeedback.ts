
import {
  BatchWriteItemCommand,
} from "@aws-sdk/client-dynamodb";
import {
  PutMetricDataCommand,
} from "@aws-sdk/client-cloudwatch";
import { dynamoDBClient, cloudWatchClient, wait, returnData } from "./helper"

// Sent the metrics to cloudwatch
const sendFeedbackMetric = async (isPositive: boolean = false) => {
  const params: any = {
    MetricData: [
      {
        MetricName: "Sentiment",
        Dimensions: [
          {
            Name: "Feedback",
            Value: isPositive ? "Positive" : "Negative",
          },
        ],
        Unit: "Count",
        Value: 1,
      },
    ],
    Namespace: "segment.parf.ai",
  };
  const command = new PutMetricDataCommand(params);

  try {
    const data = await cloudWatchClient.send(command);
    console.log("Metric sent");
  } catch (err) {
    throw err
  }
};

type SUBMIT_FEEDBACK_INPUT = {
  text: string,
  sentiment: number,
  feedback: boolean
}

// Create a new feedback to DynamoDB
export const submitFeedback = async ({ text, sentiment, feedback }: SUBMIT_FEEDBACK_INPUT) => {
  const input:any = {
    RequestItems: {
      ["sentiment-feedback"]: [
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
              sentiment: {
                N: sentiment.toString() || "",
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
    await sendFeedbackMetric(feedback)
  } catch (error: any) {
    response = { success: false, message: error?.message };
  }

  return returnData(response)
}