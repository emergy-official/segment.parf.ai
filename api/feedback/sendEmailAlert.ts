
import {
  SendEmailCommand,
} from "@aws-sdk/client-ses";

import {
  sesClient,
} from "./helper";
import { getFeedbacks } from "./getFeedbacks";

export const handler = async () => {
  // I'm getting the last feedbacks to send the negative one by email.
  // Here I'm not filtering by key positive/negative because it will cost me money to maintain the secondary index
  // which I haven't created, so since it's a test project, I'm returning the last 20 assuming I'll get the 5 negative within 5 minutes.
  const lastFeedbacks = await getFeedbacks();
  const negativeFeedbacks = JSON.parse(lastFeedbacks.body).items.filter((e: any) => e.feedback === false).slice(0, 3)

  const fromEmail = process.env.FROM_EMAIL || "it@parf.ai"
  const toEmail = process.env.TO_EMAIL || "contact@parf.ai"


  let text = ["You got 3 negative feedbacks within the last 5 minutes\n"]

  text.push(...negativeFeedbacks.map((e: any) => {
    return `${new Date(parseInt(e.timestamp)).toLocaleString()} - ${e.text} - predicted as ${Math.round(e.sentimemt) ? "negative" : "positive"} the user flagged it has incorrect`
  }))


  const params = {
    Source: fromEmail, // Replace with your verified SES email address  
    Destination: {
      ToAddresses: toEmail.split(","), // Replace with the recipient email address  
    },
    Message: {
      Subject: {
        Data: 'Alarm Negative Feedback triggered',
        Charset: 'UTF-8',
      },
      Body: {
        Text: {
          Data: text.join('\n'),
          Charset: 'UTF-8',
        },
      },
    },
  };

  try {
    const data = await sesClient.send(new SendEmailCommand(params));
    console.log('Email sent:', data);
    return { status: 'SUCCESS', data };
  } catch (error) {
    console.error('Error sending email:', error);
    return { status: 'ERROR', error };
  }
};