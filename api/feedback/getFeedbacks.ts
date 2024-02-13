
import {
    QueryCommand,
} from "@aws-sdk/client-dynamodb";

import { dynamoDBClient, returnData } from "./helper"

// Get the latest 20 feedbacks
export const getFeedbacks = async () => {
    const tableName = process.env.TABLE_NAME || "segment-feedback";

    // DynamoDB Query
    const command = new QueryCommand({
        "TableName": tableName,
        "Limit": 20,
        "KeyConditionExpression": "#kn0 = :kv0",
        "ScanIndexForward": false,
        "ExpressionAttributeNames": {
            "#kn0": "partition"
        },
        "ExpressionAttributeValues": {
            ":kv0": {
                "N": "2024"
            }
        },
        "Select": "ALL_ATTRIBUTES"
    });

    let response;
    try {
        const results = await dynamoDBClient.send(command);
        response = {
            success: true, items: results.Items?.map(e => ({
                text: e.text.S,
                timestamp: e.timestamp.N,
                segment: e.segment.N,
                feedback: e.feedback.BOOL
            }))
        }
    } catch (err: any) {
        console.error("Error fetching latest items:", err);
        response = { success: true, message: err.message }
    }

    return returnData(response)
}