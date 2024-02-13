# Segment.parf.ai - Feedback API

## Introduction

The feedback API allows a user to improve the model over time to teach it what is correct/incorrect.  
You can either get the latest 20 feedbacks or submit a feedback.

This API is executed on AWS Lambda in the `NodeJS20.x` runtime.
Using free tier, the first execution will take ~5sec time to start and will then stop after ~15min of inactiivty. (reserved concurrency lambda makes it always available for some $$)

The response time should be ~200ms per request after coldstart.

## API Usage example

- Prod: https://segment.parf.ai/api
- Dev: https://dev.segment.parf.ai/api

### Get the latest 20 feedbacks

```js
const request = require("request");
let options = {
  method: "GET",
  url: "https://dev.segment.parf.ai/api/segment/feedback",
};
request(options, (error, response) => {
  if (error) throw new Error(error);
  console.log(response.body);
});
```

Response example

```json
{
    "success": true,
    "items": [
        {
            "text": "The mirror is my best friend because when I cry 😭 it never laughs😂.",
            "timestamp": "1707494974834",
            "segment": "0.9879910945892334",
            "feedback": true
        },
        ...
    ]
}
```

### Post a feedback

```js
const request = require("request");
let options = {
  method: "POST",
  url: "https://segment.parf.ai/api/segment/feedback",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    text: "I am so sad, this is very bad news, terrible!",
    segment: 0.88,
    feedback: false,
  }),
};
request(options, (error, response) => {
  if (error) throw new Error(error);
  console.log(response.body);
});
```

Response example

```json
{
  "success": true
}
```

## How to test locally

```bash
# Go within the feedback api
cd api/feedback

# Install dependencies (using pnpm or npm, yarn, ...)
pnpm install

# Authentication to allow using DB queries and email sending
export DEV_ACCOUNT_ID=REPLACE_ME
export PROD_ACCOUNT_ID=REPLACE_ME
export INFRA_ACCOUNT_ID=REPLACE_ME
export ACCOUNT_ID=$DEV_ACCOUNT_ID

# Login from infra account to dev account
eval $(aws sts assume-role --profile $INFRA_ACCOUNT_ID --role-arn "arn:aws:iam::"$ACCOUNT_ID":role/provision" --role-session-name AWSCLI-Session | jq -r '.Credentials | "export AWS_ACCESS_KEY_ID=\(.AccessKeyId)\nexport AWS_SECRET_ACCESS_KEY=\(.SecretAccessKey)\nexport AWS_SESSION_TOKEN=\(.SessionToken)\n"')

# Run test
npm run test
```

If you need to build the lambda layer for the infrastructure and push it locally
```sh
# Go within the feedback api
cd api/feedback

# Create the lambda layer ready to be used by terraform
npm run prepare:layer
```

## Structure

**Scripts and Configuration Files**

- **feedback.test.ts**: Unit test done within Github actions
- **getFeedbacks.ts**: Script to gather feedbacks within DynamoDB
- **submitFeedback.ts**: Script to send a feedback of a text
- **helper.ts**: Reusable functions to share accross other scripts.
- **index.ts**: Script invoked by the API

**Configuration and Package Management**
- **package.json**: Libraries used
- **pnpm-lock.yaml**: By PNPM to reuse the same versions
- **tsconfig.json**: Typescript config
