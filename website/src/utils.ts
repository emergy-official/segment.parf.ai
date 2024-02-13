import axios from 'axios';

const getAPIURL = () => {
    return document.location.host.includes("localhost") ? "https://dev.segment.parf.ai/api" : "/api"
}

export const getRandomElement = (arr: any) => {
    return arr[Math.floor(Math.random() * arr.length)];
}

export const getLatestFeedbacks = async (feedbackStore: any, showFeedbackLoading: any) => {
    try {
        showFeedbackLoading.set(true)
        const res: any = await axios.get(`${getAPIURL()}/segment/feedback`)
        if (res?.data?.items) {
            feedbackStore.set(res?.data?.items)
        }
        showFeedbackLoading.set(false)
    } catch (e) {
        console.error("Error", e)
        alert(`Error response of the feedback API ${e.message}`)
        return null
    }
}

export const startLambda = async (isStarted: any, setIsStarted: any) => {
    if (isStarted) return;

    let attempt = 0;
    const maxAttempts = 10;

    while (attempt < maxAttempts) {
        try {
            console.log(`Attempt ${attempt}`)
            const timeout = attempt === 0 ? 180000 : 20000;
            const res: any = await sendPredictRequest("any text here to start the lambda to improve speed for future request", true, timeout);
            if (res?.segment > -1) {
                setIsStarted(true);
                return;  // Successful, exit the function  
            }
        } catch (e) {

            console.error(`Attempt ${attempt + 1} failed. Error:`, e);
            if (attempt === maxAttempts - 1) {
                alert(`Cannot initialize the service: ${e.message}`);
            }

        }

        attempt++;
    }
    return null;
};

export const sendPredictRequest = async (text: string, throwError: boolean = false, timeout: number = 180000) => {
    try {
        const res: any = await axios({
            method: "post",
            url: `${getAPIURL()}/segment`,
            data: { text },
            timeout
        })
        return res?.data
    } catch (e) {
        if (throwError) {
            throw e
        } else {
            console.error("Error", e)
            alert(`Error response of the API ${e.message}`)
            return null
        }
    }
}

export const sendFeedbackRequest = async (text: string, segment: number, feedback: boolean) => {
    try {
        const res: any = await axios.post(`${getAPIURL()}/segment/feedback`, { text, segment, feedback })
        return res?.data
    } catch (e) {
        console.error("Error", e)
        alert(`Error response of the API ${e.message}`)
        return null
    }
}