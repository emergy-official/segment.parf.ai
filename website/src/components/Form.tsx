import { useState, useEffect } from "preact/hooks";
import { sendPredictRequest, sendFeedbackRequest, getLatestFeedbacks, getRandomElement, startLambda } from "../utils";

import { feedbacks, showFeedbackLoading } from "./../stores/feedbacks"

export default function Form() {

  const sampleArray = [
    "Breaking: Nepal enters the super six of ongoing U-19 Men's World Cup after winning with Afghanistan by 1 wicket. ❤️🇳🇵 #Congrats",
    "Yall see how they fake the screenshots when they get caught? #sad",
    "The mirror is my best friend because when I cry 😭 it never laughs😂.",
    "Behind every sweet smile, there is a bitter sadness that no one can ever see and feel...",
    "In the first picture, I was in the midst of my pill addiction. I was lost, broken, and hopeless. The second picture is me today. I couldn’t be more grateful of where I am today.",
    "When you are happy and don't give a sh*t about the world...",
    "Good morning everyone happy Friday, hope the day is kind to you and those around you #FlowersOnFriday to cheer up a wet stormy day, taken last summer bathed in sunshine full of positive vibes 😊",
    "I’M ACCEPTED FOR MY UNIVERSITY EXCHANGE IN TOKYO !!! (SOPHIA UNIVERSITY)",
    "Happy Throwback Thursday Corgi-Family! Hope you’re well and having a great day today friend! Peace and Blessings 📸🐶💕✌🏻",
    " TORNADO in KWAZULU-NATAL around 3pm",
    "Our newsfeed is full of sadness today as this absolutely #devastating news broke.",
    "It's truly devastating to witness the violence and bloodshed that disrupts peace and Kashmiriyat",
    "if you can build such a team, you will attract the attention of the Eye on the Hill, you will need me in the neutral zone to defeat him!!! F.E.",
    "Somewhere between emotional and emotionless"
  ]

  const [text, setText] = useState(getRandomElement(sampleArray));
  const [prediction, setPrediction] = useState(-1);
  const [isStarted, setIsStarted] = useState(false);

  const [loadingPrediction, setLoadingPrediction] = useState(false);

  const [loadingApproval, setLoadingApproval] = useState(false);
  const [loadingRefusal, setLoadingRefusal] = useState(false);

  const [showFeedback, setShowFeedback] = useState(false);

  const isPositive = Math.round(prediction) == 1


  useEffect(() => {
    startLambda(isStarted, setIsStarted)
  }, []);


  const handleInput = (e: any) => {
    const target = e.target as HTMLTextAreaElement;

    setPrediction(-1)
    setLoadingPrediction(false)
    setLoadingRefusal(false)
    setShowFeedback(false)
    setText(target.value);
  };

  const handlePredictBtn = async () => {
    setLoadingPrediction(true)
    setShowFeedback(false)
    const res = await sendPredictRequest(text)
    if (res) {
      setPrediction(res.segment)
    }
    setLoadingPrediction(false)
  };

  const handleFeedbackBtn = async (isPositive: boolean) => {
    if (isPositive) {
      setLoadingApproval(true)
      await sendFeedbackRequest(text, prediction, isPositive)
      setLoadingApproval(false)
      setPrediction(-1)
      setShowFeedback(true)
      getLatestFeedbacks(feedbacks, showFeedbackLoading)
      setText(getRandomElement(sampleArray))
    } else {
      setLoadingRefusal(true)
      await sendFeedbackRequest(text, prediction, isPositive)
      setLoadingRefusal(false)
      setPrediction(-1)
      setShowFeedback(true)
      setText(getRandomElement(sampleArray))
    }

    getLatestFeedbacks(feedbacks, showFeedbackLoading)


  };

  return (
    <div class="form">
      <h1>Segment analyzer</h1>
      <textarea onInput={handleInput} value={text} maxLength={280}></textarea>
      {prediction == -1 ?
        !isStarted ? 
          <button class={`predict icon-visible flex`} disabled={true} >
            <span>Initializing Service <i class="fa-solid fa-spinner-scale fa-spin-pulse"></i> (up to 1min to start) </span>
          </button> :
          <button class={`predict ${loadingPrediction ? "icon-visible" : ""}`} disabled={loadingPrediction} onClick={handlePredictBtn}>
            {/* FontAwesome Hack */}
            {loadingPrediction ? "" : "Predict"}
            <i class="fa-solid fa-spinner-scale fa-spin-pulse"></i>
          </button> :
        <div class="prediction">
          <span class={`prediction-value`}>
            Text is {isPositive ? `positive` : `negative`} ({prediction})
          </span>
          <div class="feedback">
            <button class={`agree ${loadingApproval ? "icon-visible" : ""}`} disabled={loadingApproval} onClick={() => { handleFeedbackBtn(true) }}>
              {loadingApproval ? "" : "Correct"}
              <i class="fa-solid fa-spinner-scale fa-spin-pulse"></i>
            </button>
            <button class={`disagree ${loadingRefusal ? "icon-visible" : ""}`} disabled={loadingRefusal} onClick={() => { handleFeedbackBtn(false) }}>
              {loadingRefusal ? "" : "Incorrect"}
              <i class="fa-solid fa-spinner-scale fa-spin-pulse"></i>
            </button>
          </div>
        </div>}
      {showFeedback ? <p>Thank you for your feedback!</p> : ""}

    </div>
  );
}
