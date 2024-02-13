import { useState, useEffect } from "preact/hooks";
import { getLatestFeedbacks } from "../utils";
import { feedbacks, showFeedbackLoading } from "./../stores/feedbacks"
import { useStore } from '@nanostores/preact';
import DivergingBarChart from './../components/DivergingBarChart';

export default function Stats() {

  const $feedbacks = useStore(feedbacks);
  const $showFeedbackLoading = useStore(showFeedbackLoading);
  const formatDate = (timestamp: number) => {
    return new Date(parseInt(`${timestamp}`)).toLocaleString();
  }
  const [expandedRows, setExpandedRows]: any = useState([]);

  const toggleRowExpansion = (index:number) => {
    if (expandedRows.includes(index)) {
      setExpandedRows(expandedRows.filter((i:number) => i !== index));
    } else {
      setExpandedRows([...expandedRows, index]);
    }
  };

  useEffect(() => {
    getLatestFeedbacks(feedbacks, showFeedbackLoading)
  }, []);

  return (
    <div class="stats">
      <h2>Last 20 feedbacks</h2>
      <div class={`loading ${$showFeedbackLoading ? "icon-visible" : ""}`}>
        <i class="fa-solid fa-spinner-scale fa-spin-pulse"></i>
      </div>
      {$showFeedbackLoading ? "" :
        <>
          <DivergingBarChart fbData={$feedbacks} />
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Text</th>
                <th>Prediction</th>
                <th>Feedback</th>
              </tr>
            </thead>
            <tbody>

              {$feedbacks.map((item: any, index) => (
                <tr key={index} class={item.feedback ? "correct" : "incorrect"}>
                  <td>{formatDate(item.timestamp)}</td>
                  <td class="pointer" onClick={() => toggleRowExpansion(index)}>
                    {expandedRows.includes(index) ? item.text : item.text.slice(0, 50)}
                    {item.text.length > 50 && !expandedRows.includes(index) && '...'}
                  </td>
                  <td>{item.segment >= 0.5 ? "Positive" : "Negative"}</td>
                  <td>{item.feedback ? '✅' : '❌'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      }
      <a class="github" href="https://github.com/emergy-official/segment.parf.ai" target="_blank">Github</a>
    </div>
  );
}
