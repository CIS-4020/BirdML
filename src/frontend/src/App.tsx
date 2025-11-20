import './App.css'
import ImageUpload from './components/ImageUpload'
import { useState, useEffect } from "react";

function App() {
    const [prediction, setPrediction] = useState<string>("Upload a Bird Image");
    const [confidence, setConfidence] = useState<string>("");
    const [confidenceStr, setConfidenceStr] = useState<string>("");

    useEffect(() => {
        if (confidence == "") {
            setConfidenceStr("");
        }
        else {
            setConfidenceStr("(" + (parseFloat(confidence) * 100).toFixed(2).toString() +"% Confident)");
        }
    }, [confidence]);
    
    return (
        <>
            <div id="container">
                <div id="header">
                    <h1>BirdML</h1>
                </div>
                <section id="main">
                    <ImageUpload 
                        onChange={() => null}
                        onPrediction={(result) => setPrediction(result)}
                        onConfidence={(result) => setConfidence(result)}
                    />
                    <div className="fileUpload">
                        <div className="image">
                            <img id="myImage" />
                        </div>
                        <button id="resultBtn">Result: {prediction.split("(")[0]} {confidenceStr}</button>
                    </div>
                </section>
            </div>
        </>
    )
}

export default App
