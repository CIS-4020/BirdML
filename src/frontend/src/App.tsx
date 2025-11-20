import './App.css'
import ImageUpload from './components/ImageUpload'
import { useState } from "react";

function App() {
    const [prediction , setPrediction] = useState<string>("Upload a Bird Image");
    
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
                    />
                    <div className="fileUpload">
                        <div className="image">
                            <img id="myImage" />
                        </div>
                        <button id="resultBtn">Result: {prediction.split("(")[0]}</button>
                    </div>
                </section>
            </div>
        </>
    )
}

export default App
