import React, { useState } from "react";
import "../App.css";
import { getPrediction } from "../server.ts"; // your helper function

interface ImageUploadProps {
  onChange?: (file: File | null) => void;
  onPrediction?: (result: string) => void;
  onConfidence?: (result: string) => void;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onChange, onPrediction, onConfidence }) => {
    const [preview, setPreview] = useState<string | null>(null);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [predictionResult, setPredictionResult] = useState<string>("");
    const [confidence, setConfidence] = useState<string>("");

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0] || null;

        if (!file) {
            setPreview(null);
            setSelectedFile(null);
            onChange?.(null);
            onPrediction?.("");
            onConfidence?.("");
            return;
        }

        const previewURL = URL.createObjectURL(file);
        setPreview(previewURL);
        setSelectedFile(file);
        onChange?.(file);
        onPrediction?.(predictionResult);
        onConfidence?.(confidence);
    };

    const handleProcess = async () => {
        if (!selectedFile) {
            alert("Upload a file first!");
            return;
        }

        try {
            const result = await getPrediction(selectedFile);
            const img = document.getElementById("myImage") as HTMLImageElement;
            img.src = `data:image/png;base64,${result.prediction_image}`;
            console.log("Prediction:", result);

            // const resultBtn = document.getElementById("resultBtn") as HTMLButtonElement;
            // resultBtn.innerHTML = "Result: " + predictionResult.split("(")[0] + "";

            onPrediction?.(result.prediction_string);
            onConfidence?.(result.confidence);
        } catch (err) {
            console.error("Prediction failed:", err);
        }
    };

    return (
        <label className="fileUpload">
            {preview ? (
                <div className="image">
                    <img src={preview} alt="Uploaded" />
                </div>
            ) : (
                <span className="plus">+</span>
            )}

            <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
            />

            <button id="processBtn" onClick={handleProcess}>
                Process
            </button>
        </label>
    );
};

export default ImageUpload;
