import React, { useState } from "react";
import "../App.css";
import { getPrediction } from "../server.ts"; // your helper function

interface ImageUploadProps {
  onChange?: (file: File | null) => void;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onChange }) => {
    const [preview, setPreview] = useState<string | null>(null);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [fileName, setFileName] = useState<string>("");

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0] || null;

        if (!file) {
            setPreview(null);
            setFileName("");
            setSelectedFile(null);
            onChange?.(null);
            return;
        }

        const previewURL = URL.createObjectURL(file);
        setPreview(previewURL);
        setFileName(file.name);
        setSelectedFile(file);

        onChange?.(file);
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
