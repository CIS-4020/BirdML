import React, { useState } from "react";
import "../App.css";

interface ImageUploadProps {
  onChange?: (file: File | null) => void;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onChange }) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;

    if (!file) {
      setPreview(null);
      setFileName("");
      onChange?.(null);
      return;
    }

    const previewURL = URL.createObjectURL(file);
    setPreview(previewURL);
    setFileName(file.name);

    onChange?.(file);
  };

  return (
    <div className="imgUpload">
        <label className="fileUpload">
          {preview ? (
            <img src={preview} alt="Uploaded" className="upload-preview" />
          ) : (
            <span className="plus">+</span>
          )}
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
          />
          <button id="processBtn">Process</button>
        </label>

    </div>
  );
};

export default ImageUpload;
