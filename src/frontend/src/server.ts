const BASE_URL = "http://localhost:8080";

export async function getPrediction(imageFile: File) {
    const formData = new FormData();
    formData.append("image", imageFile);

    try {
        const res = await fetch(`${BASE_URL}/predict`, {
            method: "POST",
            body: formData, 
        });

        if (!res.ok) {
            throw new Error(`HTTP error ${res.status}`);
        }

        return await res.json();
    } catch (err) {
        console.error("getPrediction error:", err);
        throw err;
    }
}