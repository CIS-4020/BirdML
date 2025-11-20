const BASE_URL = "http://localhost:8080";

export async function getSample(arr: number[]) {
    try {
        const res = await fetch(`${BASE_URL}/sample`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ arr }),
        });

        if (!res.ok) {
            throw new Error(`HTTP error ${res.status}`);
        }

        return await res.json();
    } catch (err) {
        console.error(err);
        throw err;
    }
}

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