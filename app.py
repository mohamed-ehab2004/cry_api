from fastapi import FastAPI, UploadFile, File
import shutil
import os

from models.cry_detection_model import CryDetectionModel
from models.cry_analysis_model import CryAnalysisModel

app = FastAPI()

detector = CryDetectionModel("saved_models/cry_detection_model.h5")
analyzer = CryAnalysisModel("saved_models/cry_analysis_model.h5")

@app.post("/analyze-cry")
async def analyze_cry(file: UploadFile = File(...)):

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # المرحلة 1: Detection
    det_result = detector.predict_file(temp_path)

    if not det_result["cry_detected"]:
        os.remove(temp_path)
        return det_result

    # المرحلة 2: Analysis
    cry_type, class_probs = analyzer.predict(temp_path)

    result = {
        **det_result,
        "cry_type": cry_type,
        "analysis_probs": class_probs
    }

    os.remove(temp_path)
    return result
