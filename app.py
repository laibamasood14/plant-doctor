from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import logging
import os
import sys
from pathlib import Path
from utils import convert_image_to_base64_and_test, test_with_base64_data

# Add Leaf Disease directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "Leaf Disease"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Plant Doctor API", version="2.0.0", description="Complete plant diagnosis system with classification, disease detection, and care recommendations")

@app.post('/disease-detection-file')
async def disease_detection_file(file: UploadFile = File(...)):
    """
    Endpoint to detect diseases in leaf images using direct image file upload.
    Accepts multipart/form-data with an image file.
    """
    try:
        logger.info("Received image file for disease detection")
        
        # Read uploaded file into memory
        contents = await file.read()
        
    # Process file directly from memory
        result = convert_image_to_base64_and_test(contents)
        
    # No cleanup needed since file is not saved locally
        
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to process image file")
        logger.info("Disease detection from file completed successfully")
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in disease detection (file): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post('/diagnose')
async def diagnose_plant(file: UploadFile = File(...)):
    """
    Complete plant diagnosis endpoint that combines plant classification, 
    disease detection, and knowledge base recommendations.
    
    This endpoint provides comprehensive plant analysis including:
    - Plant species identification (Roboflow)
    - Disease detection (Groq AI)
    - Care recommendations (Knowledge Base)
    """
    try:
        logger.info("Received image file for complete plant diagnosis")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read uploaded file into memory
        contents = await file.read()
        
        # Convert to base64 for processing
        import base64
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # Import and use the safe diagnosis pipeline with fallbacks
        from main import safe_diagnose
        
        # Run safe diagnosis with tiered fallbacks
        result = safe_diagnose(base64_image)
        
        if not result.get("pipeline_success", False):
            logger.warning("Plant diagnosis pipeline completed with issues")
        
        logger.info("Complete plant diagnosis completed successfully")
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in plant diagnosis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "Plant Doctor API - Complete Plant Diagnosis System",
        "version": "2.0.0",
        "description": "AI-powered plant identification, disease detection, and care recommendations",
        "endpoints": {
            "diagnose": "/diagnose (POST, file upload) - Complete plant diagnosis",
            "disease_detection_file": "/disease-detection-file (POST, file upload) - Disease detection only"
        },
        "features": [
            "Plant species identification (Roboflow)",
            "Disease detection and analysis (Groq AI)",
            "Plant-specific care recommendations (Knowledge Base)",
            "Treatment and prevention advice"
        ]
    }
