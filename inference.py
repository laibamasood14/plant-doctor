"""
Roboflow Inference Client for Plant Classification
================================================

This module provides a client for interacting with Roboflow's inference API
to classify plant species from images using the identify-plant model.
"""

import os
import logging
from typing import Dict, Optional, Any
from inference_sdk import InferenceHTTPClient

logger = logging.getLogger(__name__)


class RoboflowInferenceClient:
    """
    Client for Roboflow inference API to identify plant species.
    
    This class handles communication with Roboflow's serverless inference
    endpoint to classify plant images and return species identification results.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_id: str = "identify-plant/1"):
        """
        Initialize the Roboflow inference client.
        
        Args:
            api_key (Optional[str]): Roboflow API key. If None, will attempt to
                                   load from ROBOFLOW_API_KEY environment variable.
            model_id (str): Model ID for plant identification (default: "identify-plant/1")
        """
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY not found in environment variables")
        
        self.model_id = model_id
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=self.api_key
        )
        logger.info(f"Roboflow inference client initialized with model: {model_id}")
    
    def classify_plant(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a plant image and return species identification.
        
        Args:
            image_path (str): Path to the image file or base64 string
            
        Returns:
            Dict[str, Any]: Classification results containing:
                - plant_name: Identified plant species name
                - confidence: Confidence score (0-1)
                - predictions: Full prediction details
                - success: Boolean indicating if classification was successful
        """
        try:
            logger.info(f"Starting plant classification for: {image_path[:50]}...")
            
            # Make inference request
            result = self.client.infer(image_path, model_id=self.model_id)
            
            # Extract plant name and confidence from predictions
            predictions = result.get('predictions', [])
            
            if not predictions:
                logger.warning("No predictions returned from Roboflow")
                return {
                    "plant_name": "Unknown Plant",
                    "confidence": 0.0,
                    "predictions": predictions,
                    "success": False
                }
            
            # Get the top prediction
            top_prediction = predictions[0]
            plant_name = top_prediction.get('class', 'Unknown Plant')
            confidence = float(top_prediction.get('confidence', 0.0))
            
            logger.info(f"Plant classified as: {plant_name} (confidence: {confidence:.2f})")
            
            return {
                "plant_name": plant_name,
                "confidence": confidence,
                "predictions": predictions,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Plant classification failed: {str(e)}")
            return {
                "plant_name": "Unknown Plant",
                "confidence": 0.0,
                "predictions": [],
                "success": False,
                "error": str(e)
            }
    
    def classify_plant_from_base64(self, base64_image: str) -> Dict[str, Any]:
        """
        Classify a plant from base64 encoded image data.
        
        Args:
            base64_image (str): Base64 encoded image data
            
        Returns:
            Dict[str, Any]: Classification results (same format as classify_plant)
        """
        try:
            # Create data URL for base64 image
            image_data_url = f"data:image/jpeg;base64,{base64_image}"
            return self.classify_plant(image_data_url)
            
        except Exception as e:
            logger.error(f"Base64 plant classification failed: {str(e)}")
            return {
                "plant_name": "Unknown Plant",
                "confidence": 0.0,
                "predictions": [],
                "success": False,
                "error": str(e)
            }


def main():
    """Test function for the inference client."""
    try:
        client = RoboflowInferenceClient()
        print("Roboflow inference client initialized successfully!")
        print("Use classify_plant(image_path) or classify_plant_from_base64(base64_data) methods.")
        
    except Exception as e:
        print(f"Error initializing Roboflow client: {str(e)}")


if __name__ == "__main__":
    main()