"""
Plant Diagnosis Pipeline
=======================

This module orchestrates the complete plant diagnosis pipeline combining:
1. Roboflow plant classification
2. Groq disease detection
3. Knowledge base lookup for plant-specific advice
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference import RoboflowInferenceClient
from kb_utils import PlantKnowledgeBase
from main import LeafDiseaseDetector

logger = logging.getLogger(__name__)


class PlantDiagnosisPipeline:
    """
    Complete plant diagnosis pipeline that combines plant identification,
    disease detection, and knowledge base recommendations.
    """
    
    def __init__(self, roboflow_api_key: Optional[str] = None, groq_api_key: Optional[str] = None):
        """
        Initialize the plant diagnosis pipeline.
        
        Args:
            roboflow_api_key (Optional[str]): Roboflow API key for plant classification
            groq_api_key (Optional[str]): Groq API key for disease detection
        """
        try:
            # Initialize Roboflow client for plant classification
            self.roboflow_client = RoboflowInferenceClient(api_key=roboflow_api_key)
            logger.info("Roboflow client initialized")
            
            # Initialize Groq client for disease detection
            self.disease_detector = LeafDiseaseDetector(api_key=groq_api_key)
            logger.info("Disease detector initialized")
            
            # Initialize knowledge base
            self.knowledge_base = PlantKnowledgeBase()
            logger.info("Knowledge base initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize diagnosis pipeline: {str(e)}")
            raise
    
    def diagnose_plant(self, image_path: str) -> Dict[str, Any]:
        """
        Complete plant diagnosis pipeline.
        
        This method performs the full diagnosis workflow:
        1. Classify plant species using Roboflow
        2. Detect diseases using Groq AI
        3. Lookup plant-specific care information
        4. Combine all results into comprehensive response
        
        Args:
            image_path (str): Path to plant image file or base64 encoded image
            
        Returns:
            Dict[str, Any]: Comprehensive diagnosis results containing:
                - plant_name: Identified plant species
                - health_status: Overall health assessment
                - disease_info: Disease detection results
                - kb_advice: Knowledge base recommendations
                - treatments: Treatment recommendations
                - confidence: Overall confidence score
                - pipeline_success: Boolean indicating success
        """
        try:
            logger.info("Starting complete plant diagnosis pipeline")
            
            # Step 1: Plant Classification using Roboflow
            logger.info("Step 1: Classifying plant species...")
            classification_result = self.roboflow_client.classify_plant(image_path)
            
            plant_name = classification_result.get("plant_name", "Unknown Plant")
            classification_confidence = classification_result.get("confidence", 0.0)
            classification_success = classification_result.get("success", False)
            
            logger.info(f"Plant classified as: {plant_name} (confidence: {classification_confidence:.2f})")
            
            # Step 2: Disease Detection using Groq
            logger.info("Step 2: Detecting diseases...")
            
            # Handle both file path and base64 input for disease detection
            if os.path.exists(image_path):
                # File path - convert to base64 for disease detection
                import base64
                with open(image_path, 'rb') as image_file:
                    image_bytes = image_file.read()
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
            else:
                # Assume it's already base64
                base64_image = image_path
            
            disease_result = self.disease_detector.analyze_leaf_image_base64(base64_image)
            
            # Step 3: Knowledge Base Lookup
            logger.info("Step 3: Looking up plant care information...")
            kb_info = self.knowledge_base.get_plant_care_info(plant_name)
            
            # Step 4: Get Treatment Recommendations
            treatment_recommendations = []
            if disease_result.get("disease_detected", False):
                disease_name = disease_result.get("disease_name", "")
                treatment_recommendations = self.knowledge_base.get_treatment_recommendations(
                    plant_name, disease_name
                )
            
            # If no specific treatments found, get general treatments
            if not treatment_recommendations:
                treatment_recommendations = self.knowledge_base.get_treatment_recommendations(plant_name)
            
            # Step 5: Compile Results
            diagnosis_result = {
                "plant_name": plant_name,
                "health_status": "unhealthy" if disease_result.get("disease_detected", False) else "healthy",
                "disease_info": {
                    "disease_detected": disease_result.get("disease_detected", False),
                    "disease_name": disease_result.get("disease_name"),
                    "disease_type": disease_result.get("disease_type"),
                    "severity": disease_result.get("severity"),
                    "confidence": disease_result.get("confidence"),
                    "symptoms": disease_result.get("symptoms", []),
                    "possible_causes": disease_result.get("possible_causes", [])
                },
                "classification_info": {
                    "plant_identified": classification_success,
                    "classification_confidence": classification_confidence,
                    "roboflow_predictions": classification_result.get("predictions", [])
                },
                "kb_advice": {
                    "plant_found_in_kb": kb_info.get("found", False),
                    "general_care": kb_info.get("general_care", ""),
                    "common_issues": kb_info.get("common_issues", []),
                    "prevention_tips": self.knowledge_base.get_prevention_tips(plant_name)
                },
                "treatments": {
                    "disease_treatments": disease_result.get("treatment", []),
                    "kb_treatments": treatment_recommendations,
                    "combined_treatments": list(set(
                        disease_result.get("treatment", []) + treatment_recommendations
                    ))
                },
                "confidence": {
                    "classification": classification_confidence,
                    "disease_detection": disease_result.get("confidence", 0.0),
                    "overall": (classification_confidence + disease_result.get("confidence", 0.0)) / 2
                },
                "pipeline_success": classification_success and True,  # Disease detection can fail but pipeline continues
                "timestamp": disease_result.get("analysis_timestamp", "")
            }
            
            logger.info("Plant diagnosis pipeline completed successfully")
            return diagnosis_result
            
        except Exception as e:
            logger.error(f"Plant diagnosis pipeline failed: {str(e)}")
            return {
                "plant_name": "Unknown Plant",
                "health_status": "unknown",
                "disease_info": {},
                "classification_info": {},
                "kb_advice": {},
                "treatments": {},
                "confidence": {"overall": 0.0},
                "pipeline_success": False,
                "error": str(e)
            }
    
    def diagnose_plant_from_base64(self, base64_image: str) -> Dict[str, Any]:
        """
        Diagnose plant from base64 encoded image data.
        
        Args:
            base64_image (str): Base64 encoded image data
            
        Returns:
            Dict[str, Any]: Same format as diagnose_plant
        """
        return self.diagnose_plant(base64_image)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get status of all pipeline components.
        
        Returns:
            Dict[str, Any]: Status information for all components
        """
        try:
            kb_stats = self.knowledge_base.get_kb_stats()
            
            return {
                "roboflow_client": "initialized",
                "disease_detector": "initialized", 
                "knowledge_base": kb_stats,
                "pipeline_ready": True
            }
        except Exception as e:
            return {
                "pipeline_ready": False,
                "error": str(e)
            }


def main():
    """Test function for the diagnosis pipeline."""
    try:
        pipeline = PlantDiagnosisPipeline()
        status = pipeline.get_pipeline_status()
        print("Plant Diagnosis Pipeline initialized successfully!")
        print(f"Pipeline status: {status}")
        
        # Test with a sample image if available
        test_image = "Media/brown-spot-4 (1).jpg"
        if os.path.exists(test_image):
            print(f"\nTesting with sample image: {test_image}")
            result = pipeline.diagnose_plant(test_image)
            print(f"Plant: {result['plant_name']}")
            print(f"Health: {result['health_status']}")
            print(f"Success: {result['pipeline_success']}")
        
    except Exception as e:
        print(f"Error testing diagnosis pipeline: {str(e)}")


if __name__ == "__main__":
    main()