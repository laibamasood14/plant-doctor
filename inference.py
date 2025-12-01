"""
Roboflow Inference Client for Plant Classification
================================================

This module provides a client for interacting with Roboflow's inference API
to classify plant species from images using the custom workflow.
"""

import os
import logging
import base64
import tempfile
from typing import Dict, Optional, Any, List
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import statistics

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class RoboflowInferenceClient:
    """
    Client for Roboflow inference API to identify plant species using workflow.
    
    This class handles communication with Roboflow's serverless inference
    endpoint to classify plant images and return species identification results
    with advanced confidence filtering.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        workspace_name: str = "laiba-masood-tyq7q",
        model_id: str = "identify-plant-zvd1y/1",
        min_confidence: float = 0.7,
        confidence_method: str = "adaptive"
    ):
        """
        Initialize the Roboflow inference client.
        
        Args:
            api_key (Optional[str]): Roboflow API key. If None, will attempt to
                                   load from ROBOFLOW_API_KEY environment variable.
            workspace_name (str): Roboflow workspace name
            model_id (str): Model ID in format "project-name/version" (e.g., "identify-plant-zvd1y/1")
            min_confidence (float): Minimum confidence threshold (0-1)
            confidence_method (str): Confidence filtering method:
                                   - "adaptive": Uses statistical analysis
                                   - "strict": Only top prediction if above threshold
                                   - "weighted": Weighted average of top predictions
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY not found. Please set it in your .env file or pass it as a parameter."
            )
        
        # Validate API key format (Roboflow keys typically start with specific prefixes)
        if not isinstance(self.api_key, str) or len(self.api_key.strip()) == 0:
            raise ValueError("ROBOFLOW_API_KEY is invalid or empty")
        
        self.workspace_name = workspace_name
        self.model_id = model_id
        self.min_confidence = min_confidence
        self.confidence_method = confidence_method
        
        # Initialize InferenceHTTPClient exactly as per Roboflow example
        # 2. Connect to your workflow
        try:
            self.client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=self.api_key
            )
            logger.info(
                f"Roboflow inference client initialized successfully with "
                f"workspace: {workspace_name}, model: {model_id}"
            )
        except TypeError as e:
            # Handle case where InferenceHTTPClient might have different parameters
            logger.error(f"Failed to initialize InferenceHTTPClient - parameter error: {str(e)}")
            raise ValueError(
                f"Failed to initialize Roboflow client. "
                f"Please check your inference-sdk version: pip install --upgrade inference-sdk. "
                f"Error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize InferenceHTTPClient: {str(e)}")
            raise ValueError(f"Failed to initialize Roboflow client: {str(e)}")
    
    def _apply_advanced_confidence_filtering(
        self, 
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Apply advanced confidence filtering methods to select the best prediction.
        
        Args:
            predictions: List of prediction dictionaries with 'class' and 'confidence'
            
        Returns:
            Dict with filtered plant_name, confidence, and method used
        """
        if not predictions:
            return {
                "plant_name": "Unknown Plant",
                "confidence": 0.0,
                "method": "none"
            }
        
        # Extract confidence scores
        confidences = [float(p.get('confidence', 0.0)) for p in predictions]
        top_confidence = max(confidences) if confidences else 0.0
        top_prediction = predictions[confidences.index(top_confidence)] if confidences else predictions[0]
        
        if self.confidence_method == "adaptive":
            # Adaptive method: Statistical analysis
            if len(confidences) >= 3:
                mean_conf = statistics.mean(confidences[:3])
                std_conf = statistics.stdev(confidences[:3]) if len(confidences) >= 2 else 0
                
                # If top confidence is significantly higher than others (2 std devs)
                if top_confidence >= mean_conf + (2 * std_conf) and top_confidence >= self.min_confidence:
                    return {
                        "plant_name": top_prediction.get('class', 'Unknown Plant'),
                        "confidence": top_confidence,
                        "method": "adaptive_statistical"
                    }
                
                # If top 2 are close, check if they're both high confidence
                if len(confidences) >= 2:
                    top2_avg = statistics.mean(confidences[:2])
                    if top2_avg >= self.min_confidence and confidences[0] - confidences[1] < 0.15:
                        # Top 2 are close, use weighted average
                        weighted_conf = (confidences[0] * 0.7 + confidences[1] * 0.3)
                        return {
                            "plant_name": top_prediction.get('class', 'Unknown Plant'),
                            "confidence": weighted_conf,
                            "method": "adaptive_weighted"
                        }
            
            # Fallback: Use top if above threshold
            if top_confidence >= self.min_confidence:
                return {
                    "plant_name": top_prediction.get('class', 'Unknown Plant'),
                    "confidence": top_confidence,
                    "method": "adaptive_threshold"
                }
        
        elif self.confidence_method == "strict":
            # Strict method: Only accept if above threshold
            if top_confidence >= self.min_confidence:
                return {
                    "plant_name": top_prediction.get('class', 'Unknown Plant'),
                    "confidence": top_confidence,
                    "method": "strict"
                }
        
        elif self.confidence_method == "weighted":
            # Weighted method: Average of top 3 predictions
            top_n = min(3, len(confidences))
            weights = [0.5, 0.3, 0.2][:top_n]
            weighted_sum = sum(confidences[i] * weights[i] for i in range(top_n))
            total_weight = sum(weights[:top_n])
            weighted_conf = weighted_sum / total_weight if total_weight > 0 else 0
            
            if weighted_conf >= self.min_confidence:
                return {
                    "plant_name": top_prediction.get('class', 'Unknown Plant'),
                    "confidence": weighted_conf,
                    "method": "weighted_average"
                }
        
        # Default: Return top prediction even if below threshold (for logging)
        return {
            "plant_name": top_prediction.get('class', 'Unknown Plant'),
            "confidence": top_confidence,
            "method": "default"
        }
    
    def classify_plant(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a plant image and return species identification using workflow API.
        
        Args:
            image_path (str): Path to the image file (absolute or relative path)
            
        Returns:
            Dict[str, Any]: Classification results containing:
                - plant_name: Identified plant species name
                - confidence: Confidence score (0-1)
                - predictions: Full prediction details
                - success: Boolean indicating if classification was successful
                - confidence_method: Method used for confidence filtering
        """
        try:
            # Convert to absolute path if relative
            if not os.path.isabs(image_path):
                image_path = os.path.abspath(image_path)
            
            # Verify file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            logger.info(f"Starting plant classification for: {image_path[:50] if len(image_path) > 50 else image_path}...")
            
            # 3. Run inference on the image using the model
            try:
                # Use infer method for model inference (not workflow)
                result = self.client.infer(
                    image_path,
                    model_id=self.model_id
                )
                logger.info(f"Inference result received: {type(result)}")
                logger.debug(f"Inference result content: {result}")
            except AttributeError as e:
                error_msg = (
                    f"infer method not found. Please upgrade inference-sdk: "
                    f"pip install --upgrade inference-sdk. Error: {str(e)}"
                )
                logger.error(error_msg)
                raise AttributeError(error_msg)
            except Exception as e:
                error_msg = f"Failed to run inference: {str(e)}"
                logger.error(error_msg)
                # Provide helpful error messages for common issues
                error_str = str(e).lower()
                if "401" in error_str or "unauthorized" in error_str:
                    error_msg += " (Authentication failed - check your API key)"
                elif "404" in error_str or "not found" in error_str:
                    error_msg += f" (Model not found - verify workspace: {self.workspace_name}, model: {self.model_id})"
                elif "timeout" in error_str:
                    error_msg += " (Request timeout - check your internet connection)"
                raise Exception(error_msg)
            
            # Extract predictions from model inference result
            # Model results structure may vary, so we handle multiple formats
            predictions = []
            
            # Log the raw result structure for debugging (INFO level for visibility)
            logger.info(f"Raw inference result type: {type(result)}")
            if isinstance(result, dict):
                logger.info(f"Raw workflow result keys: {list(result.keys())}")
                # Log a sample of the result structure
                logger.info(f"Raw result sample: {str(result)[:300]}")
            else:
                logger.info(f"Raw result (first 300 chars): {str(result)[:300]}")
            
            if isinstance(result, dict):
                # Try different possible result structures
                if 'predictions' in result:
                    predictions = result.get('predictions', [])
                    logger.debug("Found predictions in 'predictions' key")
                elif 'results' in result:
                    predictions = result.get('results', [])
                    logger.debug("Found predictions in 'results' key")
                elif 'output' in result:
                    output = result.get('output', {})
                    if isinstance(output, list):
                        predictions = output
                        logger.debug("Found predictions in 'output' as list")
                    elif isinstance(output, dict) and 'predictions' in output:
                        predictions = output.get('predictions', [])
                        logger.debug("Found predictions in 'output.predictions'")
                    elif isinstance(output, dict):
                        # Try to find predictions in output dict
                        for key in ['predictions', 'results', 'classes', 'top_predictions']:
                            if key in output and isinstance(output[key], list):
                                predictions = output[key]
                                logger.debug(f"Found predictions in 'output.{key}'")
                                break
                elif 'image' in result:
                    # Workflow might return results nested under image key
                    image_data = result.get('image', {})
                    if isinstance(image_data, dict):
                        for key in ['predictions', 'results', 'classes']:
                            if key in image_data and isinstance(image_data[key], list):
                                predictions = image_data[key]
                                logger.debug(f"Found predictions in 'image.{key}'")
                                break
                else:
                    # Try to find any list of predictions in nested structure
                    for key, value in result.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], dict) and ('class' in value[0] or 'confidence' in value[0] or 'name' in value[0]):
                                predictions = value
                                logger.debug(f"Found predictions in '{key}' key")
                                break
                        elif isinstance(value, dict):
                            # Check nested dicts
                            for nested_key in ['predictions', 'results', 'classes']:
                                if nested_key in value and isinstance(value[nested_key], list):
                                    predictions = value[nested_key]
                                    logger.debug(f"Found predictions in '{key}.{nested_key}'")
                                    break
                            if predictions:
                                break
            elif isinstance(result, list):
                # If result itself is a list
                predictions = result
                logger.debug("Result is a list, using directly as predictions")
            
            # Log what we found
            logger.info(f"Extracted {len(predictions)} predictions from model inference result")
            
            if not predictions:
                logger.error("No predictions returned from Roboflow model")
                logger.error(f"Raw result type: {type(result)}")
                if isinstance(result, dict):
                    logger.error(f"Raw result keys: {list(result.keys())}")
                    # Log the full structure for debugging
                    import json
                    try:
                        logger.error(f"Raw result JSON: {json.dumps(result, indent=2, default=str)[:1000]}")
                    except:
                        logger.error(f"Raw result (string): {str(result)[:1000]}")
                else:
                    logger.error(f"Raw result: {str(result)[:1000]}")
                return {
                    "plant_name": "Unknown Plant",
                    "confidence": 0.0,
                    "predictions": [],
                    "success": False,
                    "error": "No predictions found in model inference result",
                    "raw_result": result
                }
            
            # Normalize prediction format - handle different field names
            normalized_predictions = []
            for pred in predictions:
                if isinstance(pred, dict):
                    # Normalize field names (class/name, confidence/score)
                    normalized = {}
                    normalized['class'] = pred.get('class') or pred.get('name') or pred.get('label') or pred.get('plant_name') or 'Unknown'
                    # Handle confidence in different formats (0-1, 0-100, percentage)
                    conf = pred.get('confidence') or pred.get('score') or pred.get('prob') or 0.0
                    if isinstance(conf, str):
                        conf = float(conf.replace('%', '')) / 100.0 if '%' in conf else float(conf)
                    elif conf > 1.0:  # Assume 0-100 scale
                        conf = conf / 100.0
                    normalized['confidence'] = float(conf)
                    normalized_predictions.append(normalized)
                elif isinstance(pred, str):
                    # If prediction is just a string (plant name)
                    normalized_predictions.append({
                        'class': pred,
                        'confidence': 1.0  # Default confidence if not provided
                    })
            
            predictions = normalized_predictions
            
            # Sort predictions by confidence (descending)
            if predictions and isinstance(predictions[0], dict) and 'confidence' in predictions[0]:
                predictions = sorted(predictions, key=lambda x: float(x.get('confidence', 0)), reverse=True)
                logger.debug(f"Top prediction: {predictions[0].get('class')} (confidence: {predictions[0].get('confidence'):.2f})")
            
            # Apply advanced confidence filtering
            filtered_result = self._apply_advanced_confidence_filtering(predictions)
            
            # Check if confidence meets threshold
            success = filtered_result['confidence'] >= self.min_confidence
            
            if not success:
                logger.warning(
                    f"Classification confidence {filtered_result['confidence']:.2f} below threshold {self.min_confidence}. "
                    f"Plant: {filtered_result['plant_name']}"
                )
            
            logger.info(
                f"Plant classified as: {filtered_result['plant_name']} "
                f"(confidence: {filtered_result['confidence']:.2f}, "
                f"method: {filtered_result['method']}, "
                f"success: {success})"
            )
            
            return {
                "plant_name": filtered_result['plant_name'],
                "confidence": filtered_result['confidence'],
                "predictions": predictions,
                "success": success,
                "confidence_method": filtered_result['method'],
                "raw_result": result
            }
            
        except FileNotFoundError as e:
            logger.error(f"Image file not found: {str(e)}")
            return {
                "plant_name": "Unknown Plant",
                "confidence": 0.0,
                "predictions": [],
                "success": False,
                "error": f"Image file not found: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Plant classification failed: {str(e)}", exc_info=True)
            # Log more details about the error
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "workspace_name": self.workspace_name,
                "model_id": self.model_id,
                "api_key_set": bool(self.api_key)
            }
            logger.error(f"Error details: {error_details}")
            return {
                "plant_name": "Unknown Plant",
                "confidence": 0.0,
                "predictions": [],
                "success": False,
                "error": str(e),
                "error_details": error_details
            }
    
    def classify_plant_from_base64(self, base64_image: str) -> Dict[str, Any]:
        """
        Classify a plant from base64 encoded image data.
        
        Args:
            base64_image (str): Base64 encoded image data (with or without data URL prefix)
            
        Returns:
            Dict[str, Any]: Classification results (same format as classify_plant)
        """
        try:
            # Clean base64 string (remove data URL prefix if present)
            if base64_image.startswith('data:'):
                base64_image = base64_image.split(',', 1)[1]
            
            # Decode base64 to bytes
            try:
                image_bytes = base64.b64decode(base64_image)
            except Exception as e:
                logger.error(f"Failed to decode base64 image: {str(e)}")
                return {
                    "plant_name": "Unknown Plant",
                    "confidence": 0.0,
                    "predictions": [],
                    "success": False,
                    "error": f"Invalid base64 image data: {str(e)}"
                }
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name
            
            try:
                # Classify using the temporary file
                result = self.classify_plant(temp_path)
                return result
            finally:
                # Clean up temporary file
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_path}: {str(e)}")
            
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
        client = RoboflowInferenceClient(
            workspace_name="laiba-masood-tyq7q",
            model_id="identify-plant-zvd1y/1",
            min_confidence=0.7,
            confidence_method="adaptive"
        )
        print("Roboflow inference client initialized successfully!")
        print("Use classify_plant(image_path) or classify_plant_from_base64(base64_data) methods.")
        print(f"Confidence method: adaptive (min threshold: 0.7)")
        
    except Exception as e:
        print(f"Error initializing Roboflow client: {str(e)}")


if __name__ == "__main__":
    main()