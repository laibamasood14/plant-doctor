"""
Knowledge Base Utilities for Plant Care Information
=================================================

This module provides utilities for loading and querying the plant knowledge base
JSON file containing plant-specific care information, common issues, and treatments.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import re

logger = logging.getLogger(__name__)


def _normalize_plant_name(name: str) -> str:
    """
    Normalize plant names for fuzzy matching.
    
    This helps match variants like:
    - "Pothos (Ivy Aerum)"
    - "Money Plant / Golden Pothos (Epipremnum aureum)"
    """
    if not isinstance(name, str):
        return ""
    
    # Lowercase
    name = name.lower()
    # Remove content in parentheses
    name = re.sub(r"\([^)]*\)", " ", name)
    # Replace separators with spaces
    name = re.sub(r"[\/,&\-]+", " ", name)
    # Keep only letters and spaces
    name = re.sub(r"[^a-z\s]", " ", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


class PlantKnowledgeBase:
    """
    Knowledge base manager for plant care information.
    
    This class handles loading and querying plant-specific information from
    the JSON knowledge base including care instructions, common issues,
    symptoms, and treatment recommendations.
    """
    
    def __init__(self, kb_file_path: Optional[str] = None):
        """
        Initialize the plant knowledge base.
        
        Args:
            kb_file_path (Optional[str]): Path to the knowledge base JSON file.
                                        If None, uses default path.
        """
        if kb_file_path is None:
            # Default to the data directory
            current_dir = Path(__file__).parent
            kb_file_path = current_dir / "data" / "Plants_KB_Plant_Doc.json"
        
        self.kb_file_path = Path(kb_file_path)
        self.knowledge_base = {}
        self.load_knowledge_base()
    
    def load_knowledge_base(self) -> None:
        """
        Load the plant knowledge base from JSON file.
        
        Raises:
            FileNotFoundError: If the knowledge base file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        try:
            if not self.kb_file_path.exists():
                raise FileNotFoundError(f"Knowledge base file not found: {self.kb_file_path}")
            
            with open(self.kb_file_path, 'r', encoding='utf-8') as file:
                raw_data = json.load(file)

            # The JSON file is a LIST of dicts, each containing many plants.
            # Flatten into a single dict: {plant_name: plant_data}
            kb: Dict[str, Any] = {}
            if isinstance(raw_data, list):
                for block in raw_data:
                    if isinstance(block, dict):
                        kb.update(block)
            elif isinstance(raw_data, dict):
                kb = raw_data
            else:
                raise ValueError("Unexpected knowledge base JSON structure")

            self.knowledge_base = kb
            
            logger.info(f"Loaded knowledge base with {len(self.knowledge_base)} plants from {self.kb_file_path}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse knowledge base JSON: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {str(e)}")
            raise
    
    def _best_fuzzy_match(self, plant_name: str) -> Tuple[Optional[str], float]:
        """
        Find the best fuzzy match for a plant name in the knowledge base.
        
        Returns:
            (best_key, score) where score is between 0 and 1
        """
        if not plant_name or not isinstance(plant_name, str):
            return None, 0.0

        target_norm = _normalize_plant_name(plant_name)
        if not target_norm:
            return None, 0.0

        target_tokens = set(target_norm.split())
        best_key = None
        best_score = 0.0

        for plant_key in self.knowledge_base.keys():
            key_norm = _normalize_plant_name(plant_key)
            if not key_norm:
                continue

            key_tokens = set(key_norm.split())
            if not key_tokens:
                continue

            # Exact normalized match
            if target_norm == key_norm:
                return plant_key, 1.0

            # Token-based Jaccard similarity
            intersection = target_tokens & key_tokens
            union = target_tokens | key_tokens
            if not union:
                continue
            jaccard = len(intersection) / len(union)

            # Boost score if any token matches strongly (e.g., "pothos")
            token_overlap = len(intersection) / max(len(target_tokens), 1)
            score = max(jaccard, token_overlap)

            if score > best_score:
                best_score = score
                best_key = plant_key

        return best_key, best_score

    def search_plant(self, plant_name: str) -> Optional[Dict[str, Any]]:
        """
        Search for plant information by name (case-insensitive partial matching).
        
        Args:
            plant_name (str): Name of the plant to search for
            
        Returns:
            Optional[Dict[str, Any]]: Plant information if found, None otherwise
        """
        if not plant_name or not isinstance(plant_name, str):
            return None
        
        # First try simple exact / substring match for speed
        plant_name_lower = plant_name.lower().strip()
        
        for plant_key, plant_data in self.knowledge_base.items():
            plant_key_lower = plant_key.lower()
            
            if (plant_name_lower == plant_key_lower or 
                plant_name_lower in plant_key_lower or
                plant_key_lower in plant_name_lower):
                
                logger.info(f"Found direct plant match: {plant_key}")
                return {
                    "plant_name": plant_key,
                    "plant_data": plant_data,
                    "match_confidence": 1.0
                }

        # Fuzzy match using normalized names (handles aliases like Golden Pothos vs Pothos)
        best_key, score = self._best_fuzzy_match(plant_name)
        if best_key and score >= 0.4:  # threshold to avoid bad matches
            logger.info(f"Found fuzzy plant match: '{plant_name}' -> '{best_key}' (score: {score:.2f})")
            return {
                "plant_name": best_key,
                "plant_data": self.knowledge_base[best_key],
                "match_confidence": float(score)
            }
        
        logger.warning(f"No plant found matching: {plant_name}")
        return None
    
    def get_plant_care_info(self, plant_name: str) -> Dict[str, Any]:
        """
        Get comprehensive care information for a specific plant.
        
        Args:
            plant_name (str): Name of the plant
            
        Returns:
            Dict[str, Any]: Plant care information including:
                - plant_name: The matched plant name
                - common_issues: List of common issues and their details
                - general_care: General care recommendations
                - found: Boolean indicating if plant was found
        """
        plant_info = self.search_plant(plant_name)
        
        if not plant_info:
            return {
                "plant_name": plant_name,
                "common_issues": [],
                "general_care": "No specific care information available for this plant.",
                "found": False,
                "confidence": 0.0
            }
        
        plant_data = plant_info["plant_data"]
        match_confidence = float(plant_info.get("match_confidence", 1.0))
        common_issues = []
        
        # Extract common issues and their details
        for issue_name, issue_details in plant_data.items():
            common_issues.append({
                "issue_name": issue_name,
                "symptoms": issue_details.get("symptoms", []),
                "treatment": issue_details.get("treatment", []),
                "prevention": issue_details.get("prevention", [])
            })
        
        # Create general care summary
        general_care = f"Care information available for {len(common_issues)} common issues. "
        if common_issues:
            general_care += f"Common issues include: {', '.join([issue['issue_name'] for issue in common_issues[:3]])}"
            if len(common_issues) > 3:
                general_care += f" and {len(common_issues) - 3} more."
        
        return {
            "plant_name": plant_info["plant_name"],
            "common_issues": common_issues,
            "general_care": general_care,
            "found": True,
            "confidence": match_confidence
        }
    
    def get_treatment_recommendations(self, plant_name: str, disease_name: str = None) -> List[str]:
        """
        Get treatment recommendations for a specific plant and optionally a disease.
        
        IMPORTANT: When disease_name is provided, ONLY returns treatments for that specific
        disease/issue. Does NOT return treatments for other issues like pests, etc.
        
        Args:
            plant_name (str): Name of the plant
            disease_name (Optional[str]): Name of the specific disease/issue
            
        Returns:
            List[str]: List of treatment recommendations (filtered by disease if specified)
        """
        plant_info = self.search_plant(plant_name)
        
        if not plant_info:
            return ["No specific treatment information available for this plant."]
        
        plant_data = plant_info["plant_data"]
        treatments = []
        
        if disease_name:
            # Look for ONLY the specific disease/issue - do NOT return general treatments
            disease_lower = disease_name.lower()
            matched = False
            
            for issue_name, issue_details in plant_data.items():
                if disease_lower in issue_name.lower() or issue_name.lower() in disease_lower:
                    treatments.extend(issue_details.get("treatment", []))
                    matched = True
                    logger.info(f"Found specific treatment for '{disease_name}' in issue '{issue_name}'")
                    break  # Only get treatments for THIS specific issue
            
            if not matched:
                logger.warning(f"No specific treatment found for disease '{disease_name}' in plant '{plant_name}'")
                return [f"No specific treatment information available for '{disease_name}' in the knowledge base."]
        else:
            # If no specific disease requested, collect all available treatments
            for issue_name, issue_details in plant_data.items():
                treatments.extend(issue_details.get("treatment", []))
        
        # Remove duplicates and return unique treatments
        unique_treatments = list(dict.fromkeys(treatments))  # Preserves order while removing duplicates
        return unique_treatments if unique_treatments else ["No specific treatments available."]
    
    def get_prevention_tips(self, plant_name: str, disease_name: str = None) -> List[str]:
        """
        Get prevention tips for a specific plant, optionally filtered by disease.
        
        Args:
            plant_name (str): Name of the plant
            disease_name (Optional[str]): Name of the specific disease/issue to filter by
            
        Returns:
            List[str]: List of prevention tips (filtered by disease if specified)
        """
        plant_info = self.search_plant(plant_name)
        
        if not plant_info:
            return ["No specific prevention information available for this plant."]
        
        plant_data = plant_info["plant_data"]
        prevention_tips = []
        
        if disease_name:
            # Look for ONLY the specific disease/issue
            disease_lower = disease_name.lower()
            matched = False
            
            for issue_name, issue_details in plant_data.items():
                if disease_lower in issue_name.lower() or issue_name.lower() in disease_lower:
                    prevention_tips.extend(issue_details.get("prevention", []))
                    matched = True
                    logger.info(f"Found specific prevention tips for '{disease_name}' in issue '{issue_name}'")
                    break
            
            if not matched:
                logger.warning(f"No specific prevention tips found for disease '{disease_name}' in plant '{plant_name}'")
                return [f"No specific prevention information available for '{disease_name}' in the knowledge base."]
        else:
            # Collect all prevention tips
            for issue_name, issue_details in plant_data.items():
                prevention_tips.extend(issue_details.get("prevention", []))
        
        # Remove duplicates and return unique tips
        unique_tips = list(dict.fromkeys(prevention_tips))  # Preserves order while removing duplicates
        return unique_tips if unique_tips else ["No specific prevention tips available."]
    
    def list_all_plants(self) -> List[str]:
        """
        Get a list of all plant names in the knowledge base.
        
        Returns:
            List[str]: List of all plant names
        """
        return list(self.knowledge_base.keys())
    
    def get_kb_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dict[str, Any]: Statistics including total plants, total issues, etc.
        """
        total_plants = len(self.knowledge_base)
        total_issues = sum(len(plant_data) for plant_data in self.knowledge_base.values())
        
        return {
            "total_plants": total_plants,
            "total_issues": total_issues,
            "kb_file_path": str(self.kb_file_path),
            "loaded": True
        }


def main():
    """Test function for the knowledge base utilities."""
    try:
        kb = PlantKnowledgeBase()
        stats = kb.get_kb_stats()
        print(f"Knowledge base loaded successfully!")
        print(f"Total plants: {stats['total_plants']}")
        print(f"Total issues: {stats['total_issues']}")
        
        # Test search
        test_plants = ["Aloe Vera", "Jade Plant", "Tomato"]
        for plant in test_plants:
            info = kb.get_plant_care_info(plant)
            print(f"\n{plant}: {'Found' if info['found'] else 'Not found'}")
            if info['found']:
                print(f"  Common issues: {len(info['common_issues'])}")
        
        # Test specific disease treatment lookup
        print("\n--- Testing Specific Disease Treatment ---")
        treatments = kb.get_treatment_recommendations("Pothos", "root rot")
        print(f"Pothos - Root Rot treatments: {treatments}")
        
        treatments = kb.get_treatment_recommendations("Pothos", "overwatering")
        print(f"Pothos - Overwatering treatments: {treatments}")
        
    except Exception as e:
        print(f"Error testing knowledge base: {str(e)}")


if __name__ == "__main__":
    main()