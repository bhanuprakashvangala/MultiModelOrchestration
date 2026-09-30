"""
Transformer-based Domain Classifier for Intelligent LLM Routing
Uses a pre-trained BERT model to classify prompts into biology, chemistry, or materials science domains
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from typing import Tuple, Dict, List, Optional
import logging
from enum import Enum
import json
import os
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DomainType(Enum):
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    MATERIALS = "materials"
    GENERAL = "general"

@dataclass
class ClassificationResult:
    domain: DomainType
    confidence: float
    probabilities: Dict[str, float]
    features: Optional[Dict[str, float]] = None

class LightweightDomainClassifier:
    """
    Lightweight version using DistilBERT for faster inference
    """
    
    def __init__(self):
        """Initialize with DistilBERT for faster performance"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use DistilBERT for faster inference
        model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load or create fine-tuned model
        self.model = self._load_or_create_model(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_or_create_model(self, model_name: str):
        """Load fine-tuned model or create new one"""
        model_path = "models/domain_classifier_distilbert"
        
        if os.path.exists(model_path):
            try:
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                logger.info("Loaded fine-tuned domain classifier")
                return model
            except Exception as e:
                logger.warning(f"Could not load fine-tuned model: {e}")
        
        # Create new model with 4 classes
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=4,
            problem_type="single_label_classification"
        )
        
        # Set label mappings
        model.config.id2label = {
            0: "biology",
            1: "chemistry", 
            2: "materials",
            3: "general"
        }
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}
        
        return model
    
    def classify(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Quick classification method
        Returns: (domain, confidence, probability_dict)
        """
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            
            predicted_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_idx].item()
            
            domain_mapping = {
                0: "biology",
                1: "chemistry",
                2: "materials",
                3: "general"
            }
            
            prob_dict = {
                "biology": probabilities[0, 0].item(),
                "chemistry": probabilities[0, 1].item(),
                "materials": probabilities[0, 2].item(),
                "general": probabilities[0, 3].item()
            }
            
            return domain_mapping[predicted_idx], confidence, prob_dict


# Global instance for easy access
domain_classifier = None

def get_domain_classifier(use_lightweight: bool = True) -> Optional[object]:
    """Get or create the global domain classifier instance"""
    global domain_classifier
    
    if domain_classifier is None:
        try:
            domain_classifier = LightweightDomainClassifier()
            logger.info("Domain classifier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize domain classifier: {e}")
            return None
    
    return domain_classifier