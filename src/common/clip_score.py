"""
CLIP-based evaluation metrics for image-text alignment.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, List
from PIL import Image
import clip

class CLIPScorer:
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        """Initialize CLIP model for scoring."""
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        
    def compute_similarity(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """Compute CLIP similarity between images and texts."""
        text_tokens = clip.tokenize(texts).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text_tokens)
            
            # Normalize features
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute similarity
            similarity = torch.matmul(image_features, text_features.T)
            
        return similarity
    
    def score_batch(self, images: torch.Tensor, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score a batch of images against texts."""
        similarity = self.compute_similarity(images, texts)
        return similarity.mean(), similarity.std() 