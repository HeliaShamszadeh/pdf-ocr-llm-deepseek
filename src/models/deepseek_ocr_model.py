import os
import torch
import tempfile
import json  # Required for JSON formatting
import re    # Required for cleaning code fences
from typing import Dict, Any, List
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# Import BaseOCRModel from the same directory
from .base_model import BaseOCRModel

class DeepSeekOCRModel(BaseOCRModel):
    def __init__(self, model_config: Dict[str, Any], device_manager, inference_config: Dict[str, Any]):
        super().__init__(model_config, device_manager)
        self.device_manager = device_manager
        self.inference_config = inference_config
        
    def load_model(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config['model_id'], 
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_config['model_id'],
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
            ).to(self.device_manager.get_device()).eval()
        except Exception as e:
            print(f"[DeepSeek] Failed to load model: {e}")
            raise

    def process_image(self, image: Image.Image, prompt: str = None) -> str:
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Model not loaded.")

        formatted_prompt = f"<image>\n{prompt}" if prompt else "<image>\n<|grounding|>Convert the document to markdown."

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
            image.save(temp_img.name)
            input_path = temp_img.name

        with tempfile.TemporaryDirectory() as temp_out_dir:
            try:
                with torch.no_grad():
                    self.model.infer(
                        self.tokenizer,
                        prompt=formatted_prompt,
                        image_file=input_path,
                        output_path=temp_out_dir,
                        save_results=True 
                    )

                files_in_dir = os.listdir(temp_out_dir)
                text_files = [f for f in files_in_dir if f.endswith(('.md', '.mmd', '.txt', '.json'))]
                
                if text_files:
                    target_file = next((f for f in text_files if f.endswith(('.md', '.mmd'))), text_files[0])
                    with open(os.path.join(temp_out_dir, target_file), 'r', encoding='utf-8') as f:
                        raw_text = f.read()
                    
                    # Clean markdown blocks
                    cleaned = re.sub(r'^```(?:json|markdown|text)?\s*\n', '', raw_text, flags=re.MULTILINE)
                    cleaned = re.sub(r'\n```\s*$', '', cleaned, flags=re.MULTILINE).strip()
                    return cleaned
                return "Error: No text found."
            finally:
                if os.path.exists(input_path): os.remove(input_path)

    def process_batch(self, images: List[Image.Image], prompts: List[str] = None) -> List[str]:
        results = []
        if prompts is None:
            prompts = [None] * len(images)
        
        for img, p in zip(images, prompts):
            results.append(self.process_image(img, p))
        return results

    def unload_model(self):
        """Standard cleanup for GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()