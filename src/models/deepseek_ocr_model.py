import os
import torch
import tempfile
from typing import Dict, Any, List
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# Import BaseOCRModel from the same directory
from .base_model import BaseOCRModel

class DeepSeekOCRModel(BaseOCRModel):
    """DeepSeek-OCR model handler for the PDF-OCR-LLM pipeline."""
    
    def __init__(self, model_config: Dict[str, Any], device_manager, inference_config: Dict[str, Any]):
        super().__init__(model_config, device_manager)
        self.device_manager = device_manager
        self.inference_config = inference_config
        
    def load_model(self) -> None:
        """Load DeepSeek-OCR model and tokenizer."""
        print(f"[DeepSeek] Loading model: {self.model_config['model_id']}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config['model_id'], 
                trust_remote_code=True
            )
            
            # Using bfloat16 for efficient GPU usage
            self.model = AutoModel.from_pretrained(
                self.model_config['model_id'],
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
            ).to(self.device_manager.get_device()).eval()
            
            print(f"[DeepSeek] Successfully loaded {self.model_config['name']}")
            
        except Exception as e:
            print(f"[DeepSeek] Failed to load model: {e}")
            raise

    def process_image(self, image: Image.Image, prompt: str = None) -> str:
        """
        Process a single image and return the text.
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # 1. Handle Prompt
        if prompt:
            formatted_prompt = f"<image>\n{prompt}"
        else:
            formatted_prompt = "<image>\n<|grounding|>Convert the document to markdown."

        # 2. Save PIL Image to Temp File
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
            image.save(temp_img.name)
            input_path = temp_img.name

        # 3. Create Temp Output Directory
        with tempfile.TemporaryDirectory() as temp_out_dir:
            try:
                with torch.no_grad():
                    # Run inference and force save results to temp folder
                    self.model.infer(
                        self.tokenizer,
                        prompt=formatted_prompt,
                        image_file=input_path,
                        output_path=temp_out_dir,
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        save_results=True 
                    )

                # 4. SAFER FILE FINDER (Added .mmd support)
                files_in_dir = os.listdir(temp_out_dir)
                
                # Check for all possible text/markdown extensions
                text_files = [f for f in files_in_dir if f.endswith(('.md', '.mmd', '.txt', '.json'))]
                
                result_text = ""
                if text_files:
                    # Priority: .md > .mmd > others
                    md_files = [f for f in text_files if f.endswith(('.md', '.mmd'))]
                    target_file = md_files[0] if md_files else text_files[0]
                    
                    full_path = os.path.join(temp_out_dir, target_file)
                    with open(full_path, 'r', encoding='utf-8') as f:
                        result_text = f.read()
                else:
                    result_text = f"Error: No text output generated. Files found: {files_in_dir}"

            except Exception as e:
                result_text = f"Error during inference: {str(e)}"
            finally:
                if os.path.exists(input_path):
                    os.remove(input_path)

        return result_text

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