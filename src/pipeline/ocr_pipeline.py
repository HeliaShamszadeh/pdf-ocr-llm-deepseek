from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from PIL import Image
import logging
from tqdm import tqdm
import json

from ..config.config_manager import ConfigManager
from ..utils.device_manager import DeviceManager
from ..processors.pdf_processor import PDFProcessor
from ..models.model_factory import ModelFactory
from ..models.base_model import BaseOCRModel

logger = logging.getLogger(__name__)

class OCRPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        logger.info("Initializing OCR Pipeline")
        self.config_manager = ConfigManager(config_path)
        device_config = self.config_manager.get_device_config()
        self.device_manager = DeviceManager(device_config)
        ocr_config = self.config_manager.get_ocr_config()
        self.pdf_processor = PDFProcessor(ocr_config)
        self.inference_config = self.config_manager.get_inference_config()
        self.current_model: Optional[BaseOCRModel] = None
        self.current_model_name: Optional[str] = None
        logger.info("OCR Pipeline initialized successfully")

    def process_pdf(self, pdf_path: Union[str, Path], model_name: str = None, output_path: Union[str, Path] = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        if model_name: self.load_model(model_name)
        if self.current_model is None: raise RuntimeError("No model loaded.")

        images = self.pdf_processor.pdf_to_images(pdf_path)
        json_output_data = {}

        for idx, image in enumerate(tqdm(images, desc="Processing pages"), start=1):
            try:
                text = self.current_model.process_image(image, prompt)
                json_output_data[f"page{idx}"] = text.strip()
            except Exception as e:
                logger.error(f"Failed to process page {idx}: {e}")
                json_output_data[f"page{idx}"] = f"Error: {str(e)}"

        full_json_text = json.dumps(json_output_data, ensure_ascii=False, indent=4)

        if output_path:
            output_path = Path(output_path).with_suffix('.json')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_json_text)

        return {
            'full_text': full_json_text, 
            'output_file': str(output_path) if output_path else None,
            'num_pages': len(images)
        }

    def process_image(self, image_path: Union[str, Path], model_name: str = None, output_path: Union[str, Path] = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        image_path = Path(image_path)
        if model_name: self.load_model(model_name)
        if self.current_model is None: raise RuntimeError("No model loaded.")

        image = self.pdf_processor.load_image(image_path)
        text = self.current_model.process_image(image, prompt)

        # Force JSON structure for single images
        json_output_data = {"page1": text.strip()}
        full_json_text = json.dumps(json_output_data, ensure_ascii=False, indent=4)

        if output_path:
            output_path = Path(output_path).with_suffix('.json')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_json_text)

        return {
            'full_text': full_json_text,
            'output_file': str(output_path) if output_path else None
        }

    def load_model(self, model_name: str, custom_config=None):
        if self.current_model_name == model_name and self.current_model: return
        if self.current_model: self.current_model.unload_model()
        model_config = self.config_manager.get_model_by_name(model_name)
        self.current_model = ModelFactory.create_model(model_config, self.device_manager, self.inference_config)
        self.current_model.load_model()
        self.current_model_name = model_name

    def list_available_models(self): return self.config_manager.get_all_models()
    def cleanup(self): 
        if self.current_model: self.current_model.unload_model()