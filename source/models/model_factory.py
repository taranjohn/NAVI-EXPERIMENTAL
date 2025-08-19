import torch
from abc import ABC, abstractmethod
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, \
    AutoProcessor
from PIL import Image

import ollama
import base64

class VLMAdapter(ABC):
    """Abstract Base Class for a Vision-Language Model adapter."""

    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.model = None
        self.processor = None  # Use 'processor' for models like Idefics2
        # Prioritize device from config, fallback to auto-detection
        self.device = model_config.get('device', self._get_device())
        print(f"Adapter for '{self.model_config['name']}' will use device: {self.device}")

    def _get_device(self) -> str:
        """Detects and returns the most appropriate device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        # Check for Apple's Metal Performance Shaders (MPS)
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @abstractmethod
    def load_model(self):
        """Loads the model and tokenizer/processor."""
        pass


    @abstractmethod
    def predict(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        """Runs inference on a single image and prompt."""
        pass

class StandardPipelineAdapter(VLMAdapter):
    """Adapter for Hugging Face models compatible with the 'image-text-to-text' pipeline."""
    def load_model(self):
        # The device is now set in the VLMAdapter's __init__
        if self.device == "cuda":
            torch_dtype = torch.float16
        elif self.device == "mps":
            # Use float32 on MPS for numerical stability, mimicking the original script's behavior
            torch_dtype = torch.float32
        else:  # cpu
            torch_dtype = torch.bfloat16

        print(f"Loading model '{self.model_config['id']}' via 'image-text-to-text' pipeline with dtype {torch_dtype}...")
        try:
            self.model = pipeline(
                "image-text-to-text",
                model=self.model_config['id'],
                device=self.device,
                torch_dtype=torch_dtype,
            )
            print("Model loaded successfully.")
        except Exception as e:
            raise IOError(f"Fatal error loading pipeline for {self.model_config['name']}: {e}")

    def predict(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": f"{prompt}\nWhat is this image?"}
                ]
            }
        ]
        try:
            # The pipeline call is simplified to just pass the messages list.
            output = self.model(messages, max_new_tokens=max_new_tokens)
            # FIX: Correctly parse the chat-like output structure.
            # The model's actual response is the last content element in the generated text list.
            return output[0]["generated_text"][-1]["content"]
        except Exception as e:
            # Added more context to the error message.
            raise RuntimeError(f"Inference failed for image {image_path} using 'image-text-to-text' pipeline: {e}")


class QwenVLAdapter(VLMAdapter):
    """Adapter for the Qwen2.5-VL model series."""

    def load_model(self):
        """Loads the Qwen model and processor."""
        if self.device == "cuda":
            torch_dtype = torch.float16
        elif self.device == "mps":
            # For MPS, float32 is often more stable, but float16 can be used.
            # Sticking with float32 for consistency with your other adapter.
            torch_dtype = torch.float32
        else:  # cpu
            torch_dtype = torch.bfloat16

        print(f"Loading Qwen model '{self.model_config['id']}' with dtype {torch_dtype}...")

        try:
            # Load the specific model class for Qwen2.5-VL
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_config['id'],
                torch_dtype=torch_dtype,
                device_map=self.device, # Use device_map for better memory management
            )
            # Load the corresponding processor
            # Often, the processor can be loaded from the same model ID.
            self.processor = AutoProcessor.from_pretrained(self.model_config['id'])
            print("Qwen model and processor loaded successfully.")
        except Exception as e:
            raise IOError(f"Fatal error loading Qwen model for {self.model_config['name']}: {e}")

    def predict(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        """Runs inference using the Qwen model's specific processing steps."""
        if not self.model or not self.processor:
            raise RuntimeError("Model is not loaded. Please call load_model() first.")

        try:
            # 1. Open the image
            image = Image.open(image_path).convert("RGB")

            # 2. Format the input messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # 3. Process text and image inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[image], # Pass the PIL image object directly
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)

            # 4. Generate token IDs
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False # Using do_sample=False for more deterministic output
            )

            # 5. Trim input tokens and decode the output
            # This logic correctly removes the prompt tokens from the generated output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # batch_decode returns a list, so we return the first element.
            return response[0]
        except Exception as e:
            raise RuntimeError(f"Inference failed for image {image_path} with Qwen model: {e}")


class OllamaAdapter(VLMAdapter):
    """Adapter for Vision-Language Models served via Ollama."""

    def load_model(self):
        """
        Verifies that the Ollama service is running and the specified model is available.
        For Ollama, the model is managed by the service, not loaded into this class instance.
        """
        model_id = self.model_config['id']
        print(f"Checking for Ollama model: '{model_id}'...")
        try:
            # Get the list of models available locally in the Ollama service
            # --- FIX ---
            # The key for the model identifier in the ollama library response is 'model', not 'name'.
            available_models = [m['model'] for m in ollama.list()['models']]

            # Check if the requested model (e.g., "llava:latest") is in the list
            if model_id not in available_models:
                raise RuntimeError(
                    f"Model '{model_id}' not found in Ollama. "
                    f"Please ensure Ollama is running and you have pulled the model with `ollama pull {model_id}`."
                )

            # If the model is found, we store its name for the predict method.
            # No actual model object is loaded into memory in this script.
            self.model = model_id
            print(f"Ollama model '{self.model}' is available and ready.")

        except Exception as e:
            raise IOError(
                f"Failed to connect to Ollama or find model '{model_id}'. "
                f"Please ensure the Ollama service is running. Error: {e}"
            )

    def _image_to_base64(self, image_path: str) -> str:
        """Converts an image file to a base64 encoded string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at path: {image_path}")
        except Exception as e:
            raise IOError(f"Error reading or encoding the image file at {image_path}: {e}")

    def predict(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        """
        Runs inference on a single image and prompt using the Ollama chat API.
        """
        if not self.model:
            raise RuntimeError("Model is not loaded. Please call load_model() first.")

        try:
            # 1. Convert the image to a base64 string
            image_b64 = self._image_to_base64(image_path)

            # 2. Send the request to the Ollama API
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_b64]  # The API expects a list of base64 strings
                }],
                options={
                    'num_predict': max_new_tokens
                }
            )

            # 3. Extract the text content from the response message
            return response['message']['content']
        except Exception as e:
            raise RuntimeError(f"Inference failed for image {image_path} with Ollama model {self.model}: {e}")

# ==============================================================================
# UPDATED FACTORY FUNCTION
# ==============================================================================
def get_model_adapter(model_config: dict) -> VLMAdapter:
    """Factory to select the correct adapter based on the model's configuration."""
    adapter_type = model_config.get("adapter_type")
    if adapter_type == "standard_pipeline":
        return StandardPipelineAdapter(model_config)
    elif adapter_type == "qwen_vl":
        return QwenVLAdapter(model_config)
    elif adapter_type == "ollama":
        return OllamaAdapter(model_config)
    else:
        raise ValueError(f"Unknown or unspecified adapter type: '{adapter_type}'")
