"""
Pre-downloads the Hugging Face ViT emotion model weights.
This is run during Docker build to bake the weights into the container image,
ensuring offline execution and instantaneous startup without runtime downloading.
"""
import sys
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "trpakov/vit-face-expression"

def main():
    print(f"Downloading {MODEL_ID} to Hugging Face cache...")
    try:
        AutoImageProcessor.from_pretrained(MODEL_ID)
        AutoModelForImageClassification.from_pretrained(MODEL_ID)
        print("Model pre-download completed successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
