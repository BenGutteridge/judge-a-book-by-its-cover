"""
Functions to perform OCR on images using various libraries and APIs.
(Several are not used in the paper.)
Used in 00_iam_preprocessing.py
"""

from pathlib import Path
import pytesseract
import io
from google.cloud import vision
from google.cloud.vision_v1 import types
import requests
import os
import time
import boto3
from typing import Optional, TypedDict
from PIL import Image
from loguru import logger


def tesseract_img2txt(image_path: Path, lang: str = "eng") -> tuple[str, None]:
    """
    Extracts text from an image using Tesseract OCR.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        str: The text extracted from the image.
    """
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang=lang), None  # no bboxes


from google.cloud.vision_v1.types import EntityAnnotation


# Can be anything
class BBox(dict):
    pass


def googleocr_img2txt(image_path: Path) -> tuple[str, list[BBox]]:
    """
    Detects handwritten text in the specified image file using Google Cloud Vision API.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Extracted handwritten text from the image.
        list[BBox]: List of bounding boxes for each detected text annotation.
    """
    # Initialize the Google Vision client
    client = vision.ImageAnnotatorClient()

    # Read the image file into memory
    with io.open(image_path, "rb") as image_file:
        content = image_file.read()

    # Create an image object for the Vision API
    image = types.Image(content=content)

    # Perform text detection
    response = client.text_detection(image=image)

    # Extract the detected text and bounding boxes
    annotations = response.text_annotations
    text = annotations[0].description
    bboxes = [
        {
            "text": annotation.description,
            "vertices": [
                {"x": vertex.x, "y": vertex.y}
                for vertex in annotation.bounding_poly.vertices
            ],
        }
        for annotation in annotations
    ]
    return text, bboxes


def ocr_space_img2txt(
    image_path: Path,
    api_key=os.getenv("OCR_SPACE_API_KEY"),
    language: str = "eng",
) -> str:
    """
    This function sends an image to the OCR.space API and retrieves the recognized text.

    Parameters:
    image_path (str): Path to the image file to be processed.
    api_key (str): Your OCR.space API key.
    language (str): The language code to use for OCR (default is 'eng' for English).

    Returns:
    dict: A dictionary containing the OCR result from the API.
    """

    # Set up API endpoint and payload
    url = "https://api.ocr.space/parse/image"

    # Set up the payload for the POST request
    payload = {
        "apikey": api_key,
        "language": language,
        "isOverlayRequired": False,  # Can set this to True to get word position data
    }

    # Open the image file and send the request
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(url, data=payload, files=files)

    # Return the response in JSON format
    res = response.json()
    if res["IsErroredOnProcessing"]:
        error_msg = res["ErrorMessage"]
        logger.error(f"OCR failed:\n{error_msg}")
        return ""
    return res["ParsedResults"][0]["ParsedText"]


def azure_img2txt(
    image_path: Path,
    endpoint: str = os.getenv("AZURE_ENDPOINT"),
    api_key: str = os.getenv("AZURE_API_KEY"),
    pause: int = 5,
) -> tuple[Optional[str], Optional[list[BBox]]]:
    """
    Extracts text from an image or PDF file using Azure's Read API.

    Parameters:
    - image_path (Path): Path to the image or PDF file.
    - endpoint (str): Azure Cognitive Services endpoint.
    - api_key (str): API key for authentication.
    - pause (int): Time to wait in seconds before making the request, to get around throttling.

    Returns:
    - Optional[str]: The extracted text if successful, None otherwise.
    """
    # Set the Read API URL
    read_url = f"{endpoint}/vision/v3.2/read/analyze"

    # Set headers and parameters for the request
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/octet-stream",
    }

    # Apply pause
    time.sleep(pause)

    # Open the image/PDF file in binary mode and send the request
    with open(image_path, "rb") as image_data:
        response = requests.post(read_url, headers=headers, data=image_data)

    # Check if the initial request was successful
    if response.status_code != 202:
        print(f"Error {response.status_code}: {response.json()}")
        return None

    # Get the URL to check the read results
    operation_url = response.headers["Operation-Location"]

    # Poll for the results
    while True:
        result_response = requests.get(
            operation_url, headers={"Ocp-Apim-Subscription-Key": api_key}
        )
        result = result_response.json()

        # Check the status
        status = result.get("status")
        if status == "succeeded":
            break
        elif status == "failed":
            print("Failed to extract text.")
            return None
        else:
            # Wait a moment before polling again
            time.sleep(1)

    # Extract text and bounding boxes from the result
    text = ""
    bboxes = []
    for page in result.get("analyzeResult", {}).get("readResults", []):
        for line in page.get("lines", []):
            text += line.get("text", "") + "\n"
        bboxes.append(page)

    return text, bboxes


def textract_img2txt(image_path: Path) -> str:
    """
    Detects text (including handwritten text) in the specified image file using AWS Textract.

    Args:
        image_path (Path): Path to the image or PDF file.

    Returns:
        str: Extracted text from the image or PDF.
    """
    # Initialize the Textract client
    client = boto3.client("textract")

    # Read the image file into memory
    image_path = Path(image_path)
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    # Determine if the input is an image or PDF
    if image_path.suffix.lower() in [".pdf"]:
        response = client.analyze_document(
            Document={"Bytes": content}, FeatureTypes=["TABLES", "FORMS"]
        )
    else:
        response = client.detect_document_text(Document={"Bytes": content})

    # Extract the detected text
    extracted_text = []
    if "Blocks" in response:
        for block in response["Blocks"]:
            if block["BlockType"] == "LINE":
                extracted_text.append(block["Text"])

    return "\n".join(extracted_text)
