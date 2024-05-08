from paddleocr import PaddleOCR, draw_ocr

def perform_ocr(img_path, language='en'):
    """
    Perform Optical Character Recognition (OCR) on an image.
    
    Args:
    - img_path (str): Path to the image file.
    - language (str): Language setting for OCR. Options: 'ch', 'en', 'french', 'german', 'korean', 'japan'.
    
    Returns:
    - list of lists: OCR results, where each inner list represents a detected text line.
    """
    # Instantiate OCR model
    ocr = PaddleOCR(use_angle_cls=True, lang=language)
    
    # Perform OCR on the image
    result = ocr.ocr(img_path, cls=True)
    
    return result

def print_ocr_results(ocr_results):
    """
    Print the OCR results.
    
    Args:
    - ocr_results (list of lists): OCR results returned by the perform_ocr function.
    """
    for idx in range(len(ocr_results)):
        res = ocr_results[idx]
        for line in res:
            print(line)

def main():
    # Path to the image file
    img_path = "path/to/your/image.jpg"
    
    # Perform OCR
    ocr_results = perform_ocr(img_path, language='en')
    
    # Print OCR results
    print_ocr_results(ocr_results)

if __name__ == "__main__":
    main()
