import argparse
from paddleocr import PaddleOCR

class OCRProcessor:
    def __init__(self, language='en'):
        """
        Initialize the OCR processor.

        Args:
        - language (str): Language setting for OCR. Options: 'ch', 'en', 'french', 'german', 'korean', 'japan'.
        """
        self.language = language
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.language)

    def perform_ocr(self, img_path):
        """
        Perform Optical Character Recognition (OCR) on an image.

        Args:
        - img_path (str): Path to the image file.

        Returns:
        - list of lists: OCR results, where each inner list represents a detected text line.
        """
        result = self.ocr.ocr(img_path, cls=True)
        return result

    def print_ocr_results(self, ocr_results):
        """
        Print the OCR results.

        Args:
        - ocr_results (list of lists): OCR results returned by the perform_ocr method.
        """
        for idx in range(len(ocr_results)):
            res = ocr_results[idx]
            for line in res:
                print(line)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform OCR on an image")
    parser.add_argument("img_path", type=str, help="Path to the image file")
    parser.add_argument("--language", type=str, default="en", help="Language for OCR (default: 'en')")
    args = parser.parse_args()

    # Instantiate OCR Processor
    ocr_processor = OCRProcessor(language=args.language)
    
    # Perform OCR
    ocr_results = ocr_processor.perform_ocr(args.img_path)
    
    # Print OCR results
    ocr_processor.print_ocr_results(ocr_results)
