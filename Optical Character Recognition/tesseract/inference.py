
import pytesseract
import cv2


def preprocess_image(img):
  """Preprocesses the image for improved OCR accuracy."""
  # Grayscale conversion
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Binarization (thresholding)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

  # Noise reduction (optional)
  # kernel = np.ones((3, 3), np.uint8)
  # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

  # Deskewing (optional)
  # coords = pytesseract.image_to_config(thresh)["text_angle"].split()[1]
  # angle = float(coords)
  # if angle > 0:
  #   rotated = cv2.rotate(thresh, cv2.ROTATE_90_COUNTERCLOCKWISE)
  # else:
  #   rotated = cv2.rotate(thresh, cv2.ROTATE_90_CLOCKWISE)

  return thresh


def extract_text(img_path, lang='eng'):
  """Extracts text from an image using Tesseract with preprocessing."""
  # Load image
  img = cv2.imread(img_path)

  # Preprocess image
  preprocessed_img = preprocess_image(img)

  # Perform OCR with Tesseract
  text = pytesseract.image_to_string(preprocessed_img, config=f'--oem 1 --psm 6 {lang}')

  return text


# Example usage
image_path = 'complex_image.jpg'
extracted_text = extract_text(image_path)

print(f"Extracted Text: \n{extracted_text}")