import cv2
import barcode_detect

def get_angle(img):
  ''''Returns correction angle (deg) to make image vertical.

  Detects barcode in image and returns the angle required to make the barcode
  vertical. A positive angle denotes that the image should be rotated 
  counterclockwise by the angle amount to be corrected to vertical.

  Args:
    img: image to analyze, represented as numpy array from cv2.imread()

  Returns:
    Angle that image is offset to make vertical. Positive angle denotes that
    the image should be rotated counterclockwise by that angle to be corrected
    to vertical. Angle given in degrees.
  '''

  contours = barcode_detect.get_barcode_contours(img)
  barcode_detect.draw_all_contours(img, contours)

  # Use only the largest contour to determine rotation angle
  rect = cv2.minAreaRect(contours[0])
  angle = rect[2]

  # Angle = [-90, 0). We want positive degrees = counterclockwise rotation
  if angle < -45:
    angle += 90

  return angle
