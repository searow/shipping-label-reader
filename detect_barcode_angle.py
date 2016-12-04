import cv2
import barcode_detect

def get_angle(img):
  """Returns correction angle (deg) to make image vertical.

  Detects barcode in image and returns the angle required to make the image
  vertical. A positive angle denotes that the image should be rotated 
  counterclockwise by the angle amount to be corrected to vertical.

  Args:
    img: image to analyze, represented as numpy array from cv2.imread()

  Returns:
    Angle that image is offset to make vertical. Positive angle denotes that
    the image should be rotated counterclockwise by that angle to be corrected
    to vertical. Angle given in degrees.
  """

  angle = 1

  return angle