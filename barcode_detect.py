import cv2
import numpy as np

def scale_image(img):
  '''Returns image scaled to proper dpi for Tesseract

  Given image of UPS shipping label, returns an image that is scaled down to
  reduce processing time for image processing operations, optimized for 
  Tesseract 300 dpi minimum resolution. 

  Args:
    img: image to analyze, represented as numpy array from cv2.imread()

  Returns:
    Smaller version of image scaled based on width
  '''

  target_dpi = 300
  ups_default_label_width = 4.01  #inches
  ups_default_barcode_width = 2.79  #inches
  img_res_x = img.shape[1]
  img_res_y = img.shape[0]

  # Scale image based on target dpi and the assumption that the full image 
  # width is pretty close to the label width
  target_res_x = target_dpi * ups_default_label_width
  scale = target_res_x / img_res_x
  img_scaled = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)

  return img_scaled

def find_vertical_edges(img):
  '''Returns an image containing highlighted vertical edges

  Performs edge finding and returns a cleaned result.

  Args:
    img: image to find edges, represented as numpy array from cv2.imread(), 
         assumed to be grayscale image for edge finding operations

  Returns:
    Image with vertical edges highlighted
  '''

  # Edge finding derivative only in x direction because we made the assumption
  # that the barcode is predominantly vertically aligned. Blur and threshold 
  # to remove speckles and focus on dominant edges.

  img_temp = cv2.Sobel(img, ddepth=-1, dx=1, dy=0)
  img_temp = cv2.blur(img_temp, ksize=(3,3))
  _, img_temp = cv2.threshold(img_temp, thresh=0, maxval=255,
                              type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  return img_temp

def isolate_barcodes(img):
  '''Returns image filtered of all but the largest of barcode-like items

  Identifies barcode-like images and attempts to filter out everything but 
  these items. Resulting image can have false positives, so should be further
  filtered to get only the items of interest.

  Args:
    img: image to analyze, represented by numpy array from cv2.imread()

  Returns:
    Image with all non-barcode-like images filtered out
  '''

  # Two operations to perform: 1) Closing operation to collapse barcode
  #                               into blocks
  #                            2) Opening operation to filter out 
  #                               everything except the largest items

  img_size_x = img.shape[1]
  img_size_y = img.shape[0]

  # Closing operation uses long, narrow rectangle because barcode should span 
  # significant amount of width, but doesn't need a lot of vertical distance
  close_struct = (80, 1)
  close_elem = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize=close_struct)
  img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, close_elem)

  # Opening operation uses giant rectangle because barcode should be the 
  # biggest of items.
  open_struct = (200, 100)
  open_elem = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=open_struct)
  img_open = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, open_elem)

  return img_open

def id_and_rank_contours(img):

  # Returns contours in largest to smallest order by area
  _, cont, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(cont, key=cv2.contourArea, reverse=True)

  return contours

def get_barcode_contours(img):
  '''Returns list of barcode contours in descending order by area

  Given an image of a UPS label, detect all of the barcodes and return their 
  contours in a list of descending order. Same result as if calling 
  cv.findContours().

  Args:
    img: image to analyze, represented as numpy array from cv2.imread()

  Returns:
    List of barcode contours
  '''

  # Assumptions for barcode detection:
  # - The image represents a UPS shipping label
  # - Image has barcodes that are predominantly vertical (slight skew is fine)
  # - Image has at least 1 barcode
  # - Shipping label width is similar to image width (75% - 100% of image width)

  # UPS default label width = 4.01", barcode width = 2.79"

  # img_temp = scale_image(img)
  img_temp = img
  img_temp = cv2.cvtColor(img_temp, cv2.COLOR_RGB2GRAY)
  img_temp = find_vertical_edges(img_temp)
  img_temp = isolate_barcodes(img_temp)
  contours = id_and_rank_contours(img_temp)

  return contours

def draw_all_contours(img, contours):
  '''Draws all of the contours on the image

  Contours found with get_barcode_contours() are drawn on the image itself
  and returned. Biggest contour is red, rest are green.

  Args:
    img: image to analyze, represented as numpy array from cv2.imread()
    contours: contours to draw, result from get_barcode_contours(). 
  '''
  for idx,contour in enumerate(contours):
    if idx == 0:
      color = (0, 0, 255)
    else:
      color = (0, 255, 0)
    rect = cv2.minAreaRect(contour)
    box = np.int0(cv2.boxPoints(rect))
    img = cv2.drawContours(img, [box], 0, color, 2)
