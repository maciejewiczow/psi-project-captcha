import cv2

def getRegionsFromImage(image, margin = 5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.medianBlur(gray, 1)
    # blur = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
    tr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    contours = [cv2.boundingRect(contour) for contour in cv2.findContours(tr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]]
    contours = [(x, y, w, h) for (x, y, w, h) in contours if w > 2 and h > 5 and w/h < 3 and h/w < 8]
    contours = [(max(x - margin, 0), max(y - margin, 0), w + 2*margin, h + 2*margin) for (x, y, w, h) in contours]

    regions = sorted(contours, key=lambda x: x[0])

    return [gray[y:y + height, x:x + width] for (x, y, width, height) in regions]
