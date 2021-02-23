



def detect_face(img):
    faces = []
    classifier = CascadeClassifier(r'C:\Users\Ayse\Desktop\ayse_proje\Face Detection\haarcascade_frontalface_default.xml')
    # perform face detection
    pixels = img
    bboxes = classifier.detectMultiScale(pixels)
    # print bounding box for each detected face
    for box in bboxes:
        # extract
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        # rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
        faces.append((x, y, x2, y2))
    # show the image
    return faces


