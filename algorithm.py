import cv2
import os


class FaceDetection:
    def __init__(self, mode):
        self.mode = mode
        self.casc_path = os.path.dirname(cv2.__file__) + '/data/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.casc_path)

        if self.mode == 1:
            self.image_detection()

        if self.mode == 2:
            self.video_detection()

    def image_detection(self):
        # Scan Image and convert to grayscale
        img = cv2.imread('test.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect face
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw rectangle around the face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Output
        resize = cv2.resize(img, (1000, 1000))
        cv2.imshow('img', resize)
        cv2.waitKey()

    def video_detection(self):
        # Define video object
        video = cv2.VideoCapture(0)

        # Capture video to image frame by frame
        while True:
            ret, img = video.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect face
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Draw rectangle around the face
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Output
            cv2.imshow('Video', img)

            # Press q for break the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()