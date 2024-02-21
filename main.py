import numpy as np
import face_recognition
import cv2
import os, sys
import math

def main():
    fr = FaceRecognition()
    fr.run_recognition()

class FaceRecognition:
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLUE = (255, 0, 0)
    FONT = cv2.FONT_HERSHEY_DUPLEX
    PATH_FACES = 'known_faces/'
    faces_directory = os.listdir(PATH_FACES)

    face_locations = list()
    face_encodings = list()
    face_names = list()
    known_face_encodings = list()
    known_face_names = list()
    process_current_frame = True

    __instance = None
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __del__(self):
        FaceRecognition.__instance = None

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for face in self.faces_directory:
            current_name = os.path.splitext(face)[0]
            face_image = face_recognition.load_image_file(self.PATH_FACES + face)
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_names.append(current_name)
            self.known_face_encodings.append(face_encoding)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            success, frame = video_capture.read()

            if self.process_current_frame:
                frame_small_rgb = self.processing_format(frame)

                self.face_locations = face_recognition.face_locations(frame_small_rgb)
                self.face_encodings = face_recognition.face_encodings(frame_small_rgb, self.face_locations)

                self.face_names = list()
                for face_encoding in self.face_encodings:
                    name = 'Unknown'
                    confidence = 'None'
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = self.face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), self.COLOR_BLUE, 2)
                cv2.rectangle(frame, (left, bottom), (right, bottom + 35), self.COLOR_BLUE, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom + 25), self.FONT, 0.8, self.COLOR_WHITE, 1)

            cv2.imshow('WebCamera', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    @classmethod
    def processing_format(cls, image):
        image_small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        # frame_small_rgb = frame_small[:, :, ::-1]
        return cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

    @classmethod
    def face_confidence(cls, face_distance, face_match_threshold=0.6):
        range = (1.0 - face_match_threshold)
        linear_value = (1.0 - face_distance) / (range * 2.0)

        if face_distance > face_match_threshold:
            return str(round(linear_value * 100, 2)) + '%'
        else:
            value = (linear_value + ((1.0 - linear_value) * math.pow((linear_value - 0.5) * 2, 0.2))) * 100
            return str(round(value, 2)) + '%'

if __name__ == '__main__':
    main()
