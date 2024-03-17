import json
import numpy as np
import face_recognition
import cv2
import os
import sys
from datetime import datetime


class GodEye:
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_YELLOW = (22, 171, 240)
    FONT = cv2.FONT_HERSHEY_DUPLEX
    PATH_FACES = 'known_faces/'

    FACE_RECTANGLE_THICKNESS = 1
    BOX_MARGIN = 5

    TEXT_SMALL_SCALE = 0.3
    TEXT_REGULAR_THICKNESS = 1
    TEXT_SMALL_PADDING = 5

    TEXT_LARGE_SCALE = 0.9
    TEXT_BOLD_THICKNESS = 2
    TEXT_LARGE_PADDING = BOX_MARGIN + TEXT_SMALL_PADDING

    TRIANGLE_HEIGHT = 30

    known_face_encodings = list()

    people = dict()

    process_current_frame = True

    MIN_FACE_DISTANCE = 0.5

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __del__(self):
        GodEye.__instance = None

    def __init__(self):
        self.read_people_information()
        self.encode_known_people()
        self.run_recognition()
        self.write_people_information()

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        for person in self.people:
            self.known_face_encodings.append(person['face_encoding'])

        while True:
            success, frame = video_capture.read()

            if self.process_current_frame:
                face_locations = list()
                people_on_frame = list()
                processing_frame = self.processing_format(frame)

                face_locations = face_recognition.face_locations(processing_frame)
                face_encodings = face_recognition.face_encodings(processing_frame, face_locations)

                for face_encoding in face_encodings:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    min_face_distance_index = np.argmin(face_distances)

                    # if face_distances[min_face_distance_index] < self.MIN_FACE_DISTANCE:
                    person = self.people[min_face_distance_index]
                    people_on_frame.append(person)

            self.process_current_frame = not self.process_current_frame

            self.draw_hud(frame, face_locations, people_on_frame)

            cv2.imshow('WebCamera', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def encode_known_people(self):
        faces_directory = os.listdir(self.PATH_FACES)

        first_name = None
        second_name = None
        date_of_birth = None
        nationality = None
        occupation = None

        for face in faces_directory:
            face_image = face_recognition.load_image_file(self.PATH_FACES + face)
            face_encoding = face_recognition.face_encodings(face_image)[0]
            face_encoding_list = face_encoding.tolist()

            if face_encoding_list not in self.known_face_encodings:
                first_name = os.path.splitext(face)[0]

                person = {
                    'face_encoding': face_encoding_list,
                    'first_name': first_name,
                    'second_name': second_name,
                    'nationality': nationality,
                    'additional_information': {
                        'date_of_birth': date_of_birth,
                        'occupation': occupation
                    }
                }
                self.people.append(person)

    def read_people_information(self):
        with open('people.json', 'r', encoding='utf8') as people_file:
            self.people = json.load(people_file)

    def write_people_information(self):
        with open('people.json', 'w', encoding='utf8') as people_file:
            json.dump(self.people, people_file, indent=4)

    @classmethod
    def processing_format(cls, image):
        image_small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        return cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

    @classmethod
    def calculate_age(cls, date_of_birth):
        birth_date = datetime.strptime(date_of_birth, '%d.%m.%Y')
        today = datetime.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return str(age)

    @classmethod
    def draw_hud(cls, image, face_locations, people_information):

        for (face_top, face_right, face_bottom, face_left), person in zip(face_locations, people_information):
            face_top *= 4
            face_right *= 4
            face_bottom *= 4
            face_left *= 4

            # Face rectangle
            cv2.rectangle(image,
                          (face_left, face_top),
                          (face_right, face_bottom),
                          cls.COLOR_WHITE,
                          cls.FACE_RECTANGLE_THICKNESS)

            # Name fild
            name = person['first_name']

            name_text_size, _ = cv2.getTextSize(name, cls.FONT, cls.TEXT_LARGE_SCALE, cls.TEXT_BOLD_THICKNESS)
            name_text_width, name_text_height = name_text_size

            name_box_top = face_top + int((face_bottom - face_top) * 0.55)
            vertex_1 = [face_right, name_box_top]
            vertex_2 = [face_right + 2 * cls.TEXT_LARGE_PADDING + name_text_width, name_box_top]
            vertex_3 = [face_right + 2 * cls.TEXT_LARGE_PADDING + name_text_width,
                        name_box_top + 2 * cls.TEXT_LARGE_PADDING + name_text_height]
            vertex_4 = [face_right, name_box_top + 2 * cls.TEXT_LARGE_PADDING + name_text_height]
            vertex_5 = [face_right - cls.TRIANGLE_HEIGHT,
                        (2 * (name_box_top + cls.TEXT_LARGE_PADDING) + name_text_height) // 2]
            name_shape_vertices = np.array([vertex_1, vertex_2, vertex_3, vertex_4, vertex_5], np.int32)

            cv2.fillPoly(image, [name_shape_vertices], cls.COLOR_WHITE)
            cv2.putText(image,
                        name,
                        (face_right + cls.TEXT_LARGE_PADDING, name_box_top + name_text_height + cls.TEXT_LARGE_PADDING),
                        cls.FONT, cls.TEXT_LARGE_SCALE,
                        cls.COLOR_BLACK, cls.TEXT_BOLD_THICKNESS)

            # Nationality
            nationality = person['nationality']

            if nationality is not None:
                nationality = '[ ' + person['nationality'] + ' ]'

                nationality_text_size, _ = cv2.getTextSize(nationality,
                                                           cls.FONT,
                                                           cls.TEXT_SMALL_SCALE,
                                                           cls.TEXT_REGULAR_THICKNESS)
                nationality_text_width, nationality_text_height = nationality_text_size

                nationality_box_start = [face_right, name_box_top]
                nationality_box_end = [vertex_3[0],
                                       name_box_top - 2 * cls.TEXT_SMALL_PADDING - nationality_text_height]

                cv2.rectangle(image,
                              nationality_box_start,
                              nationality_box_end,
                              cls.COLOR_YELLOW,
                              cv2.FILLED)

                cv2.putText(image,
                            nationality,
                            (face_right + cls.BOX_MARGIN + cls.TEXT_SMALL_PADDING,
                             name_box_top - cls.TEXT_SMALL_PADDING),
                            cls.FONT,
                            cls.TEXT_SMALL_SCALE,
                            cls.COLOR_BLACK,
                            cls.TEXT_REGULAR_THICKNESS)

            # Additional Information
            additional_information = person['additional_information']
            additional_information_keys = additional_information.keys()

            birthdate_check = True
            indents_number = 0
            for key in additional_information_keys:
                value = additional_information[key]

                if value is not None:
                    if birthdate_check and key == 'date_of_birth':
                        value = cls.calculate_age(additional_information[key])
                        key = 'age'
                        birthdate_check = False

                    information_text = key.capitalize() + ' / ' + value.capitalize()

                    information_text_size, _ = cv2.getTextSize(information_text,
                                                               cls.FONT,
                                                               cls.TEXT_SMALL_SCALE,
                                                               cls.TEXT_REGULAR_THICKNESS)
                    information_text_width, information_text_height = information_text_size

                    information_box_height = 2 * cls.TEXT_SMALL_PADDING + information_text_height

                    information_box_start = [face_right + cls.BOX_MARGIN,
                                             vertex_4[1] + cls.BOX_MARGIN + indents_number * (
                                                         cls.BOX_MARGIN + information_box_height)]
                    information_box_end = [
                        information_box_start[0] + 2 * cls.TEXT_SMALL_PADDING + information_text_width,
                        information_box_start[1] + information_box_height]

                    cv2.rectangle(image,
                                  information_box_start,
                                  information_box_end,
                                  cls.COLOR_BLACK, cv2.FILLED)

                    information_text_start = [face_right + cls.BOX_MARGIN + cls.TEXT_SMALL_PADDING,
                                              information_box_start[
                                                  1] + cls.TEXT_SMALL_PADDING + information_text_height]

                    cv2.putText(image,
                                information_text,
                                information_text_start,
                                cls.FONT,
                                cls.TEXT_SMALL_SCALE,
                                cls.COLOR_WHITE,
                                cls.TEXT_REGULAR_THICKNESS)

                    indents_number += 1
