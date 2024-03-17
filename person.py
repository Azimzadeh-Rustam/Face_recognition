class Person:

    def __init__(self, id, face_encoding, first_name, second_name, date_of_birth):
        self._id = id
        self.face_encoding = face_encoding
        self.first_name = first_name
        self.second_name = second_name
        self.date_of_birth = date_of_birth

    @property
    def face_encoding(self):
        return self._face_encoding

    @face_encoding.setter
    def face_encoding(self, face_encoding):
        self._face_encoding = face_encoding

    @face_encoding.deleter
    def face_encoding(self):
        del self._face_encoding

    @property
    def first_name(self):
        return self._first_name

    @first_name.setter
    def first_name(self, first_name):
        self._first_name = first_name

    @first_name.deleter
    def first_name(self):
        del self._first_name

    @property
    def second_name(self):
        return self._second_name

    @second_name.setter
    def second_name(self, second_name):
        self._second_name = second_name

    @second_name.deleter
    def second_name(self):
        del self._second_name

    @property
    def date_of_birth(self):
        return self._date_of_birth

    @date_of_birth.setter
    def date_of_birth(self, date_of_birth):
        self._date_of_birth = date_of_birth

    @date_of_birth.deleter
    def date_of_birth(self):
        del self._date_of_birth
