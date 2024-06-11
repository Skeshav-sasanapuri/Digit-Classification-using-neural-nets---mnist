import numpy as np
class Student:

    def __init__(self, firstname, scores):
        self.name = firstname
        self.grades = scores

samarth = Student('Samarth', (72, 100, 100, 100, 100, 100, 100))
chunnu = Student("Chunnu", [100, 100, 100, 100, 100, 100, 100])

grade1 = samarth.grades[0]
print(grade1)

