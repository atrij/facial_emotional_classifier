# script to load image data
import os
class data:

    def __init__(self, label_path,image_path):
        self.label_path = label_path
        self.image_path = image_path
        self.label_data =[]
    def prin(self):
    	print self.label_data
       #	subject=os.listdir(self.label_data)

