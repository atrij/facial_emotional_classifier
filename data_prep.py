# script to load image data
import os
class data:

    def __init__(self, label_path,image_path):
        self.label_path = label_path
        self.image_path = image_path
        self.label_data =[]
    def data_import(self):
		subject=os.listdir(self.label_path)
		for i in range(0,len(subject)):
			path_subject=self.label_path+'/'+subject[i]
			subject_sequence= os.listdir(path_subject)
			for j in range(0,len(subject_sequence)):
				path_sequence=path_subject+'/'+subject_sequence[j]
				emotion=os.listdir(path_sequence)
				if len(emotion)==0:
					continue
				f = open(path_sequence+'/'+emotion[0], 'r')
				emo_tag=f.readlines()
				emo=int(float(emo_tag[0]))
				image_name = emotion[0][0:-12]
				image_path = self.image_path+'/'+subject[i]+'/'+subject_sequence[j]+'/'+image_name+'.png'
				image_tuple=(image_path,emo)
				self.label_data.append(image_tuple)


