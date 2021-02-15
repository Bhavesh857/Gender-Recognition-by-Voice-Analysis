from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import sys
import subprocess
import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors
import matplotlib.pyplot as plt
from tkinter import ttk
import pyaudio
import wave
import time


def OpenFile():
	file1= open("file1","w+")
	name = askopenfilename(initialdir="C:/Users/Batman/Documents/Programming/tkinter/",
                           filetypes =(("Text File", "*.txt"),("All Files","*.*")),
                           title = "Choose a file."
                           )
	print (name)
	file1.write(name)
	file1.close()


def predictor():
	subprocess.call ("./try.R")
	file2= open("out","r")
	buf1=file2.read()
	file2.close()
	print(buf1)
	a=[float(x) for x in buf1.split()]
	#print(a)

	start_time=time.time()
	df = pd.read_csv('voice.csv')

	df = df.rename(columns={'label': 'gender'})


	#Lets use SVM:
	#Bootstrapping

	df1=df[['meanfun','IQR','Q25','sp.ent','sd','sfm','meanfreq','gender']]
	#Producing X and y
	X = df1.drop(['gender'], 1)
	y = df1['gender']

	from sklearn.preprocessing import LabelEncoder
	labelencoder1 = LabelEncoder()
	y = labelencoder1.fit_transform(y)

	#Dividing the data randomly into training and test set
	from sklearn.model_selection import ShuffleSplit
	rs =ShuffleSplit(n_splits=30,train_size=.8,test_size=.2, random_state=0)

	for train_index, test_index in rs.split(X):
        	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        	y_train, y_test = y[train_index], y[test_index]

	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)


	from sklearn.svm import SVC
	from sklearn import svm
	classifier = SVC(kernel = 'linear', random_state=10, gamma='auto')
	classifier.fit(X_train, y_train)
	print("--%s seconds--" %(time.time()-start_time))
	a=np.array(a).reshape(1, -1)
	a = sc.transform(a)

	y_pred=classifier.predict(a)
	y_score = classifier.fit(X_train, y_train).decision_function(X_test)	

	
	if y_pred[0]==0:
		messagebox.showinfo("Prediction of Voice Sample","Prediction of Voice Sample is Female")
	else:
		messagebox.showinfo("Prediction of Voice Sample","Prediction of Voice Sample is Male")

              

def record():
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100
	CHUNK = 1024
	RECORD_SECONDS = 5
	WAVE_OUTPUT_FILENAME = "file.wav"
	file1= open("file1","w+")
	file1.write("/home/pooja/Documents/ml/file.wav")
	file1.close()
	audio = pyaudio.PyAudio()
 
# start Recording
	stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
	print ("recording...")
	frames = []
 
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)
	print ("finished recording")
 
 
# stop Recording
	stream.stop_stream()
	stream.close()
	audio.terminate()
 
	waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	waveFile.setnchannels(CHANNELS)
	waveFile.setsampwidth(audio.get_sample_size(FORMAT))
	waveFile.setframerate(RATE)
	waveFile.writeframes(b''.join(frames))
	waveFile.close()



top = Tk()
w= top.winfo_screenwidth()
h = top.winfo_screenheight()
top.geometry("600x400")
top.title("Gender Voice Recognition")
#top.geometry("400x600")
C = Canvas(top, height=400, width=600)
filename = PhotoImage(file = "sound22.png")
#C.create_image(10,10,image=filename)
background_label = Label(top, image=filename,width=w,height=h)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


b=Button(top, text ="Record", bd="10",bg="cyan2",relief=RAISED ,command=record)
c=Button(top, text ="Browse", bd="10",bg="cyan2",relief=RAISED ,command=lambda : OpenFile())
d=Button(top, text ="Predict", bd="10",bg="cyan2",relief=RAISED ,command=predictor)
photo1=PhotoImage(file="tub.png")
b.config(image=photo1,width="100",height="100")
b.pack(side=LEFT,padx=40, pady=30,anchor=NW)
photo2=PhotoImage(file="browse.png")
c.config(image=photo2,width="100",height="100")
c.pack(side=LEFT,padx=40, pady=30,anchor=NW)
photo3=PhotoImage(file="gender.png")
d.config(image=photo3,width="100",height="100")
d.pack(padx=40, pady=30,anchor=NW)


top.mainloop()




