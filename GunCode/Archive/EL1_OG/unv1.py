import sys
import os
packages_path=os.getcwd()+"/lib/python3.9/site-packages"
print(packages_path)
print(os.path.isdir(packages_path))
#print(os.path.isdir("/home/pi/programs/p4/lib/python3.9/site-packages"))
#sys.path.append("/home/pi/programs/p4/lib/python3.9/site-packages")
sys.path.append("/usr/lib/python3/dist-packages")

import socket
import select
import cv2
import time
import multiprocessing
from multiprocessing import Process, Value
from multiprocessing.sharedctypes import (
	Synchronized,
)
import numpy as np
import serial
import pygame
from pygame import mixer
import subprocess as sp

def sound_handler(channel_num, volume,sound_effect):
	global sound_names
	global t_switch_on
	if channel_num>2: #channels 0-2
		channel_num=0
	else:
		channel_num+=1

	if sound_effect==sound_names["reload"]:
		channel_num=2
	if (sound_effect==sound_names["kill"]) or (sound_effect==sound_names["dead"]):
		channel_num=3

	mixer.Channel(channel_num).set_volume(volume)
	mixer.Channel(channel_num).play(mixer.Sound(sound_effect))
	print(channel_num, sound_effect,time.time()-t_switch_on)
	return(channel_num)


def serial_process(serial_arr: Synchronized) :#serial function
		arduino_file=open("arduino_id.txt",'r')
		arduino_id=arduino_file.readline().split('\n')[0]
		serial_obj=serial.Serial(arduino_id,115200)

		print("starting serial")
		prev_av=0
		changed_from=0
		prev_state=False

		#serial arr[0] is for trigger serial arr[1] is for health serial arr[2] is for reload
		while True:
			s_bytes=serial_obj.readline()
			if (serial_arr[1]!=prev_av):
				changed_from=prev_av
				prev_av=serial_arr[1]
				print("health change:" ,serial_arr[1],prev_av)
			elif(changed_from!=serial_arr[1]):
				encoded_v=str(serial_arr[1]).encode("utf-8")
				serial_obj.write(encoded_v)
			if len(s_bytes)>2:
				if s_bytes[3]==49: #trigger logic
					serial_arr[0]=1
					if not prev_state:
						print("SHOOTING-----")
					prev_state=True
				else:
					serial_arr[0]=0
					prev_state=False

				if s_bytes[2]==49: #reload logic
					print("RELOADING")
					serial_arr[2]=1
				else:
					serial_arr[2]=0


def trigger_handler(switch_state,firing_mode, fire_rate):
	a=time.time()
	global prev_switch_state
	global shot_counter
	global last_shot_time
	return_v=False
	if (1/fire_rate)<=(time.time()-last_shot_time):#fire rate handler
		if firing_mode=='a':#full auto
			if switch_state:
				return_v=True
			else:
				return_v=False
		elif firing_mode=='s':#single fire
			if not prev_switch_state and switch_state:
				return_v=True
			else:
				return_v=False
		elif firing_mode=='b':#burst
			if (not prev_switch_state and switch_state):
				shot_counter=0
			if shot_counter<3:
				shot_counter+=1
				return_v=True
			else:
				return_v=False
		return(return_v)
	return(False) #if nothing return false



def update_crop(camera_index,zoom,x,y):#handles zoom and gets offset from file
	ts=time.time()
	offsetX=0
	offsetY=0
	with open ("campos.txt",'r') as posfile:
		data=posfile.readline().split()
		data+=posfile.readline().split()

		if camera_index==0:
			offsetX=int(data[0])
			offsetY=int(data[1])
		else:
			offsetX=int(data[2])
			offsetY=int(data[3])
	X_MIN=int(x*(1-1/zoom)/2)+offsetX
	Y_MIN=int(y*(1-1/zoom)/2)+offsetY
	X_MAX=X_MIN+int(x/zoom)
	Y_MAX=Y_MIN+int(y/zoom)
	return(X_MIN,X_MAX,Y_MIN,Y_MAX)


def capture_frames(camera_index, window_name,shared_frame):	#opencv image capture method
	cap=cv2.VideoCapture(camera_index)#,cv2.CAP_DSHOW)
	cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
	cap.set(cv2.CAP_PROP_FPS,30)
	X_DIM=640
	Y_DIM=480
	ZOOM_M=2.0
	if camera_index==2:
		ZOOM_M=2.0
		X_DIM=int(640)
		Y_DIM=int(480)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH,X_DIM)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT,Y_DIM)
	#if camera_index==0:
	#	ZOOM_M=10.0
	#	X_DIM=int(640)
	#	Y_DIM=int(480)
	#	cap.set(cv2.CAP_PROP_FRAME_WIDTH,X_DIM)
	#	cap.set(cv2.CAP_PROP_FRAME_HEIGHT,Y_DIM)
	offsetX=0
	offsetY=0
	with open ("campos.txt",'r') as posfile:
		data=posfile.readline().split()
		data+=posfile.readline().split()
		if camera_index==0:
			offsetX=int(data[0])
			offsetY=int(data[1])
		else:
			offsetX=int(data[2])
			offsetY=int(data[3])
	X_MIN=int(X_DIM*(1-1/ZOOM_M)/2)+offsetX
	Y_MIN=int(Y_DIM*(1-1/ZOOM_M)/2)+offsetY
	X_MAX=X_MIN+int(X_DIM/ZOOM_M)
	Y_MAX=Y_MIN+int(Y_DIM/ZOOM_M)
	X_MIN=160
	X_MAX=480
	Y_MIN=120
	Y_MAX=360
	t1s=time.time()
	X_MIN,X_MAX,Y_MIN,Y_MAX=update_crop(camera_index,ZOOM_M,X_DIM,Y_DIM)
	X_MIN2,X_MAX2,Y_MIN2,Y_MAX2=update_crop(camera_index,2,X_DIM,Y_DIM)
	t_l=time.time()
	while True:
		if (time.time()-t1s)>1:
			t1s=time.time()
			X_MIN,X_MAX,Y_MIN,Y_MAX=update_crop(camera_index,ZOOM_M,X_DIM,Y_DIM)
		t1=time.time()
		ret= cap.grab()
		ret, frame = cap.retrieve()
		if ret:
			dim_frame=frame.shape
			frame_r=frame
			frame=frame[Y_MIN:Y_MAX,X_MIN:X_MAX]
			frame2=frame_r[Y_MIN2:Y_MAX2,X_MIN2:X_MAX2]
			if camera_index==2:
				frame=cv2.resize(frame,(320,240))
				shared_frame["im2"]=frame2
			#if camera_index==0:
			#	frame=cv2.resize(frame,(320,240))
			#	shared_frame["im2"]=frame2
			shared_frame["im"]=frame
		#time.sleep(0.01)
		if (time.time()-t_l)>2:
			print(camera_index, ": ",1/(time.time()-t1))
			t_l=time.time()
	cap.release()


def ffmpeg_capture(camera_index, window_name,shared_frame):
	command = [# Start the ffmpeg process
			'ffmpeg',
			# Input device
			'-loglevel','error',
			'-input_format','mjpeg',
			'-video_size','640x480',
			'-i', f'/dev/video{camera_index}', 
			'-vf', 'fps=30', #'30',  # Filter for 1 frame per second
			'-pix_fmt', 'bgr24',  # Pixel format for OpenCV
			'-f','rawvideo','-',
			#'-vcodec', 'rawvideo', '-an', '-',  # Output encoding and destination
	]
	

	size=[480,640]#set paramters for crop
	X_DIM=size[1]
	Y_DIM=size[0]
	ZOOM_M=2.0

	time_update_crop=time.time()
	print_time=time.time()
	t_loop=time.time()

	X_MIN,X_MAX,Y_MIN,Y_MAX=update_crop(0,ZOOM_M,X_DIM,Y_DIM)#set crop

	using_double=False
	if using_double:
		ZOOM_M2=10.0
		X_MIN2,X_MAX2,Y_MIN2,Y_MAX2=update_crop(camera_index,ZOOM_M2,X_DIM,Y_DIM)
	with sp.Popen(command, stdout=sp.PIPE, bufsize=10**5) as pipe:
		while True:
			if (time.time()-time_update_crop)>1:#update crop values each second
				X_MIN,X_MAX,Y_MIN,Y_MAX=update_crop(0,ZOOM_M,X_DIM,Y_DIM)
				if using_double:
					X_MIN2,X_MAX2,Y_MIN2,Y_MAX2=update_crop(0,ZOOM_M2,X_DIM,Y_DIM)
				time_update_crop=time.time()

            
			raw_image = pipe.stdout.read(size[0]*size[1]*3)# Read one frame's worth of data\
			pipe.stdout.flush()# Clear the buffer to prevent overflow

			frame =  np.frombuffer(raw_image, dtype='uint8')#turn image from bytes to array
			frame_o = frame.reshape((size[0],size[1], 3))	#reshape image

			frame=frame_o[Y_MIN:Y_MAX,X_MIN:X_MAX]	#crop the image
			shared_frame["im"]=cv2.resize(frame,(320,240))
			if using_double:
				frame2=frame_o[Y_MIN2:Y_MAX2,X_MIN2:X_MAX2]
				shared_frame["im2"]=cv2.resize(frame2,(320,240))


			if (time.time()-print_time)>2:#print frame rate
				print(camera_index, ": ",1/(time.time()-t_loop))
				print_time=time.time()
			t_loop=time.time()






if __name__=="__main__":
	sound_select=0
	#from gpiozero import Button
	import RPi.GPIO as GPIO
	GPIO.setmode(GPIO.BOARD)
	GPIO.setup(38,GPIO.IN,pull_up_down=GPIO.PUD_UP)
	#t_button=Button(20)
	
	sound_names={}
	sound_names["shoot"]="sounds/shoot/"+str(os.listdir("sounds/shoot")[sound_select])
	sound_names["hit"]="sounds/hit/"+str(os.listdir("sounds/hit")[sound_select])
	sound_names["reload"]="sounds/reload/"+str(os.listdir("sounds/reload")[sound_select])
	sound_names["kill"]="sounds/kill/"+str(os.listdir("sounds/kill")[sound_select])
	sound_names["dead"]="sounds/dead/"+str(os.listdir("sounds/dead")[sound_select])
	print(sound_names["kill"],"\n")
	pygame.init()#setup sound
	mixer.init()
	mixer.Channel(3).set_volume(1.0)
	mixer.Channel(3).play(mixer.Sound(sound_names["dead"]))
	chan_num=0 #keeps track of what channel is being used on mixer

	


	run_arduino=True#arduino thread
	if run_arduino:
		serial_arr=multiprocessing.Array("i",5)
		cam_arr=multiprocessing.Array("i",5)
		switch_on=False
		prev_switch_state=switch_on
		a_p=Process(target=serial_process, args=(serial_arr,))
		a_p.start()


	a=time.time()	#pre while
	run_main_loop=True
	print_time_m=time.time()
	send_time=time.time()


	failed_conn=False	#start socket
	s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	print("attempting connection to .23")
	s.connect(("192.168.1.3",2345))
	print("connected to",s)


	health=100	#initialize player stats
	prev_health=health
	dead=False
	dead_time=time.time()
	mag_size=10 #shots = + 1
	fire_rate=11
	firing_mode='a' #define firing mode
	mag_count=mag_size
	shot_counter=3	#to initialize burst
	reload_dur=3.7
	reload_time=time.time()
	reload_playing=False
	reload_sound_playing=False
	last_shot_time=time.time()
	shoot_override=False


	manager=multiprocessing.Manager()	#make sharing dicts for iamges
	shared_1=manager.dict()
	shared_2=manager.dict()
	shared_1["im"]=[]
	shared_1["im2"]=[]
	shared_2["im"]=[]
	shared_2["im2"]=[]
	cam2_thread=multiprocessing.Process(target=capture_frames, args=(2,"cam2",shared_2,))
	#cam1_thread=multiprocessing.Process(target=capture_frames, args=(0,"cam1",shared_1,))
	cam1_ffthread=multiprocessing.Process(target=ffmpeg_capture,args=(0,"cam1",shared_1,))


	cam2_on=True
	cam1_on=True

	if cam2_on:
		cam2_thread.start()	#start cameras
		print("cam_thread2 started")
		time.sleep(2)
	if cam1_on:
		cam1_ffthread.start()
		print("cam_thread1 started")
		time.sleep(1)

		# Make sure cam comes on
		time.sleep(3)
		ff_cam_index=0
		while not cam1_ffthread.is_alive():
			ff_cam_index+=1
			cam1_ffthread=multiprocessing.Process(target=ffmpeg_capture,args=(ff_cam_index,"cam1",shared_1,))
			cam1_ffthread.start()
			time.sleep(3)

	gpio_trigger=False
	def trigger_clicked():
		global gpio_trigger
		gpio_trigger=True
		print("TRUE GPIO")
	time.sleep(1)




	print("starting main")
	t_switch_on=time.time()
	while (run_main_loop):
		if (time.time()-print_time_m)>1:#print out player info
			print("health: ",health,"loop: ",time.time()-a,"send: ",send_time) #" image capture time: ",t3-t2,t4-t2)	#time printing
			try:
				print("T: ",t_bottom_while_loop-t_before_player_logic,t_before_recieve-t_before_player_logic,t_before_send-t_before_recieve,t_bottom_while_loop-t_before_send)
			except:
				pass
			print_time_m=time.time()
		


		t_before_player_logic=time.time()
		if health<=0:#check if dead
			dead=True
			mag_count=mag_size
		else:
			dead=False
		prev_switch_state=switch_on
		gpio_trigger=False
		#print(GPIO.input(38))
		#t_button.when_pressed=trigger_clicked
		#print(gpio_trigger)
		if not GPIO.input(38):#if serial_arr[0]==1:	#switch on or off
			switch_on=True
			t_switch_on=time.time()
		else:
			switch_on=False
		t2=time.time()
		if not dead:
			shoot_or_no=(trigger_handler(switch_on,firing_mode,fire_rate))	#transfer block and firing mode handler
			if shoot_or_no:
				if mag_count>0:
					last_shot_time=time.time()
					sound_handler(chan_num,1.5,sound_names["shoot"])

			if shoot_override:	#if remote shooting
				shoot_or_no=True
				shoot_override=False
			if mag_count>0:#check if mag empty
				if shoot_or_no:
					mag_count-=1
			else: #blanks
				reload_playing=True
				shoot_or_no=0

			if serial_arr[2]:	#if reload start reloading
				reload_playing=True
				serial_arr[2]=0

			if not reload_playing:	#reload control logic
				reload_time=time.time()
			else:
				if not reload_sound_playing and (time.time()-reload_time)>0.4:
					sound_handler(7,1.3,sound_names["reload"])
					reload_sound_playing=True
				shoot_or_no=False
				if((time.time()-reload_time)>reload_dur):
					mag_count=mag_size
					reload_playing=False
					reload_sound_playing=False
		else:#if dead dont shoot
			shoot_or_no=False





		t_before_recieve=time.time()
		ready=select.select([s],[],[],0.005)	#recieve data start
		if ready[0]:
			in_data=s.recv(1024).decode("utf-8")	#get data
			index_s=0
			string_arr=in_data.split(',')	#turn data to array
			try:
				if string_arr[0]=='1':#got a hit
					sound_handler(chan_num+3,1,sound_names["hit"])
				if string_arr[0]=='2':#got a kill
					sound_handler(chan_num+3,1,sound_names["hit"])
					sound_handler(8,1,sound_names["kill"])
				if string_arr[0]=='3':#remote shoot
					shoot_override=True
				if len(string_arr)>1:
					health=int(string_arr[1])#update health
					if health>101:
						health=100
					serial_arr[1]=health#send health to arduino
					if (health<=0) and not dead:
						sound_handler(8,1,sound_names["dead"])#play sound if dead
				if len(string_arr)>2:
					fire_rate=int(string_arr[2])#get fire rate paramter
				if len(string_arr)>3:
					mag_size=int(string_arr[3])#get mag size
					if mag_count>mag_size:
						mag_count=mag_size
				if len(string_arr)>4:
					firing_mode=string_arr[4]#get firing mode
			except:
				pass






		t_before_send=time.time()
		if shoot_or_no:	#send shot logic
			frame=np.concatenate((shared_1["im"],shared_2["im"]),axis=0)	#concatinate frames
			cv2.imwrite("dimg.jpg",frame)
			im_file=open("dimg.jpg",'rb')
			data=im_file.read(1024*5)#read in first packet to be sent
			#print("sending data")
			try:
				send_time=time.time()
				while data:#send data for length of image file
					s.send(data)
					data=im_file.read(1024*5)
				im_file.close()
				s.send(b"next")#send b'next' as cutoff term when all data has been sent
				send_time=time.time()-send_time


			except:#if sending fails then try to restart connection
				failed_conn=True
				while failed_conn:
					try:
						s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
						s.connect(("192.168.1.3",2345))
						print("connected to",s)
						failed_conn=False
					except:
						time.sleep(0.5)
						print("conn failed retrying")
						failed_conn=True
				try:
					im_file.close()
				except:
					print("error closing file")
		t_bottom_while_loop=time.time()


	a_p.join()#join arduino after while is finished
