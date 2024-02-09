import os, sys
import cv2
import numpy as np
import time
import multiprocessing
import torch


#server
import socket
import select
import datetime
import os
import threading

player_imgs=[""]*4
server_ims=[""]*4

inferencing_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(inferencing_path, os.pardir))
comm_dir=r"\linkfiles\comm"
os.chdir(inferencing_path)
num_players=3
def main():
    global num_players
    global comm_dir
    headshots_on=True
    run_while=True
    import torch, torchvision
    import ultralytics
    from ultralytics import YOLO
    import math

    if not os.path.isdir(parent_dir+r"\linkfiles"):
        os.mkdir(parent_dir+r"\linkfiles")
    if not os.path.isdir(parent_dir+comm_dir):
        os.mkdir(parent_dir+comm_dir)
    if not os.path.isdir(parent_dir+r"\linkfiles\incoming_images"):
        os.mkdir(parent_dir+r"\linkfiles\incoming_images")
    for i in range(num_players):
        if not os.path.exists(parent_dir+comm_dir+r"\comm"+str(i)+".txt"):
            comm_file=open(parent_dir+comm_dir+r"\comm"+str(i)+".txt",'w')
            comm_file.write("0,100,5,25,a,25,150")
            comm_file.close()
        if not os.path.isdir(parent_dir+r"\linkfiles\incoming_images"+r"\_p"+str(i)+"_"):
            os.mkdir(parent_dir+r"\linkfiles\incoming_images"+r"\_p"+str(i)+"_")

    file_que=[]
    
    
    

    #needs to be initialized with values
    players_ims_list=[]*num_players
    players_ims_list_prev=[]*num_players


    slow_print_time=time.time()



    #player activity and setup
    players_teams=[0]*num_players
    players_health=[100]*num_players
    players_dead=[False]*num_players
    players_dead_prev=[False]*num_players
    players_shooting=[False]*num_players
    for i in range(num_players):
        if (i>=math.ceil(num_players/2)):
            players_teams[i]=1

    #players_teams[1]=0

    #player stats
    players_accuracy=[10000]*num_players
    players_shot_count=[0.01]*num_players
    players_hit_count=[0]*num_players
    players_death_count=[0]*num_players


    players_dead_time=[time.time()]*num_players


    warm_time=time.time()
    t_p_i=time.time()

    l_t=time.time()
    t_u=time.time()
    respawn_time=3  #respawn time


    manager = multiprocessing.Manager()#data share
    # shared_data=manager.dict()
    # shared_data["acc"]=0
    # shared_data["mask"]=[]
    # shared_data["img"]=[]
    # shared_data["headshot"]=False
    # shared_data["result"]=[]
    # shared_data["category"]=1
    # shared_data["read"]=0
    # for i in range(num_players):
    #     shared_data["stats_p"+str(i)]=[players_shot_count[i],players_hit_count[i],players_death_count[i]]
    #view_thread = multiprocessing.Process(target=view_handler, args=(shared_data,))
    #view_thread.start()



    shared_que_info=manager.dict()
    shared_que_info["file_que_new"]=[]
    shared_que_info["signal"]=0
    for i in range(num_players):
        shared_que_info["server_ims"+str(i)]=""
    shared_que_thread=multiprocessing.Process(target=file_que_handler, args=(shared_que_info,))
    shared_que_thread.start()

    server_thread=multiprocessing.Process(target=server,args=(shared_que_info,))
    server_thread.start()



    respawns_on=True
    update_que=True


    #This is the head and body seg model
    model_main = YOLO('custom.pt')


    #categorizer model
    model_categorizer=torchvision.models.resnet18(pretrained=False)
    num_ftrs = model_categorizer.fc.in_features
    model_categorizer.fc = torch.nn.Linear(num_ftrs, 2)
    model_categorizer.load_state_dict(torch.load("model.pth"))
    model_categorizer=model_categorizer.to("cuda")
    model_categorizer.eval()


    #setup gun configs
    players_gun_details=[""]*num_players
    players_damage_mult_b=[0]*num_players
    players_damage_headshot=[0]*num_players
    if not os.path.exists("gun_info.txt"):
        gun_info_file=open("gun_info.txt",'w')
        for i in range(num_players):
            gun_info_file.write(",5,25,a,25,150\n")
        gun_info_file.close()
    gun_info_file=open("gun_info.txt",'r')
    for i in range(num_players):
        gun_info=gun_info_file.readline()
        if gun_info!="":
            gun_info=",5,25,a,25,150"
        players_gun_details[i]=gun_info
        players_damage_mult_b[i]=int(players_gun_details[i].split(',')[4])
        players_damage_headshot[i]=int(players_gun_details[i].split(',')[5])
    gun_info_file.close()


    t_last_inf=0
    hit_class="REG"
    while run_while:
        top_while_time=time.time()



        #Update player's gun configuration
        gun_info_file=open("gun_info.txt",'r')
        for i in range(num_players):
            players_gun_details[i]=gun_info_file.readline()
            players_damage_mult_b[i]=int(players_gun_details[i].split(',')[4])
            players_damage_headshot[i]=int(players_gun_details[i].split(',')[5])
        gun_info_file.close()
        


        #file que updates
        if update_que:
            file_que=file_que+shared_que_info["file_que_new"]
            if len(file_que)>0:
                if shared_que_info["signal"]>5:
                    shared_que_info["signal"]=0
                else:
                    shared_que_info["signal"]+=1
        tf=time.time()



        #slow prints
        if ((time.time()-slow_print_time)>5):
            slow_print_time=time.time()
            for i in range(num_players):
                print("P"+str(i)+" stats: ",players_health[i])
            print("Warm:",warm_time, " Loop T: ",time.time()-l_t, " FILE QUE: ",file_que)
        l_t=time.time()



        #respawn logic
        if respawns_on:
            for i in range(num_players):
                if players_health[i]<=0:
                    players_dead[i]=True
                    if (time.time()-players_dead_time[i])>respawn_time:
                        players_dead_prev[i]=False
                        players_health[i]=100
                        players_dead[i]=False
                        comm_file=open(parent_dir+comm_dir+"\comm"+str(i)+".txt",'w')
                        comm_file.write("0,"+str(players_health[i])+players_gun_details[i])
                        comm_file.close()
                else:
                    players_dead_time[i]=time.time()
                    players_dead[i]=False
            t2=time.time()



        #Get from file que
        tpfq=time.time()
        if(len(file_que)!=0):#if a file is in file que
            if len(file_que)>10:
                file_que=file_que[(len(file_que)-9):len(file_que)]
            img=file_que[0]

            
            #time.sleep(0.003)
            if(len(img)>0):#determine who is shooting
                for i in range(num_players):
                    if("_p"+str(i)+"_" in img):
                        players_shooting[i]=True
                        players_shot_count[i]+=1
                    else:
                        players_shooting[i]=False

            #remove current file from que
            if (len(file_que)>1):
                file_que=file_que[1:]
            else:
                file_que=[]
            
            for active_player_index in range(num_players):
                if players_shooting[active_player_index] and not players_dead[active_player_index]:
                    try:
                        result_main=model_main.predict(img,verbose=False)#get inference from model
                        hs_mask_bool=False
                        valid_hit=0.0
                        hs_hit=0.0
                        #getting results
                        try:
                            masks_generated=[]
                            mask_generated=torch.zeros(result_main[0].masks[0].data[0].cpu().shape)
                            hs_mask_genereated=torch.zeros(result_main[0].masks[0].data[0].cpu().shape)
                            for ind in range(len(result_main[0].masks.cpu())):
                                it_mask=result_main[0].masks[ind].data[0].cpu()
                                it_box=result_main[0].boxes[ind].data[0].cpu()
                                if it_box[5]==0.0:#only humans
                                    if (it_box[4]>0.3):#accuracy threshold
                                        masks_generated.append(it_mask)
                                        mask_generated+=it_mask
                                elif it_box[5]==1.0:
                                    if (it_box[4]>0.3):
                                        hs_mask_genereated+=it_mask
                                        hs_mask_bool=True
                                        
                        except:
                            print("found nothing")
                        if ((len(masks_generated)>0) or hs_mask_bool):#if there are humans
                            dims=mask_generated.shape
                            valid_hit+=(mask_generated[int(dims[0]*3/4)][int(dims[1]/2)])
                            valid_hit+=(mask_generated[int(dims[0]/4)][int(dims[1]/2)])
                            shape_hs=hs_mask_genereated.shape
                            hs_hit+=hs_mask_genereated[int(shape_hs[0]/4)][int(shape_hs[1]/2)]
                            hs_hit+=hs_mask_genereated[int(shape_hs[0]*3/4)][int(shape_hs[1]/2)]
                            if hs_hit>0.0:
                                head_shot_var=True
                                valid_hit=1000.09
                            else:
                                head_shot_var=False


     
                            if (valid_hit>0.0):#if there is a valid hit
                                #CATEGORIZE
                                hit_class=categorize_image(model_categorizer,img,mask_generated)
                                attacking_player_index=active_player_index
                                #Divvy up damage and hit markers 
                                if hit_class=="REG":
                                    if (players_teams[active_player_index])==0:
                                        if num_players>=3:
                                            attacked_player_index=2
                                        else:
                                            attacked_player_index=1
                                    else:
                                        attacked_player_index=0
                                elif hit_class=="ORG":
                                    if (players_teams[active_player_index])==0:
                                        if num_players==4:
                                            attacked_player_index=3
                                        elif num_players==3:
                                            attacked_player_index=2
                                        else:
                                            attacked_player_index=1
                                    elif num_players>2:
                                        attacked_player_index=1
                                    else:
                                        attacked_player_index=0



                                #Determine damage
                                damage_mult=players_damage_mult_b[active_player_index]
                                if head_shot_var:
                                    damage_mult=players_damage_headshot[active_player_index]
                                



                                #print("P"+str(active_player_index)+" SHOT P"+str(attacked_player_index)+" for "+str(damage_mult)+" damage")
                                players_health[attacked_player_index]-=damage_mult
                                if players_health[attacked_player_index]<=0:
                                    players_health[attacked_player_index]=0
                                    players_dead[attacked_player_index]=True
                                    players_death_count[attacked_player_index]+=1



                                #Write to comm files
                                comm_file=open(parent_dir+comm_dir+"\\comm"+str(active_player_index)+".txt",'w')
                                if players_dead[attacked_player_index]:
                                    if not players_dead_prev[attacked_player_index]:
                                        comm_string="2,"+str(players_health[active_player_index])
                                        players_dead_prev[attacked_player_index]=True
                                    else:
                                        comm_string="0,"+str(players_health[active_player_index])
                                else:
                                    comm_string="1,"+str(players_health[active_player_index])
                                comm_file.write(comm_string+players_gun_details[active_player_index])#give hit marker
                                comm_file.close()

                                comm_file=open(parent_dir+comm_dir+"\comm"+str(attacked_player_index)+".txt",'w')
                                comm_string="0,"+str(players_health[attacked_player_index])#write health
                                comm_file.write(comm_string+players_gun_details[attacked_player_index])
                                comm_file.close()   


                        if valid_hit>0:
                            print("Full loop: ",time.time()-top_while_time," p",attacking_player_index," HIT ", "p",attacked_player_index, " for ",damage_mult)
                        else:
                            print("Full loop: ",time.time()-top_while_time)#" detection: ",time.time()-tpfq," update: ",tf-top_while_time, " respawn: ",tpfq-tf)
                    except Exception as e:
                        t, o, b=sys.exc_info()
                        print("Exception:",e, b.tb_lineno)
                    t_last_inf=time.time()
        else:
            if (time.time()-t_last_inf)>0.5:
                t_last_inf=time.time()
                temp_result=model_main("testim.JPG",verbose=False)
                temp_class=categorize_image(model_categorizer,"testim.JPG",temp_result[0].masks[0].data[0].cpu())


def categorize_image(model_categorizer, img,mask_generated):
    import torch.nn.functional as F
    im_data=cv2.imread(img)
    mask_tensor=(mask_generated).float().unsqueeze(0).unsqueeze(0).to("cuda")
    resized_mask_tensor=F.interpolate(mask_tensor, size=(640,448),mode='bilinear')
    resized_mask_tensor=resized_mask_tensor.expand(-1,3,-1,-1)
    im_tensor=torch.tensor(im_data).permute(2,1,0).float().unsqueeze(0).to("cuda")
    resized_im_tensor=F.interpolate(im_tensor, size=(640,448),mode='bilinear')
    mask_image_tensor=resized_im_tensor*resized_mask_tensor
    #categorize
    with torch.no_grad():
        output=model_categorizer(mask_image_tensor)
        _, predicted = torch.max(output.data, 1)
    #confidence=torch.nn.functional.softmax(output,dim=1)
    if predicted.item()==1:
        return("REG")
    elif predicted.item()==0:
        return("ORG")


def new_in_lists(que,parent_dir,image_dir,new_l,old_l):
    for item in new_l:
        if item not in old_l:
            que.append(parent_dir+"\\"+image_dir+"\\"+item)
    return(que)



def file_que_handler(shared_que_info):
    global parent_dir
    global player_imgs
    os.chdir(parent_dir)
    #This thread handles searching the incoming images directory and passes new files to the
    #  file que in the main loop while co-ordinating such that no images are lost or not inferenced on
    print("Starting file que thread")
    global num_players
    images_dir=r"linkfiles\incoming_images"
    image_lists=[]
    image_lists_prev=[]
    for i in range(num_players):
        try:
            image_lists.append(os.listdir(images_dir+r"\_p"+str(i)+"_"))
        except Exception as e:
            image_lists.append([""])
    for image_list in image_lists:
        image_lists_prev.append(image_list)

    t_print=time.time()
    signal_prev=shared_que_info["signal"]
    local_que=[]
    prev_file=[""]*num_players
    newest_file=[""]*num_players
    while(True):
        #This is the co-ordinating signal to the main, when the previous value is different from the value in main 
        # it clears the local_que because the main has appended the local que and no longer needs its contents
        time.sleep(0.001)
        if signal_prev!=shared_que_info["signal"]:
            local_que=[]
            signal_prev=shared_que_info["signal"]
        t_top_while=time.time()
        # for i in range(num_players):
        #     image_lists_prev[i]=image_lists[i]
        #     try:
        #         image_lists[i]=os.listdir(images_dir+r"\_p"+str(i)+"_")
        #     except:
        #         image_lists[i]=[""]
        #local_que_prev=local_que
        for i in range(num_players):
            pass
            #local_que=new_in_lists(local_que,parent_dir,images_dir+r"\_p"+str(i)+"_",image_lists[i],image_lists_prev[i])
            prev_file[i]=newest_file[i]
            newest_file[i]=shared_que_info["server_ims"+str(i)]
            if prev_file[i]!=newest_file[i]:
                if newest_file[i]!="":
                    if not newest_file[i] in local_que:
                        local_que.append(newest_file[i])


        shared_que_info["file_que_new"]=local_que
        
        if (time.time()-t_print)>10:
            pl_im_stngs=[]
            for i in range(num_players):
                pl_im_stngs.append(shared_que_info["server_ims"+str(i)])
            print("File que loop time: ", time.time()-t_top_while,pl_im_stngs)
            t_print=time.time()















































































def server(shared_que_info):
    inferencing_path = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(inferencing_path, os.pardir))


    linksfiles_dir=parent_dir+r"\linkfiles"
    images_parent_dir=linksfiles_dir+r"\incoming_images"
    comm_dir=linksfiles_dir+r"\comm"
    global saving_path
    my_ip = socket.gethostname()#"192.168.1.2"
    print("HOSTNAME: ",my_ip)
    s = socket.socket()
    s.settimeout(3)



    s.bind((my_ip, 2345))
    print("Server Listening")
    s.listen(1)
    print("Server Starting")

    global server_ims

    while True:
        time.sleep(0.05)
        try:
            conn, addr = s.accept()
            print("Connected to:", addr)
        except:
            addr=[""]
        if addr[0] == '192.168.1.11':  # Device 1 IP
            if not os.path.isdir(images_parent_dir+r"\_p0_"):
                os.mkdir(images_parent_dir+r"\_p0_")
            if not os.path.exists(comm_dir+r"\comm0.txt"):
                comm_file=open(comm_dir+r"\comm0.txt",'w')
                comm_file.write("0,100,5,25,a,25,150")
                comm_file.close()
            t1 = threading.Thread(target=conn_comm, args=(conn, images_parent_dir+r"\_p0_", comm_dir+r"\comm0.txt",shared_que_info))
            print("Starting connection to Device 1")
            t1.start()
        elif addr[0] == '192.168.1.12':  # Device 2 IP
            if not os.path.isdir(images_parent_dir+r"\_p1_"):
                os.mkdir(images_parent_dir+r"\_p1_")
            if not os.path.exists(comm_dir+r"\comm1.txt"):
                comm_file=open(comm_dir+r"\comm1.txt",'w')
                comm_file.write("0,100,5,25,a,25,150")
                comm_file.close()
            t2 = threading.Thread(target=conn_comm, args=(conn, images_parent_dir+r"\_p1_", comm_dir+r"\comm1.txt",shared_que_info))#,shared_que_info))
            print("Starting connection to Device 2")
            t2.start()
        elif addr[0] == '192.168.1.13' or addr[0] == '192.168.1.15':  # Device 2 IP
            if not os.path.isdir(images_parent_dir+r"\_p2_"):
                os.mkdir(images_parent_dir+r"\_p2_")
            if not os.path.exists(comm_dir+r"\comm2.txt"):
                comm_file=open(comm_dir+r"\comm2.txt",'w')
                comm_file.write("0,100,5,25,a,25,150")
                comm_file.close()
            t3 = threading.Thread(target=conn_comm, args=(conn, images_parent_dir+r"\_p2_", comm_dir+r"\comm2.txt",shared_que_info))
            print("Starting connection to Device 3")
            t3.start()
        elif addr[0]!="":
            if not os.isdir(images_parent_dir+r"\_p3_"):
                os.mkdir(images_parent_dir+r"\_p3_")
            if not os.exists(comm_dir+r"\comm3.txt"):
                comm_file=open(comm_dir+r"\comm3.txt",'w')
                comm_file.write("0,100,5,25,a,25,150")
                comm_file.close()
            t4 = threading.Thread(target=conn_comm, args=(conn, images_parent_dir+r"\_p3_", comm_dir+r"\comm3.txt",shared_que_info))
            print("Starting connection to Device 4")
            t4.start()

def clear_folders(dir_images):
    try:
        for f_name in os.listdir(dir_images):
            try:
                os.remove(str(dir_images)+"/"+str(f_name))
            except:
                print("Could not remove: ",f_name)
        print("Done clearing")
    except Exception as e:
        print("Did not clear error: ",e)


def conn_comm(conn_local, dir_images,comm_file_name,shared_que_info):
    clear_folders(dir_images)
    global player_imgs
    global num_players
    global server_ims
    player_index=0
    for i in range(num_players):
        if ("_p"+str(i)+"_") in dir_images:
            player_index=i

    last_send=time.time()
    comm_string="1,100"

    got_data=False
    im_data=b""
    extra_data=b""
    data_overflow=False

    hit_prev=False
    kill_prev=False

    count_variable=0
    print("Starting connection with: ", conn_local)
    while True:
        ready=select.select([conn_local],[],[],0.01)
        try:
            #This is responsible for sending data to the raspberry pis.
            if ((time.time()-last_send)>0.01):
                comm_file=open(comm_file_name,'r+')
                comm_string=""
                comm_string+=comm_file.readline()
                if(len(comm_string)>0):
                    if(comm_string[0]=='1'):#send hit marker
                        if hit_prev:
                            comm_string='0'+comm_string[1:]
                            comm_file.seek(0)
                            comm_file.write(comm_string)
                        hit_prev=True
                    else:
                        hit_prev=False

                    if(comm_string[0]=='2'):#send kill notification
                        if kill_prev:
                            comm_string='0'+comm_string[1:]
                            comm_file.seek(0)
                            comm_file.write(comm_string)
                        kill_prev=True
                    else:
                        kill_prev=False
                comm_file.close()
                conn_local.send(comm_string.encode("ascii"))#send the player data
                last_send=time.time()



            
            #This recieves the image data from the raspberry pis.
            ready=select.select([conn_local],[],[],0.01)
            if ready[0]:
                in_data=conn_local.recv(1024*5)
                if in_data:
                    chunks=[]
                    if data_overflow:
                        chunks.append(excess_data)
                    in_start_time=time.time()

                    #Loop to grab all image data
                    while in_data!=b"next":
                        if (len(in_data)>1):
                            if (in_data.find(b"next")>0):
                                chunks.append(in_data[0:in_data.find(b"next")])
                                #When the data length exceeds the "next" stop signal the overflow is saved for the next loop
                                if (len(in_data)-in_data.find(b"next"))>4:
                                    excess_data=in_data[in_data.find(b"next")+4:]
                                    data_overflow=True
                                else:
                                    data_overflow=False
                                    excess_data=b""
                                break
                            else:
                                chunks.append(in_data)
                                in_data=conn_local.recv(1024*5)

                    #print("len data: ",len(in_data))
                    #print("image recieved")

                    #Joins the image data and saves the file
                    im_data = b''.join(chunks)
                    count_variable+=1
                    file_test=open(str(dir_images)+"\\"+"incoming_img"+str(count_variable)+".JPG", 'wb')
                    shared_que_info["server_ims"+str(player_index)]=str(dir_images)+"\\"+"incoming_img"+str(count_variable)+".JPG"
                    file_test.write(im_data)
                    file_test.close()
        except Exception as e:
            print("\nException: ", e)
            break
    print("Connection Ended")
    return(0)













































































































































def view_handler(shared_data):
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    import pygame
    import pygame.mixer
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    import tkinter as tk
    riddle_list=[""]
    riddle_answers=[""]
    global parent_dir
    global num_players
    os.chdir(parent_dir)


    mask_folder=parent_dir+r"\saved_data\masks"
    raw_folder=parent_dir+r"\saved_data\raw_images"
    results_folder=parent_dir+r"\saved_data\results"
    if not os.path.isdir(parent_dir+r"\saved_data"):
        os.mkdir(parent_dir+r"\saved_data")
    if not os.path.isdir(mask_folder):
        os.mkdir(mask_folder)
    if not os.path.isdir(raw_folder):
        os.mkdir(raw_folder)
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    
    timestr = time.strftime("%m-%d-%Y-%H-%M-%S")

    mask_dir=mask_folder+"\\"+timestr
    os.mkdir(mask_dir)

    raw_dir=raw_folder+"\\"+timestr
    os.mkdir(raw_dir)

    results_dir=results_folder+"\\"+timestr
    os.mkdir(results_dir)
    
    player_mask_dirs=[]
    for i in range(num_players):
        player_mask_dirs.append(mask_dir+"\\p"+str(i))
        os.mkdir(player_mask_dirs[i])
    player_raw_dirs=[]
    for i in range(num_players):
        player_raw_dirs.append(raw_dir+"\\p"+str(i))
        os.mkdir(player_raw_dirs[i])
    player_results_dirs=[]
    for i in range(num_players):
        player_results_dirs.append(results_dir+"\\p"+str(i))
        os.mkdir(player_results_dirs[i])
    

    #Riddles or math problems game to defuse a bomb
    if not os.path.exists("riddles.txt"):
        with open ("riddles.txt","w") as riddle_file:
            riddle_file.write("I rise in the i fall in the ?\n baskin robins")
    with open ("riddles.txt","r") as riddle_file:
        i=0
        for line in riddle_file:
            line=line.strip()
            if (i%2)==0:
                riddle_list.append(line)
            else:
                riddle_answers.append(line)
            i+=1
    correct_state=True
    correct_state_prev=True

    #This checks if user input is the correct answer
    def submit_answer(answer,correct_answer):
        global correct_state
        #case insensitive
        if  answer.lower()==correct_answer.lower():
            result_label.configure(text="Correct")
            correct_state=True
        else:
            result_label.configure(text="Incorrect")

    #create tkinter window
    root = tk.Tk()
    root.title("Riddle Game")
    root.geometry("500x500")
    root.configure(bg='black')

    #display options for selecting a riddle
    riddle_var=tk.StringVar(root)
    riddle_var.set(riddle_list[0])
    riddle_menu=tk.OptionMenu(root,riddle_var,*riddle_list) 
    riddle_menu.config(height=5,width=100,font=("gameovercre", 18, "bold"))

    #wrap text
    riddle_menu.config(wraplength=400)
    riddle_menu.pack()

    #display entry box for answer
    answer_entry=tk.Entry(root)
    answer_entry.configure(width=20)
    answer_entry.config(font=("gameovercre", 24, "bold"))
    answer_entry.pack()

    #display button for submitting answer
    submit_button=tk.Button(root,text="Submit",command=lambda: submit_answer(answer_entry.get(),riddle_answers[riddle_list.index(riddle_var.get())]))
    submit_button.config(height=5,width=20)
    submit_button.pack()

    #display label for displaying result
    result_label=tk.Label(root,text="Result")
    result_label.config(height=5,width=20)
    result_label.pack()
    prev_riddle=riddle_var.get()

    #initialize pygame mixer
    pygame.mixer.init()
    pygame.mixer.Channel(0).set_volume(1.0)
    bomb_mode=True
    read_prev_value=0

    #setup plotting
    plt.ion()
    temp_img=[[[3,3,3]]]
    plt_img=plt.imshow(temp_img)
    temp_img=np.random.rand(480,640,3)
    plt_img.set_data(temp_img)
    players_stats=[]
    for i in range(num_players):
        players_stats.append(shared_data["stats_p"+str(i)])#shots hits deaths

    
    time_update=time.time()
    p2_dead=False   #defusal character
    masks_saved_count=0
    #Create results directory
    # current_result_save_dir=parent_dir+r"\saved_data\results"+"\\"+str(timestr)
    # os.mkdir(current_result_save_dir)
    # os.mkdir(current_result_save_dir+r"\hits")
    # os.mkdir(current_result_save_dir+r"\misses")
    
    # p1_hit_result_dir=current_result_save_dir+r"\hits\p1"
    # p2_hit_result_dir=current_result_save_dir+r"\hits\p2"
    # p1_miss_result_dir=current_result_save_dir+r"\misses\p1"
    # p2_miss_result_dir=current_result_save_dir+r"\misses\p2"

    # os.mkdir(p1_hit_result_dir)
    # os.mkdir(p2_hit_result_dir)
    # os.mkdir(p1_miss_result_dir)
    # os.mkdir(p2_miss_result_dir)

    

    while True:
        if bomb_mode:#bomb display:
            if (time.time()-time_update)>0.25:
                if not p2_dead:
                    root.update()
                time_update=time.time()
            if riddle_var.get()!=prev_riddle:
                riddle_menu.configure(state="disabled")
                correct_state=False
            prev_riddle=riddle_var.get()
            
            if correct_state_prev and not correct_state:
                pygame.mixer.Channel(0).play(pygame.mixer.Sound("bomb2.mp3"))

            if correct_state:
                pygame.mixer.Channel(0).stop()
                correct_state=True
            
            if not correct_state_prev and correct_state:
                riddle_menu.configure(state="active")
            correct_state_prev=correct_state

        #shared_data["read"] is a signal from main that a new image is ready to be displayed
        if shared_data["read"]!=read_prev_value:
            shared_data["read"]=0
            read_prev_value=shared_data["read"]
            dot_size=5
            try:
                im_data=cv2.imread(shared_data["img"])     #plotting
                img=shared_data["img"]
                if(len(img)>0):#determine who is shooting
                    for i in range(num_players):
                        if ("_p"+str(i)+"_") in img:
                            active_player_index=i
                    im_data=cv2.cvtColor(im_data,cv2.COLOR_BGR2RGB)
                    img_name=img.split("\\")[-1]
                    file_name=player_raw_dirs[active_player_index]+img_name
                    cv2.imwrite(file_name,im_data)
                        
                try:#get data from main
                    im_mask=shared_data["mask"]
                    shot_acc=shared_data["acc"].item()
                    got_mask=True
                    if shared_data["headshot"]:
                        img_name=img.split("\\")[-1]
                        file_name=player_mask_dirs[active_player_index]+img_name[:-4]+"_HS.jpg"
                        cv2.imwrite(file_name,np.array(shared_data["mask"]*255))
                    else:
                        img_name=img.split("\\")[-1]
                        file_name=player_mask_dirs[active_player_index]+img_name
                        cv2.imwrite(file_name,np.array(shared_data["mask"]*255))
                except Exception as e:
                    print(e)
                    im_mask=[]
                    shot_acc=0.00
                    got_mask=False

                
                file_inc=0
                if shared_data["headshot"]:
                    img_name=img.split("\\")[-1]
                    file_name=player_results_dirs[active_player_index]+img_name[:-4]+"_HS.jpg"
                    cv2.imwrite(file_name,shared_data["annotated_frame"])
                else:
                    img_name=img.split("\\")[-1]
                    file_name=player_results_dirs[active_player_index]+img_name
                    cv2.imwrite(file_name,shared_data["annotated_frame"])
                    

                
                try:
                    mask3=np.array(im_mask)
                    dims_t=mask3.shape
                    dims=(dims_t[0],dims_t[1])
                    out_data=cv2.resize(im_data,(dims[1],dims[0]))
                    mask3=mask3[:,:,np.newaxis]

                    #make masked image where the masked area is the regular image and the rest is black
                    mask_image=np.zeros((dims[0],dims[1],3))
                    mask_image[:,:,2]=mask3[:,:,0]*out_data[:,:,0]
                    mask_image[:,:,1]=mask3[:,:,0]*out_data[:,:,1]
                    mask_image[:,:,0]=mask3[:,:,0]*out_data[:,:,2]

                    # masks_saved_count+=1
                    # if p1_shooting:
                    #     if shared_data["headshot"]:
                    #         cv2.imwrite(p1_mask+"\\"+str(masks_saved_count)+"_"+str(int(shot_acc*100))+"_hs.jpg",mask_image)
                    #     else:
                    #         cv2.imwrite(p1_mask+"\\"+str(masks_saved_count)+"_"+str(int(shot_acc*100))+".jpg",mask_image)
                    # else:
                    #     if shared_data["headshot"]:
                    #         cv2.imwrite(p2_mask+"\\"+str(masks_saved_count)+"_"+str(int(shot_acc*100))+"_hs.jpg",mask_image)
                    #     else:
                    #         cv2.imwrite(p2_mask+"\\"+str(masks_saved_count)+"_"+str(int(shot_acc*100))+".jpg",mask_image)


                    out_data=out_data*(mask3+0.5)
                    #Create dots in the center of both images from Pis
                    for i in range(dot_size):
                        for j in range(dot_size):
                            out_data[int(dims[0]/4)+int(dot_size/2)+j][int(dims[1]/2)+int(dot_size/2)+i]=[255,255,255]
                            out_data[int(dims[0]*3/4)+int(dot_size/2)+j][int(dims[1]/2)+int(dot_size/2)+i]=[255,255,255]
                    if shared_data["category"]==0:
                        cv2.circle(out_data,(50,50),8,(200,150,0),-1)
                    else:
                        cv2.circle(out_data,(50,50),8,(50,50,50),-1)
                    plt_img.set_data(out_data/255)

                    cv2.imwrite(file_name,out_data)
                    out_data=cv2.imread(file_name)
                    saving_data=(cv2.cvtColor(out_data, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(file_name,saving_data)
                except Exception as e:
                    t,o,b=sys.exc_info()
                    print(e, b.tb_lineno)
                    im_data=cv2.resize(im_data,(dims[1],dims[0]))
                    for i in range(dot_size):
                        for j in range(dot_size):
                            im_data[int(dims[0]/4)+int(dot_size/2)+j][int(dims[1]/2)+int(dot_size/2)+i]=[255,255,255]
                            im_data[int(dims[0]*3/4)+int(dot_size/2)+j][int(dims[1]/2)+int(dot_size/2)+i]=[255,255,255]
                    plt_img.set_data(im_data/255)
                    
                    #write image to file
                    cv2.imwrite(file_name,cv2.cvtColor(im_data, cv2.COLOR_RGB2BGR))
                if(shot_acc>0):
                    plt.title(str(shot_acc*100))
                else:
                    plt.title(str(0.00))
            except Exception as e:
                print("View error: ",e)


if __name__ == "__main__":
    main()