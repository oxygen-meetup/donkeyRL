#!/usr/bin/env python
'''
Predict Server
Create a server to accept image inputs and run them against a trained neural network.
This then sends the steering output back to the client.
Author: Tawn Kramer
'''
from __future__ import print_function
import os
import argparse
import sys
import json
import time
from datetime import datetime
import asyncore
import json
import shutil
import base64
import random

import numpy as np
import h5py
from PIL import Image
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import keras
from copy import deepcopy
import conf
import throttle_manager
import cv2,my_cv
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from numpy.linalg import norm
import ast
import math
import pickle

first=True
cntr=0
sio = socketio.Server()
app = Flask(__name__)
throttle_man = throttle_manager.ThrottleManager()
model = None
iSceneToLoad = 0
time_step = 0.1
step_mode = "synchronous"
predict=False
cvsolve=True
frame_out=False
vertex_hold=[]
left_hold=[]
left_angle=0
right_hold=[]
right_angle=0
uV=0
best_t=0

def get_slopes(lines):
    slopes=[]
    lengths=[]

    for line in range(len(lines)):
        for x1,y1,x2,y2 in lines[line]:
            try:
                slopes.append((y2-y1)/(x1-x2+.0001))
            except:
                if x1-x2==0:
                    slopes.append(np.inf)
            lengths.append(np.sqrt((x1-x2)**2+(y1-y2)**2))
    return slopes,lengths

def lane_check(lines,slopes):
    left=[]
    left_start=[]
    left_slope=[]
    right=[]
    right_start=[]
    right_slope=[]
    vertex=[]
    for line in range(len(lines)):
        for x1,y1,x2,y2 in lines[line]:
            if slopes[line]>0:
                left.append(np.cross([x1,y1,1],[x2,y2,1]))
                left_start.append([x1,y1])
                left_slope.append(slopes[line])
            elif slopes[line]<0:
                right.append(np.cross([x1,y1,1],[x2,y2,1]))
                right_start.append([x1,y1])
                right_slope.append(slopes[line])
                
    for line1 in left:
        for line2 in right:
            vertex.append(np.cross(line1,line2))

    return vertex,right,left,left_start,right_start,np.array(left_slope),np.array(right_slope)


def delete_lines(lines,slopes):
    dellist=[]
    if len(slopes)==len(lines):
        for index in range(len(slopes)):
            if slopes[index]==np.inf or np.abs(slopes[index])<.4:
              dellist.append(index)  
        newlines=np.delete(lines,dellist,0)
        newslopes=np.delete(slopes,dellist,0)
    else:
        print('slopes != lines')
        
    return newlines,newslopes

def calc_mean(vertex,left_start,right_start,vertex_hold,left_hold,right_hold):
    v=[]
    
    for i in range(len(vertex)):
        scale=vertex[i][2]
        if scale==0:
            scale=.00001
        yy=int(vertex[i][1]/scale)
        if yy<0:
            yy=0
        elif yy>=240:
            yy=239
            
        xx=int(vertex[i][0]/scale)
        if xx<0:
            xx=0
        elif xx>=320:
            xx=319
        v.append([xx,yy])
         
    return np.nanmean(v,0).astype(int),np.nanmean(left_start,0).astype(int),np.nanmean(right_start,0).astype(int)

def extend_line(vertex,point,shape):
    m=(point[1]-vertex[1])/(point[0]-vertex[0]+.000001)
    b=point[1]-m*point[0]
    x=(shape[0]-b)/m
    if np.isnan(x) or np.abs(x)==np.inf:
        x=point[0]
    return int(np.round(x)),shape[0]

def draw_lanes(img,left,right,vertex,color):
    points=[left,right]
    for point in points:
        newx,newy=extend_line(vertex,point,img.shape)
        cv2.line(img, (newx,newy), (int(vertex[0]),int(vertex[1])), color, 2)   


def process_image(image_array):
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    global left_hold
    global right_hold
    global left_angle
    global right_angle
    global uV
    done=False
    colors=[(255,0,0),(0,255,0),(0,0,255)]
    img=cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    manip=deepcopy(img)

    gray = cv2.cvtColor(manip,cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray,50,150)
    
    threshold=20
    minLineLength=5
    maxLineGap=15
    buff_len=6
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength, maxLineGap ) 
        
    if not (lines is None): # some lanes
        slopes,lengths=get_slopes(lines) # distinguish between left/right lanes
        lines,slopes=delete_lines(lines,slopes) # delete horiz lines
        vertex,lineR,lineL,left_start,right_start,left_slopes,right_slopes=lane_check(lines,slopes)

        if vertex: # if there is a vertex from left/right
            uV,uL,uR=calc_mean(vertex,left_start,right_start,vertex_hold,left_hold,right_hold)
            
            vertex_hold.append(uV)

            if len(vertex_hold)>=buff_len:
                del vertex_hold[0]
  
            
            left_angle=np.degrees(np.arctan(np.nanmean(left_slopes[np.isfinite(left_slopes)])))
            right_angle=np.degrees(np.arctan(np.nanmean(right_slopes[np.isfinite(right_slopes)])))
            if math.isnan(right_angle):
                right_angle=0
                
            left_hold.append(left_angle)
            right_hold.append(right_angle)    
            if len(left_hold)>=buff_len:
                del left_hold[0]
            if len(right_hold)>=buff_len:
                del right_hold[0]      
                
            draw_lanes(manip,uL,uR,uV,colors[0])
            cv2.putText(manip,str(int(left_angle)),(uL[0],uV[1]), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(manip,str(int(right_angle)),(uR[0],uV[1]), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(manip,str(uV[0]),(uV[0],uV[1]-20), font, 1,(255,255,255),2,cv2.LINE_AA)
        else:
            
            try:
                del vertex_hold[0] # decrease buffer to eventual stop simulation ie some lines not showing up
            except:
                done=True
            if len(vertex_hold)==0:
                done=True 
                #send_exit_scene()      
                #send_reset_car()
    else:
       done=True 
       #send_exit_scene()      
       #send_reset_car()

    return manip,done

class RLAgent(object):
    def __init__(self):
        self.alpha=10
        self.gamma=.55  # discount factor 1=infinite memory depth
        self.epsilon=.95# for exploration vs exploitation   
        self.q_val=100+np.random.rand(4,4,4,4, 7)*.0001 # high Q to explore more
        self.episode_count=0
        self.discrete_actions=[-.95,-.25,-.125,0,.125,.25,.95]
        self.index_old=tuple([2,2,1,1])
        self.action=3 # default do nothing
        self.reward=0
        self.t=0
        self.reset=True
        self.finished=0
       


    def get_features(self,ob):
        
        features=np.zeros(4)
        features[0]=(ob[0])/320
        features[1]=np.abs(ob[1])/90
        features[2]=ob[2]/90
        features[3]=0#(ob[3])/320
        # will try more advanced  features
    
        return features

    def get_action(self,index):
        q_max=0
        explore=np.random.sample() # should we explore or exploit
        if explore>self.epsilon: # if yes explore
            action=np.round(np.random.sample()*(len(self.discrete_actions)-1)).astype(int) # uniform rand var 0 thru 3 action
            print('random action')
        else: # else exploit the q-values for max in this state, take action          
            action=np.argmax(self.q_val[index])
        
        q_max=self.q_val[index][action]
           
        return q_max,action
    
    #### discretize the feature space
    def get_index(self,features): 
        index=np.zeros(len(features),dtype=np.int8)
        discrete=np.zeros([4,3])
        discrete[0]=(np.array([150,160,170]))/320 
        discrete[1]=(np.array([30,40,50]))/90
        discrete[2]=(np.array([30,40,50]))/90
        discrete[3]=(np.array([130,160,190]))/320 #not being ussed

        for i in range(len(features)): #cleaner than a switch or massive if-else
            j=0
            index[i]=3
            while j<(len(discrete[i][:])):
                
                if features[i]<discrete[i][j]:
                    index[i]=j
                    j=14
                j+=1  
                 
        return tuple(index)
    
    def episode_reset(self):
        self.t=0
        self.alpha=10+self.episode_count*.5  # reset the learning parameter - seems to do better than letting it converge to 0
        self.epsilon+=0.005 # lower exlploration variable as we learn more
        self.action=3
        self.episode_count+=1       
        self.index_old=tuple([2,2,1,1])
        self.reward=0
        self.reset=True
    def rollout_step(self,ob,done):
        self.t+=1
                   
        features=self.get_features(ob)
        
        self.index_new=self.get_index(features) # get index of current state
        last_reward=self.reward
        self.reward=self.t

        # only positive reward for staying in the lane
        if 29<ob[2]<41 and 29<np.abs(ob[1])<41 and 148<ob[0]<172:
            self.reward+=1000
    
        # positive or negative reward for keep the car aimed ahead  
        self.reward=15*(27-np.abs(np.abs(ob[1])-ob[2]))
        
        # punish for being out of bounds
        if ob[0]<149 or ob[0]>171:
            self.reward-=np.abs(ob[0]-160)*12

        # boost reward if improved the reward from last time
        if last_reward-self.reward<0:
            self.reward+=np.abs(last_reward-self.reward)*20
         
        q_max,_ = self.get_action(self.index_new) # get new argmax of Q with new state           
        
        # to handle some weird reset thing happening in simulator
        if not self.reset: 
            self.q_val[self.index_old][self.action]=(1-1/self.alpha)*self.q_val[self.index_old][self.action]+(1/self.alpha)*(self.reward+self.gamma*q_max)

        self.index_old=self.index_new
        self.reset=False
        _,self.action=self.get_action(self.index_new) # take a new action 
        
        if self.episode_count>5000 or self.t>11000:
            self.finished+=1
            print('finished!!!')
            self.episode_reset()
            send_exit_scene()      
            send_reset_car()

        return self.discrete_actions[self.action]
        
class FPSTimer(object):
    def __init__(self):
        self.t = time.time()
        self.iter = 0

    def reset(self):
        self.t = time.time()
        self.iter = 0

    def on_frame(self):
        self.iter += 1
        if self.iter == 100:
            e = time.time()
            #print('fps', 100.0 / (e - self.t))
            self.t = time.time()
            self.iter = 0

timer = FPSTimer()
agent=RLAgent()
@sio.on('Telemetry')
def telemetry(sid, data):
    global timer
    global num_frames_to_send,cntr
    global agent,best_t
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)

        #name of object we just hit. "none" if nothing.
        hit = data["hit"]

        x = data["pos_x"]
        y = data["pos_y"]
        z = data["pos_z"]

        #Cross track error not always present.
        #Will be missing if path is not setup in the given scene.
        #It should be setup in the 3 scenes available now.
        try:
            cte = data["cte"]
        except:
            pass

####  INSERTED RL CODE STUFF

        if predict:
            outputs = model.predict(image_array[None, :, :, :])

        else:
            outputs=[[0],[.0125]]
            img,done=process_image(image_array)

            if not done:    
                if speed<5:
                    outputs=[[agent.rollout_step([vertex_hold[-1][0],right_angle,left_angle,vertex_hold[-1][0]],done),.5]]
                else:
                    outputs=[[agent.rollout_step([vertex_hold[-1][0],right_angle,left_angle,vertex_hold[-1][0]],done),.0125]]

            else:
                print('===================')                  
                print(str(agent.episode_count),' ',str(agent.t),' best_t=',str(best_t),' ',str(agent.finished))
                print('===================') 
                if agent.t>10:
                    # really punish the qvalue on a fail (neg-make bigger pos-make smaller by factor)
                    if agent.q_val[agent.index_old][agent.action]<0:
                        agent.q_val[agent.index_old][agent.action]*=10
                    else:
                        agent.q_val[agent.index_old][agent.action]/=10
                        
                if best_t<agent.t:
                    best_t=agent.t
                    
        if agent.episode_count%100==0 and done:
            # Saving the objects:
            with open('agent.pkl', 'wb') as f: 
                pickle.dump([agent], f)
        if frame_out and agent.episode_count>0 and agent.episode_count<600:     
            strcnt="{:012d}".format(cntr)
            cv2.imwrite('rl_out/houghlines'+strcnt+'.jpg',img)
            cntr+=1 
            
        if done:
            agent.episode_reset()
            send_exit_scene()      
            send_reset_car()
####################################################         
            
        #steering
        steering_angle = outputs[0][0]

        #do we get throttle from our network?
        if conf.num_outputs == 2 and len(outputs[0]) == 2:
            throttle = outputs[0][1]
        else:
            #set throttle value here
            throttle, brake = throttle_man.get_throttle_brake(speed, steering_angle)
        
        #print(steering_angle, throttle)
        #throttle=.25
        #reset scene to start if we hit anything.
        if hit != "none":
            send_exit_scene()
        else:
            send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            #image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('RequestTelemetry', data={}, skip_sid=True)

    timer.on_frame()

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    global timer
    global time_step
    global step_mode
    timer.reset()

    send_settings({"step_mode" : step_mode.__str__(),\
         "time_step" : time_step.__str__()})
    
    send_control(0, 0)

@sio.on('ProtocolVersion')
def on_proto_version(sid, environ):
    #print("ProtocolVersion ", sid)
    pass
@sio.on('SceneSelectionReady')
def on_fe_loaded(sid, environ):
    #print("SceneSelectionReady ", sid)
    send_get_scene_names()

@sio.on('SceneLoaded')
def on_scene_loaded(sid, data):
    #print("SceneLoaded ", sid)
    pass

@sio.on('SceneNames')
def on_scene_names(sid, data):
    #print("SceneNames ", sid)
    if data:
        names = data['scene_names']
        #print("SceneNames:", names)
        global iSceneToLoad
        send_load_scene(names[iSceneToLoad])

def send_get_scene_names():
    sio.emit(
        "GetSceneNames",
        data={            
        },
        skip_sid=True)

def send_control(steering_angle, throttle):
    sio.emit(
        "Steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

def send_load_scene(scene_name):
    #print("Loading", scene_name)
    sio.emit(
        "LoadScene",
        data={
            'scene_name': scene_name.__str__()
        },
        skip_sid=True)

def send_exit_scene():
    sio.emit(
        "ExitScene",
        data={
            'none': 'none'
        },
        skip_sid=True)

def send_reset_car():
    sio.emit(
        "ResetCar",
        data={            
        },
        skip_sid=True)

def send_settings(prefs):
    sio.emit(
        "Settings",
        data=prefs,
        skip_sid=True)

def go(model_fnm, address, iScene):
    global model
    global app
    global iSceneToLoad

    model = keras.models.load_model(model_fnm)

    #In this mode, looks like we have to compile it
    model.compile("sgd", "mse")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    #which scene to load
    iSceneToLoad = iScene

    # deploy as an eventlet WSGI server
    try:
        eventlet.wsgi.server(eventlet.listen(address), app)
    except KeyboardInterrupt:
        #unless some hits Ctrl+C and then we get this interrupt
        print('stopping')


# ***** main loop *****
if __name__ == "__main__":
    np.random.seed(0) 
    parser = argparse.ArgumentParser(description='sim_server')
    parser.add_argument('model', type=str, help='model name')
    parser.add_argument('--i_scene', default=0, help='which scene to load')
    parser.add_argument('--step_mode', default="asynchronous", help='how to advance time in sim (asynchronous|synchronous)')
    parser.add_argument('--time_step', type=float, default=0.1, help='how far to advance time in sim when synchronous')
    parser.add_argument(
          'image_folder',
          type=str,
          nargs='?',
          default='',
          help='Path to image folder. This is where the images from the run will be saved.'
      )
    parser.add_argument('predict',default=False, help='use predictor to output actions')

    parser.add_argument('frame_out',default=True, help='output analyzed frame')
    args = parser.parse_args()

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    predict=ast.literal_eval(args.predict)

    frame_out=ast.literal_eval(args.frame_out)
    time_step = args.time_step
    step_mode = args.step_mode
    iScene = int(args.i_scene)
    model_fnm = args.model
    address = ('0.0.0.0', 9090)
    go(model_fnm, address, iScene)

