#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$
 
## Simple talker demo that listens to std_msgs/Strings published 
## to the 'chatter' topic
 
import rospy
from std_msgs.msg import Int16
from std_msgs.msg import Float32
from std_msgs.msg import String
from fred_rl.msg import fred_msg
import numpy as np
from scipy import interpolate
import pickle
import datetime
import gc
import time
import matlab.engine
from scipy.io import loadmat

gc.collect()
tt = datetime.datetime.now()
# publishers
pub = rospy.Publisher('motor', Int16, queue_size=10)
pub2 = rospy.Publisher('target', Float32, queue_size=10)
pub3 = rospy.Publisher('greedy_motor', Int16, queue_size=10)
pub4 = rospy.Publisher('ext_frq', Float32, queue_size=10)
pub5 = rospy.Publisher('greedy_ext', Float32, queue_size=10)

target_mode = 5 # target_mode/ 0: unexpected step, 1: expected step, 2: expected sinusodial, 3: expected random spline
target_reuse = False
exploit_learn = False
if target_mode == 0: target_info = "unexpected step"
elif target_mode == 1: target_info = "expected step"
elif target_mode == 2: target_info = "expected sinusodial"
elif target_mode == 3: target_info = "expected random spline"
elif target_mode == 4: target_info = "frequency sweep"
elif target_mode == 5: target_info = "chirp step"

date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M.txt")
f = open(date_time, "w+")
f.write(target_info + "\r\n")
f.close()

switch = -1 # 0: explore, 1: exploit, 2: pretrain (changes as a subscriber subscribes a message)

window = 51

learn_start_steps = 100 # the stepcount when starts learning
max_steps = 1000000 # maximum step, 72000 steps = 1 hour (when timestep = 50 ms)
timestep = 250 # time per each step in ms

### generate target trajectory ###
if target_reuse:
    fileObject = open("trajectory",'r')
    trajectory = pickle.load(fileObject)
    fileObject.close()
else:
    if target_mode == 0 or target_mode == 1:
        t_points = np.linspace(0,(max_steps+window-1)*timestep/1000,(max_steps+window)/200)
        s_points = np.random.randint(300,600,size=t_points.shape)
        s_points[:15] = [450,450,550,350,550,400,500,350,500,400,550,350,500,400,550]
        trajectory = np.hstack([np.tile(s_point,200) for s_point in s_points])
    elif target_mode == 2:
        t = np.linspace(0,(max_steps+window-1)*timestep/1000,max_steps+window)
        omega = 0.25
        ampl = 125.
        offset = 450.
        trajectory = ampl*np.sin(t*omega) + offset
    elif target_mode == 3:
        t_points = np.linspace(0,(max_steps+window-1)*timestep/1000,(max_steps+window)/200)
        s_points = np.random.randint(350,550,size=t_points.shape)
        s_points[:15] = [450,450,550,350,550,400,500,350,500,400,550,350,500,400,550]
        spl = interpolate.splrep(t_points, s_points)
        t = np.linspace(0,(max_steps+window-1)*0.25,max_steps+window)
        trajectory = interpolate.splev(t, spl)
    elif target_mode == 4:
    	t = np.linspace(0,(max_steps+window-1)*timestep/1000,max_steps+window)
    	f_low = 0.01
    	f_high = 0.08
    	chirpyness = (f_high-f_low)/700.
    	ampl = 50.#100.
    	offset = 450.
    	phi = 2*np.pi*(chirpyness/2*np.power(t,2)+f_low*t)
    	trajectory = ampl*np.sin(phi) + offset
    elif target_mode == 5:
        long_interval = 200
        decay = 0.89
        num_intervals = 25
        trajectory = [np.tile(450,400)]
        for i in range(num_intervals):
            if i % 2 == 0:
                trajectory.append(np.tile(550,int(long_interval*decay**i)))
            else:
                trajectory.append(np.tile(350,int(long_interval*decay**i)))
        trajectory = np.hstack(trajectory)
        
        
        
step = 0 # global variable for the step
step_i = 0
spool = 0.
ema = 0.
a0 = 100.0 # global variable for the action
a1 = .0
err_vel_intgl = 0
err_vel_prev = 0
ext_f = 25
com_vel = 1.4

########################## build QDMC Controller ##############################

eng = matlab.engine.start_matlab()

dt = 0.25
n_sr = 100;  # Length of the step response model

Su_sp = loadmat('mean_spool_step.mat')['data']
y_0 = Su_sp[0]
Su_sp = (Su_sp[:100] - Su_sp[0])/100.
Su_sp = np.transpose(Su_sp)
Su_sp = np.expand_dims(Su_sp, 0)

Su_ext = loadmat('mean_extruder_step.mat')['data']
Su_ext = (Su_ext[:100] - Su_ext[0])/100.
Su_ext = np.transpose(Su_ext)
Su_ext = np.expand_dims(Su_ext, 0)

Su = np.concatenate((Su_sp,Su_ext),axis=1)
Su = matlab.double(Su.tolist())

ysp = np.expand_dims(trajectory, 0)

n_in = matlab.double([[2]]);       # Number of inputs
n_out = matlab.double([[1]]);      # Number of outputs
n_dis = matlab.double([[0]]);      # Number of measured disturbances

Sd = matlab.double([[]]);        # Step resposne of measured disturbances
p = matlab.double([[50]]);        # Prediction horizon (number of time steps)
c = matlab.double([[25]]);         # Control horizon (number of time steps)

La = matlab.double([[1e5],[1e5]]);         # Weight for input movement
Q = matlab.double([[25]]);         # Weight for output error
ctg = matlab.double([[]]);       # Cost to go

u_past = matlab.double([[0.1],[0.9]]);     # Past input (assume process starts at steady state)
y_past = matlab.double([[0]]);     # Past output (assume process starts at steady state)
d_past = matlab.double([[]]);    # Past measured disturbance
u_int = matlab.double([[]]);     # Integrating inputs
d_int = matlab.double([[]]);     # Integrating measured disturbances

u_min = matlab.double([[0],[0]]);     # Minimum input
u_max = matlab.double([[1],[1]]);    # Maximum input
D_u_min = matlab.double([[-100],[-100]]); # Minimum input movement
D_u_max = matlab.double([[100], [100]]);  # Maximum input movement
y_min = matlab.double([[-100]]);   # Minimum output
y_max = matlab.double([[100]]);    # Maximum output

soft = matlab.double([[1,1]]);   # Both, y_max and y_min are soft constraints
w_eps = matlab.double([[10]]);     # Weight for linear constraint violation
W_eps = matlab.double([[10]]);     # Weight for quadratic constraint violation

n_x = matlab.double([[1]]);        # Number of states

Controller = eng.QDMC_controller_soft(n_in,n_out,n_dis,Su,Sd,p,c,La,Q,ctg, u_min,u_max,D_u_min,D_u_max,y_min,y_max, u_past,y_past,d_past, u_int,d_int,soft,w_eps,W_eps)

###############################################################################

print("ready")

def host():
    
    global switch
    rospy.init_node('host', anonymous=True)
    rospy.Subscriber('mode', String, mode_switch)
    rospy.Subscriber('freddie', fred_msg, callback)
 
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

def callback(data):
           
    if switch == 0: 
        callback_closed_loop(data)

def callback_closed_loop(data):        
    global tt
    dt = datetime.datetime.now() - tt
    tt = datetime.datetime.now()
    global o, h, a0, step, step_i, spool, switch, err_vel_intgl, err_vel_prev, com_vel, ext_f # declare global variables

    ysp_curr = ysp[:,step:min(step+100,ysp.shape[-1])];
    ysp_curr = np.concatenate((ysp_curr, ysp_curr[0,-1]+np.zeros([1,100-ysp_curr.shape[-1]])),1)
    ysp_curr = (ysp_curr-y_0[0])/100.
    
    y = matlab.double([[(data.diam-y_0[0])/100.]])
    d = matlab.double([[]])
    
    com_u = eng.run_controller(Controller, y, matlab.double(ysp_curr.tolist()), d)
    b = 1 / np.sqrt(1.4)
    a = 1 / np.sqrt(0.2) - b
    com_vel = 1 / (a*com_u[0][0] + b)**2
    d = np.sqrt(5)
    c = np.sqrt(30)-d
    ext_f = (c*com_u[1][0] + d)**2

    vp = 35
    vi = 20
    vd = 0

    err_vel = float(data.encv)/840 - com_vel #############################################open loop -> model_vel##### closed loop -> com_vel
    err_vel_intgl = err_vel_intgl + err_vel
    err_vel_diff = err_vel - err_vel_prev

    vpi = - vp * err_vel - vi * err_vel_intgl - vd * err_vel_diff
    
    err_vel_prev = err_vel

    a0_ = np.clip(vpi,0,256)

    # publish messages    
    pub.publish(a0_)
    pub2.publish(trajectory[step])
    pub3.publish(0)
    pub4.publish(ext_f)
    pub5.publish(0)
    a0 = a0_
    
    step = step + 1
    
    # data logging
    data_str = "qdmc, step: {0:6d}, spool: {1:.2f}, diameter: {2:.2f}, target_diam: {3:.2f}, ext_f: {4:.2f}, speed: {5:.2f}, com_speed: {6:.2f}, duty_cycle: {7:.2f}, temp: {8:.2f}".format(step, spool/6.5, data.diam, trajectory[step], ext_f, float(data.encv)/840, com_vel, a0, data.temp)
    print(data_str, dt)
    with open(date_time, "a+") as file:
        file.write(data_str + "\r\n")
        
    spool = spool + 0.01*ext_f
    
    if step > 2100:
        pub.publish(0)
        pub4.publish(0)
        switch = -1
        spool = 0
        step_i = step
        print("run over")         
 
def mode_switch(data):
    global switch, step
    print(data.data)
    if data.data == "closed":
        switch = 0
        
if __name__ == '__main__':
    host()
