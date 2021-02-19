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

gp_list = [2.10E-04, 1.63E-04  ,  3.79E-05  ,  9.29E-05 ,   2.58E-04  ,  1.67E-04  ,  1.23E-04  ,  5.17E-05   , 2.43E-04  ,  2.06E-04
]
gi_list = [2.60E-05, 1.54E-05  ,  1.79E-05  ,  1.36E-05  ,  5.60E-06   , 1.74E-05   , 1.16E-05  ,  1.69E-05    ,1.30E-05  ,  1.78E-05
]

particle_num = np.arange(0,10)
np.random.shuffle(particle_num)

date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M.txt")
f = open(date_time, "w+")
f.write(target_info + "\r\n")
f.write(np.array_str(particle_num) + "\r\n")
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
        t_points = np.linspace(0,(max_steps+window-1)*timestep/1000,(max_steps+window)/120)
        s_points = np.random.randint(550,551,size=t_points.shape)
        #s_points[:15] = [450,450,550,350,550,400,500,350,500,400,550,350,500,400,550]
        #s_points[:50] = [450,450,550,350,450,450,450,550,350,450,450,450,550,350,450,450,450,550,350,450,450,450,550,350,450,450,450,550,350,450,450,450,550,350,450,450,450,550,350,450,450,450,550,350,450,450,450,550,350,450]
        trajectory = np.hstack([np.tile(s_point,120) for s_point in s_points])
    elif target_mode == 2:
        t = np.linspace(0,(max_steps+window-1)*timestep/1000,max_steps+window)
        omega = 0.2
        ampl = 100.
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
err_diam_intgl = 0
err_vel_intgl = 0
err_vel_prev = 0
#ext_f = 20
ext_f = 10

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
    global o, h, a0, step, step_i, spool, switch, err_diam_intgl, err_vel_intgl, err_vel_prev # declare global variables

    ext_speed = 9.45*2*np.pi*ext_f/200/16
    model_vel = 2 * ext_speed * 7.112**2 / (float(trajectory[step])/1000)**2 / 20 / 2 / np.pi
    
    #gp = 0.00015
    #gi = 0.000015
    #if step%600==0: err_diam_intgl = 0
    #gp = gp_list[particle_num[int(step/600)]]
    #gi = gi_list[particle_num[int(step/600)]]
    gp = 1.67E-04
    gi = 1.26E-05
    
    err_diam = data.diam - trajectory[step]
    err_diam_intgl = err_diam_intgl + err_diam
    
    pi = gp * err_diam + gi * err_diam_intgl
    
    com_vel = model_vel + pi
    
    vp = 35
    vi = 20
    vd = 0

    err_vel = float(data.encv)/840 - com_vel #############################################open loop -> model_vel##### closed loop -> com_vel
    err_vel_intgl = err_vel_intgl + err_vel
    err_vel_diff = err_vel - err_vel_prev

    vpi = - vp * err_vel - vi * err_vel_intgl - vd * err_vel_diff
    
    err_vel_prev = err_vel

    a0_ = np.clip(vpi,15,256)

    # publish messages    
    pub.publish(a0_)
    pub2.publish(trajectory[step])
    pub3.publish(0)
    pub4.publish(ext_f)
    pub5.publish(0)
    a0 = a0_
    
    step = step + 1
    
    # data logging
    data_str = "closed, step: {0:6d}, spool: {1:.2f}, diameter: {2:.2f}, target_diam: {3:.2f}, speed: {4:.2f}, model_speed: {5:.2f}, com_speed: {6:.2f}, duty_cycle: {7:.2f}, temp: {8:.2f}".format(step, spool/6.5, data.diam, trajectory[step], float(data.encv)/840, model_vel, com_vel, a0, data.temp)
    print(data_str, dt)
    with open(date_time, "a+") as file:
        file.write(data_str + "\r\n")
        
    spool = spool + 0.01*ext_f
    
    if step==2100:
    #if spool > 650.:
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
