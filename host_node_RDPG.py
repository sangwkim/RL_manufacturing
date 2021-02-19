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
from RDPGmodel import RDPG
import numpy as np
from OU_custom_noise import OrnsteinUhlenbeckProcess as OU
import datetime
from scipy import interpolate
import pickle
import multiprocessing
import gc
import thread as thread
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

# generating a text file for logging the data
date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M.txt")
f = open(date_time, "w+")
f.write(target_info + "\r\n")
f.close()

switch = -1 # 0: explore, 1: exploit, 2: pretrain (changes as a subscriber subscribes a message)

o_dim = 9 # dimension of the state
a_dim = 2 # dimension of the action (voltage input to the motor / frequency input to the extruder stepper)
window = 51
stride = 5

a_bound_low0 = 20 # bound of the action (30~70)
a_bound_high0 = 256
a_bound_low1 = 5
a_bound_high1 = 30

BATCH_SIZE = 32

lr_c = 0.000005
lr_a = 0.000001
RDPGnet = RDPG(o_dim, a_dim, window, BATCH_SIZE, lr_c, lr_a) # RDPG session
#RDPGnet = RDPG(o_dim, a_dim, window, BATCH_SIZE)

learn_start_steps = 100 # the stepcount when starts learning
max_steps = 1000000 # maximum step, 72000 steps = 1 hour (when timestep = 50 ms)
timestep = 250 # time per each step in ms

# generate exploration noise
# (motor)
#vol_noise0 = 20 # volatility of the noise
vol_noise0 = 10 # volatility of the noise
speed0 = 0.1 # attraction to the mean
decay0 = 0.999925 # decay rate of the noise
# (extruder)
#vol_noise1 = 15 # volatility of the noise
vol_noise1 = 10 # volatility of the noise
speed1 = 0.1 # attraction to the mean
decay1 = 0.999925 # decay rate of the noise

vol_noise2 = 1
speed2 = 1
decay2 = 1

expl0 = OU(speed=speed0, mean=0, vol=vol_noise0, t=max_steps*timestep/1000)
expl1 = OU(speed=speed1, mean=0, vol=vol_noise1, t=max_steps*timestep/1000)
expl2 = OU(speed=speed2, mean=0, vol=vol_noise2, t=max_steps*timestep/1000)
expl_noise0 = expl0.sample(max_steps,initial=0,decay=decay0)
expl_noise1 = expl1.sample(max_steps,initial=0,decay=decay1)
expl_noise2 = expl2.sample(max_steps,initial=0,decay=decay2)

### generate target trajectory ###
if target_reuse:
    fileObject = open("trajectory",'r')
    trajectory = pickle.load(fileObject)
    fileObject.close()
else:
    if target_mode == 0 or target_mode == 1:
        t_points = np.linspace(0,(max_steps+window-1)*timestep/1000,(max_steps+window)/120)
        s_points = np.random.randint(300,600,size=t_points.shape)
        #s_points[:15] = [450,450,550,350,550,400,500,350,500,400,550,350,500,400,550]
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

o = np.zeros(o_dim)
step = 0 # global variable for the step
step_i = 0
spool = 0.
ema = 0.
a0 = .0 # global variable for the action
a1 = .0
h = []

h_test = np.load("h_indices_0813.npy")
hh = 0

#Q_exam = RDPGnet.compute_Q_single(h_test[:,hh,:])[0][0]
#a1_exam = RDPGnet.compute_action_single(h_test[:,hh,:])[0][0]
#a2_exam = RDPGnet.compute_action_single(h_test[:,hh,:])[0][1]
Q_exam = 0
a1_exam = 0
a2_exam = 0


print("ready")

def host():
    
    global switch
    rospy.init_node('host', anonymous=True)
    rospy.Subscriber('mode', String, mode_switch)
    rospy.Subscriber('save_load', String, save_model)
    rospy.Subscriber('freddie', fred_msg, callback)
 
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

def train():
    global Q_exam, a1_exam, a2_exam, hh, step, step_i
    while True:
        if switch==0 and step-step_i>window:
            RDPGnet.train()
            hh = hh+1
            if hh==10: hh=0
            Q_exam = RDPGnet.compute_Q_single(h_test[:,hh,:])[0][0]
            a1_exam = RDPGnet.compute_action_single(h_test[:,hh,:])[0][0]
            a2_exam = RDPGnet.compute_action_single(h_test[:,hh,:])[0][1]
        time.sleep(0.01)

def callback(data):
           
    if switch == 0: 
        callback_explore(data)
    elif switch == 1: callback_exploit(data)
    elif switch == 4: callback_random(data)

def callback_random(data):        
    global tt
    dt = datetime.datetime.now() - tt
    tt = datetime.datetime.now()
    global o, h, a0, a1, step, step_i, spool, switch # declare global variables

    if a0==0: a0 = np.random.rand()*100.
    elif np.random.rand() < 0.01: a0 = np.random.rand()*100.
    if a1==0: a1 = np.random.rand()*100.
    elif np.random.rand() < 0.01: a1 = np.random.rand()*100.
    
    a0 = np.clip(a0+expl_noise2[step], 0, 100)
    a1 = np.clip(a1+expl_noise2[step], 0, 100)
    
    a0_dist = 4.221e-8 * a0**5 - 5.874e-6 * a0**4 + 2.514e-4 * a0**3 - 5.58e-4 * a0**2 + 0.1951 * a0

    a0_scaled_dist = a_bound_low0 + a0_dist/100.*(a_bound_high0-a_bound_low0)
    a1_scaled = a_bound_low1 + a1/100.*(a_bound_high1-a_bound_low1)

    # publish messages    
    pub.publish(int(a0_scaled_dist))
    pub2.publish(0)
    pub3.publish(0)
    pub4.publish(a1_scaled)
    pub5.publish(0)
    
    # store transition and proceed to the next step
    #if step-step_i > window+1:
    #    if np.random.rand() < 0.2:
    #        RDPGnet.add_to_replay(h) 
    
    step = step + 1
    
    """
    if spool*10 > learn_start_steps:
        if step%20==0:
            RDPGnet.train()            
            mpr = multiprocessing.Process(target=RDPGnet.train())
            mpr.start()
    """
    # data logging
    data_str = "exploit, step: {0:6d}, spool: {1:.2f}, diameter: {2:.2f}, speed: {3:.2f}, motorV: {4:.2f}, extfrq: {5:.2f}, target: {6:.2f}, greedy_m: {7:.2f}, greedy_e: {8:.2f}, reward: {9:.3f}, Q_exam: {10:.3f}, am_exam: {11:.3f}, a_exam: {12:.3f}, {13:2d}".format(step, spool/6.5, data.diam/10., float(data.encv)/12.5, a0, a1, 0, 0, 0, 0, Q_exam, a1_exam, a2_exam, hh)
    print(data_str, dt)
    with open(date_time, "a+") as file:
        file.write(data_str + "\r\n")
        
    spool = spool + 0.01*a1_scaled
    
    if spool > 650:
        pub.publish(0)
        pub4.publish(0)
        switch = -1
        spool = 0
        step_i = step
        print("run over")         
       
       
def callback_explore(data):     
    
    global tt
    dt = datetime.datetime.now() - tt
    tt = datetime.datetime.now()
    global o, h, a0, a1, step, step_i, spool, switch # declare global variables

    #o_ = np.hstack((spool/6.50, data.diam/10.00, float(data.encv)/12.50, a0, a1, trajectory[step]/10.00, trajectory[step+10]/10.00, trajectory[step+20]/10.00, trajectory[step+30]/10.00, trajectory[step+40]/10.00, trajectory[step+50]/10.00))
    o_ = np.hstack((spool/6.50, data.diam/10.00, float(data.encv)/12.50, trajectory[step]/10.00, trajectory[step+10]/10.00, trajectory[step+20]/10.00, trajectory[step+30]/10.00, trajectory[step+40]/10.00, trajectory[step+50]/10.00))   
    #o_ = np.hstack((spool/6.50, data.diam/10.00, float(data.encv)/12.50, a0, a1, trajectory[step]/10.00, trajectory[step+10]/10.00, trajectory[step+20]/10.00, trajectory[step+30]/10.00, trajectory[step+40]/10.00, trajectory[step+50]/10.00))
    #r = 1.-(data.diam - trajectory[step])**2*0.0001
    r = 1.-abs(data.diam - trajectory[step])*0.01 + a1/2000.
    
    a_greedy = RDPGnet.evaluate_actor(np.hstack((a0,a1)),o_)
    a0_greedy_dist = 4.221e-8 * a_greedy[0]**5 - 5.874e-6 * a_greedy[0]**4 + 2.514e-4 * a_greedy[0]**3 - 5.58e-4 * a_greedy[0]**2 + 0.1951 * a_greedy[0]
    a0_ = a_greedy[0] + expl_noise0[step]
    a1_ = a_greedy[1] + expl_noise1[step]
    """
    if (a0_temp > 100):
        a0_ = 200 - a0_temp
    elif (a0_temp < 0):
        a0_ = -a0_temp
    else:
        a0_ = a0_temp
    if (a1_temp > 100):
        a1_ = 200 - a1_temp
    elif (a1_temp < 0):
        a1_ = -a1_temp
    else:
        a1_ = a1_temp
    """
    a0_ = np.clip(a0_, 0, 100) # mix exploration noise to the greedy action
    a0_dist = 4.221e-8 * a0_**5 - 5.874e-6 * a0_**4 + 2.514e-4 * a0_**3 - 5.58e-4 * a0_**2 + 0.1951 * a0_
    a1_ = np.clip(a1_, 0, 100) 

    a0_greedy_dist_scaled = a_bound_low0 + a0_greedy_dist/100.*(a_bound_high0-a_bound_low0)
    a1_greedy_scaled = a_bound_low1 + a_greedy[1]/100.*(a_bound_high1-a_bound_low1)
    a0_dist_scaled = a_bound_low0 + a0_dist/100.*(a_bound_high0-a_bound_low0)
    a1_scaled = a_bound_low1 + a1_/100.*(a_bound_high1-a_bound_low1)

    # publish messages    
    pub.publish(int(a0_dist_scaled))
    pub2.publish(trajectory[step])
    pub3.publish(int(a0_greedy_dist_scaled))
    pub4.publish(a1_scaled)
    pub5.publish(a1_greedy_scaled)
    
    RDPGnet.log_history(np.expand_dims(np.hstack((r,o_,a0_,a1_)),0))
    
    o, a0, a1 = o_, a0_, a1_
    step = step + 1
    spool = spool + 0.01*a1_scaled
    """
    if spool*10 > learn_start_steps:
        if step%20==0:
            RDPGnet.train()            
            mpr = multiprocessing.Process(target=RDPGnet.train())
            mpr.start()
    """
    # data logging
    data_str = "explore, step: {0:6d}, spool: {1:.2f}, diameter: {2:.2f}, speed: {3:.2f}, motorV: {4:.2f}, extfrq: {5:.2f}, target: {6:.2f}, greedy_m: {7:.2f}, greedy_e: {8:.2f}, reward: {9:.3f}, Q_exam: {10:.3f}, am_exam: {11:.3f}, a_exam: {12:.3f}, {13:2d}, temp: {14:.3f}".format(step, o[0], o[1], o[2], a0, a1, o[3], a_greedy[0], a_greedy[1], r, Q_exam, a1_exam, a2_exam, hh, data.temp)
    print(data_str, dt)
    with open(date_time, "a+") as file:
        file.write(data_str + "\r\n")
        
    if spool > 650:
        pub.publish(0)
        pub4.publish(0)
        switch = -1
        RDPGnet.save_session(date_time[:-4], step)
        RDPGnet.state_indice[:,:,:] = 0
        RDPGnet.action_indice[:,:,:] = 0
        spool = 0
        step_i = step
        print("saved the session")   
 
def callback_exploit(data):
    
    global tt
    dt = datetime.datetime.now() - tt
    tt = datetime.datetime.now()
    global o, h, a0, a1, step, step_i, spool, switch # declare global variables

    #o_ = np.hstack((spool/6.50, data.diam/10.00, float(data.encv)/12.50, a0, a1, trajectory[step]/10.00, trajectory[step+10]/10.00, trajectory[step+20]/10.00, trajectory[step+30]/10.00, trajectory[step+40]/10.00, trajectory[step+50]/10.00))
    o_ = np.hstack((spool/6.50, data.diam/10.00, float(data.encv)/12.50, trajectory[step]/10.00, trajectory[step+10]/10.00, trajectory[step+20]/10.00, trajectory[step+30]/10.00, trajectory[step+40]/10.00, trajectory[step+50]/10.00))   
    #o_ = np.hstack((spool/6.50, data.diam/10.00, float(data.encv)/12.50, a0, a1, trajectory[step]/10.00, trajectory[step+10]/10.00, trajectory[step+20]/10.00, trajectory[step+30]/10.00, trajectory[step+40]/10.00, trajectory[step+50]/10.00))
    #r = 1.-(data.diam - trajectory[step])**2*0.0001
    r = 1.-abs(data.diam - trajectory[step])*0.01 + a1/2000.
    
    a_greedy = RDPGnet.evaluate_actor(np.hstack((a0,a1)),o_)
    a0_greedy_dist = 4.221e-8 * a_greedy[0]**5 - 5.874e-6 * a_greedy[0]**4 + 2.514e-4 * a_greedy[0]**3 - 5.58e-4 * a_greedy[0]**2 + 0.1951 * a_greedy[0]
    #a0_greedy_dist = 5.359e-8 * a_greedy[0]**5 - 9.618e-6 * a_greedy[0]**4 + 7.026e-4 * a_greedy[0]**3 - 0.0234 * a_greedy[0]**2 + 0.5719 * a_greedy[0]
    a0_ = np.clip(a_greedy[0], 0, 100) # mix exploration noise to the greedy action
    a0_dist = 4.221e-8 * a0_**5 - 5.874e-6 * a0_**4 + 2.514e-4 * a0_**3 - 5.58e-4 * a0_**2 + 0.1951 * a0_
    #a0_dist = 5.359e-8 * a0_**5 - 9.618e-6 * a0_**4 + 7.026e-4 * a0_**3 - 0.0234 * a0_**2 + 0.5719 * a0_
    a1_ = np.clip(a_greedy[1], 0, 100) 
    #if spool==0: h = np.hstack([o, a0, a1, r])
    #h = np.append(h, np.hstack([o, a0, a1, r]),axis=0)
    #if h.shape[0] > window: h = np.delete(h,0,0)

    a0_greedy_dist_scaled = a_bound_low0 + a0_greedy_dist/100.*(a_bound_high0-a_bound_low0)
    a1_greedy_scaled = a_bound_low1 + a_greedy[1]/100.*(a_bound_high1-a_bound_low1)
    a0_dist_scaled = a_bound_low0 + a0_dist/100.*(a_bound_high0-a_bound_low0)
    a1_scaled = a_bound_low1 + a1_/100.*(a_bound_high1-a_bound_low1)

    # publish messages    
    pub.publish(int(a0_dist_scaled))
    pub2.publish(trajectory[step])
    pub3.publish(int(a0_greedy_dist_scaled))
    pub4.publish(a1_scaled)
    pub5.publish(a1_greedy_scaled)
    
    RDPGnet.log_history(np.expand_dims(np.hstack((r,o_,a0_,a1_)),0))
    
    o, a0, a1 = o_, a0_, a1_
    step = step + 1
    spool = spool + 0.01*a1_scaled
    """
    if spool*10 > learn_start_steps:
        if step%20==0:
            RDPGnet.train()            
            mpr = multiprocessing.Process(target=RDPGnet.train())
            mpr.start()
    """
    # data logging
    data_str = "exploit, step: {0:6d}, spool: {1:.2f}, diameter: {2:.2f}, speed: {3:.2f}, motorV: {4:.2f}, extfrq: {5:.2f}, target: {6:.2f}, greedy_m: {7:.2f}, greedy_e: {8:.2f}, reward: {9:.3f}, Q_exam: {10:.3f}, am_exam: {11:.3f}, a_exam: {12:.3f}, {13:2d}, temp: {14:.3f}".format(step, o[0], o[1], o[2], a0, a1, o[3], a_greedy[0], a_greedy[1], r, Q_exam, a1_exam, a2_exam, hh, data.temp)
    print(data_str, dt)
    with open(date_time, "a+") as file:
        file.write(data_str + "\r\n")
        
    if step == 2050:#spool > 650:
        pub.publish(0)
        pub4.publish(0)
        switch = -1
        #RDPGnet.save_session(date_time[:-4], step)
        RDPGnet.state_indice[:,:,:] = 0
        RDPGnet.action_indice[:,:,:] = 0
        spool = 0
        step_i = step
        print("run over")  
 
def mode_switch(data):
    global switch, step
    print(data.data)
    if data.data == "explore":
        switch = 0
    elif data.data == "exploit":
        switch = 1
        step = 0
    elif data.data == "precritic":
        switch = 2
    elif data.data == "PID":
        switch = 3
    elif data.data == "random":
        switch = 4

def save_model(msg):
    global switch, step, spool, Q_exam, a1_exam, a2_exam
    switch = -1
    data = msg.data
    if data[:4] == "save":
        RDPGnet.save_session(date_time[:-4], step)
        spool = 0
        print("saved the session")
    elif data[:4] == "load":
        date_time_load = data[5:9] + '_' + data[10:12] + '_' + data[13:15] + '_' + data[16:18] + '_' + data[19:21]
        step = int(data[22:])
        RDPGnet.restore_session(date_time_load, step)
        print("loaded the session")
        #Q_exam = RDPGnet.compute_Q(h_test[0])[0]
        #a1_exam = RDPGnet.compute_action(h_test[0])[0]
        #a2_exam = RDPGnet.compute_action(h_test[0])[1]
         
if __name__ == '__main__':
    thread.start_new_thread(train,())
    host()
