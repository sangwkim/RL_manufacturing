
#include <ros.h>
#include <std_msgs/Float32.h>

ros::NodeHandle nh;

float time_now = 0;
float segment = 1000;
float ph = 0;
float a = segment/5000.0;
float ext_frq = 1;

const int ext_stp = 14;
const int ext_dir = 12;

void extruder_frq(const std_msgs::Float32& cmd_msg){
  ext_frq = cmd_msg.data;
}

ros::Subscriber<std_msgs::Float32> sub0("ext_frq", extruder_frq);

void setup() {
  nh.initNode();
  nh.subscribe(sub0);
  pinMode(ext_stp, OUTPUT);
  pinMode(ext_dir, OUTPUT);
  digitalWrite(ext_stp, LOW);
  digitalWrite(ext_dir, LOW);
  
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  nh.spinOnce();
  ph = ph + a * ext_frq;
  Serial.println(ph);
  if (ph >= 100){
    digitalWrite(ext_stp, HIGH-digitalRead(ext_stp));
    digitalWrite(LED_BUILTIN, HIGH-digitalRead(LED_BUILTIN));
    ph = 0;
  }
  while(micros() < time_now + segment){
        //wait
  }
  //float dt = micros() - time_now;
  //Serial.println(dt);
  time_now = micros();
  
  //digitalWrite(LED_BUILTIN, HIGH-digitalRead(LED_BUILTIN));
}
