#include <ros.h>
#include <std_msgs/Int16.h>
#include <std_msgs/Float32.h>
#include <fred_rl/fred_msg.h>
#include <Encoder.h>
#include <Adafruit_MAX31865.h>

ros::NodeHandle  nh;
fred_rl::fred_msg fred_msg;
ros::Publisher pub("freddie", &fred_msg);

Adafruit_MAX31865 maxx = Adafruit_MAX31865(31, 0, 1, 32);
#define RREF      430.0
#define RNOMINAL  100.0

int period = 250;
unsigned long time_now = 0;
unsigned long time_now2 = 0;

//SPOOL
const int spl_dir_0 = 2;
const int spl_dir_1 = 3;
const int spl_pwm = 4;
const int spl_enc_0 = 5;
const int spl_enc_1 = 6;
Encoder myEnc(spl_enc_0,spl_enc_1);
long oldPosition = 0;
//LIMITER
const int lim_r = 7;
const int lim_l = 8;
//STAGE
const int stg_stp = 9;
const int stg_dir = 11;
//EXTRUDER
//const int ext_stp = 14;
//const int ext_dir = 12;
//HEATER
const int htr_pwm = 29;

float err = 0;
float err_ = 0;
float err_mavg = 0;
float err_mavg_ = 0;
float erri = 0;
float errd = 0;
int temp_set = 0;
int pwm_inp = 0;

const int stack_size = 20;
float temp_stack[stack_size];
int temp_index = 0;
float temp_sum = 0;
float temp_mavg = 0;

void motor_ctl(const std_msgs::Int16& cmd_msg){
  analogWrite(spl_pwm, cmd_msg.data);
  digitalWrite(LED_BUILTIN, HIGH-digitalRead(LED_BUILTIN));
}

void stage_frq(const std_msgs::Int16& cmd_msg){
  analogWriteFrequency(stg_stp, cmd_msg.data/10.0);
}

//void extruder_frq(const std_msgs::Int16& cmd_msg){
//  analogWriteFrequency(ext_stp, cmd_msg.data/10.0);
//}

//void heater_pwm(const std_msgs::Int16& cmd_msg){
//  analogWrite(htr_pwm, cmd_msg.data);
//}

void temp_ctrl(const std_msgs::Int16& cmd_msg){
  erri = 0;
  temp_set = cmd_msg.data;
  fred_msg.temp_set = temp_set;
}

int PID_ctrl(float Kp, float Ki, float Kd, int setpoint, float actual, float mavg){
  err = setpoint - actual;
  err_mavg = setpoint - mavg;
  errd = err_mavg - err_mavg_;
  if (abs(err) < 2){erri = erri + err;}
  float inp = Kp * err + Kd * errd + Ki * erri;
  err_ = err;
  err_mavg_ = err_mavg;
  return (int)inp;
}

ros::Subscriber<std_msgs::Int16> sub0("motor", motor_ctl);
ros::Subscriber<std_msgs::Int16> sub1("stg_frq", stage_frq);
//ros::Subscriber<std_msgs::Int16> sub2("ext_frq", extruder_frq);
ros::Subscriber<std_msgs::Int16> sub3("temp_set", temp_ctrl);

void setup() {  
  nh.initNode();
  nh.advertise(pub);
  nh.subscribe(sub0);
  nh.subscribe(sub1);
  //nh.subscribe(sub2);
  nh.subscribe(sub3);
  pinMode(spl_dir_0, OUTPUT);
  pinMode(spl_dir_1, OUTPUT);
  pinMode(spl_pwm, OUTPUT);
  pinMode(lim_r, INPUT_PULLDOWN);
  pinMode(lim_l, INPUT_PULLDOWN);
  pinMode(stg_stp, OUTPUT);
  pinMode(stg_dir, OUTPUT);
  //pinMode(ext_stp, OUTPUT);
  //pinMode(ext_dir, OUTPUT);
  pinMode(htr_pwm, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(stg_dir, HIGH);
  //digitalWrite(ext_dir, LOW);
  analogWrite(htr_pwm, 0);
  analogWrite(stg_stp, 128);
  //analogWrite(ext_stp, 100);
  analogWriteFrequency(stg_stp,0); //FTM0: 5,6,9,10,20,21,22,23
  //analogWriteFrequency(ext_stp,0); //FTM3: 2,7,8,14,35,36,37,38
  Serial.begin(115200);
  Serial.begin(9600);
  Serial5.begin(9600);
  maxx.begin(MAX31865_3WIRE);
  for (int thisReading = 0; thisReading < stack_size; thisReading++){temp_stack[thisReading] = 0;}
  digitalWrite(spl_dir_0, LOW);
  digitalWrite(spl_dir_1, HIGH);
}

void loop() { 
  while(millis() < time_now + period){
        //wait
  }
  long newPosition = myEnc.read();
  float ds = newPosition-oldPosition;
  float dt = millis() - time_now;
  Serial.println(dt);
  fred_msg.encv = int(ds/dt*100);
  time_now = millis();
   
  //fred_msg.encv = vel;
  oldPosition = newPosition;
  //Serial.println(newPosition);

  bool sw_r = digitalRead(lim_r);
  bool sw_l = digitalRead(lim_l);
  if (sw_r) {digitalWrite(stg_dir, HIGH);}
  if (sw_l) {digitalWrite(stg_dir, LOW);}

  unsigned long incomingByte;
  
  Serial5.write(0x53);
  Serial5.write(0x52);
  Serial5.write(0x2C);
  Serial5.write(0x30);
  Serial5.write(0x30);
  Serial5.write(0x2C);
  Serial5.write(0x30);
  Serial5.write(0x33);
  Serial5.write(0x37);
  Serial5.write(0x0D);
  Serial5.write(0x0A);
  char dia[7]; int ii = 0;
  while (Serial5.available() > 0 ) {
  incomingByte = Serial5.read();
  if (ii > 9 && ii < 17){dia[ii-10] = char(incomingByte);} ii++;
  //Serial.print(char(incomingByte));
  }
  //Serial.println(dia);
  float diam = atof(dia);
  if (diam < 0) {diam=0;}
  //Serial.println(diam);
  fred_msg.diam = 1000 * diam;
  //fred_msg.diam = fred_msg.encv;
  
  float temp = maxx.temperature(RNOMINAL, RREF);
  fred_msg.temp = temp;
  Serial.println(temp);
  temp_sum = temp_sum - temp_stack[temp_index];
  temp_stack[temp_index] = temp;
  temp_sum = temp_sum + temp_stack[temp_index];
  temp_index = temp_index + 1;
  if (temp_index >= stack_size){temp_index = 0;}
  temp_mavg = temp_sum / stack_size;
  fred_msg.temp_ma = temp_mavg;
  //Serial.println(temp_mavg);

  pwm_inp = PID_ctrl(50, 0.1, 100, temp_set, temp, temp_mavg);
  if (pwm_inp > 255) {pwm_inp = 255;}
  if (pwm_inp < 0) {pwm_inp = 0;}
  analogWrite(htr_pwm, pwm_inp);
  fred_msg.htr_pwm = pwm_inp;
  
  pub.publish( &fred_msg );
  nh.spinOnce();
  
}
