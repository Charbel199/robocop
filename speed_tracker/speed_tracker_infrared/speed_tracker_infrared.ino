#include <ros.h>
#include <std_msgs/Float32.h>

// PIN declarations
#define IR1 A0 // first IR pin
#define IR2 A1 // second IR pin
#define LED_PIN 52 // led pin, lights up when first sensor active, down when second
#define DISTANCE 8 // disntace in cm between sensors
#define VOLTAGE_THRESHOLD 1.4 // voltage threshold that must be exceeded to consider that the IR sensor detected an object

//velocity 
float velocity;

//helpers
long time_first;
long time_second;
float delta;
bool passed = false;
bool obj_passing = false;
float IR1_analog;
float IR2_analog;

//rosserial setup
ros::NodeHandle nh;
std_msgs::Float32 velocity_msg;
char val_str[10];

ros::Publisher pub_velocity("/speed_tracker/velocity", &velocity_msg);

void  setup()

{
  nh.initNode();
  nh.advertise(pub_velocity);
  
  pinMode(LED_PIN, OUTPUT);
}

void loop()
{
  float val = analogRead(IR1) * 5.0 / 1024;
  String(val).toCharArray(val_str,10);
  //nh.loginfo(val_str);
  IR1_analog = analogRead(IR1) * 5.0 / 1024;
  IR2_analog = analogRead(IR2) * 5.0 / 1024;

  if(IR1_analog > VOLTAGE_THRESHOLD && IR2_analog > VOLTAGE_THRESHOLD && !obj_passing){
    obj_passing = true;
    nh.loginfo("OBJ PASSING");
    digitalWrite(LED_PIN, HIGH);
      
    }
  
  if(IR1_analog < VOLTAGE_THRESHOLD && obj_passing && !passed){
      nh.loginfo("IR1 TRIGG");
      time_first = millis();
      digitalWrite(LED_PIN, LOW);
      passed = true;
    }
    
  if(IR2_analog < VOLTAGE_THRESHOLD && passed && obj_passing){
      nh.loginfo("IR2 TRIGG");
      time_second = millis();
      delta = time_second - time_first;
      velocity = (DISTANCE/delta)*1000; // in cm/s
      velocity_msg.data = velocity;
      pub_velocity.publish(&velocity_msg);
      passed = false;
      obj_passing = false;
    }

  
  nh.spinOnce();
  delay(5);
}




 
