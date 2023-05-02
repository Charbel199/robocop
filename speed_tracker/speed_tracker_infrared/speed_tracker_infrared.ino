#include <ros.h>
#include <SharpIR.h>
#include <std_msgs/Float32.h>

// PIN declarations
#define IR1_pin A0 // first IR pin
#define IR2_pin A1 // second IR pin
#define LED_PIN 52 // led pin, lights up when first sensor active, down when second
#define DISTANCE 10 // disntace in cm between sensors
#define DISTANCE_THRESHOLD 30 // voltage threshold that must be exceeded to consider that the IR sensor detected an object
#define model 1080


/* Model :
  GP2Y0A02YK0F --> 20150
  GP2Y0A21YK0F --> 1080
  GP2Y0A710K0F --> 100500
  GP2YA41SK0F --> 430
*/

//Sharp sensor definition
SharpIR IR1 = SharpIR(IR1_pin, model);
SharpIR IR2 = SharpIR(IR2_pin, model);

//velocity 
float velocity;

//helpers
long time_first;
long time_second;
float delta;
bool passed = false;
bool obj_passing = false;
float IR1_distance;
float IR2_distance;

//rosserial setup
ros::NodeHandle nh;
std_msgs::Float32 velocity_msg;
char val_str[10];
char val_str2[10];

ros::Publisher pub_velocity("/speed_tracker/velocity", &velocity_msg);

void  setup()

{
  nh.initNode();
  nh.advertise(pub_velocity);
  
  pinMode(LED_PIN, OUTPUT);
}

void loop()
{
  
  //nh.loginfo(val_str);
  IR1_distance = IR1.distance();
  IR2_distance = IR2.distance();
  String(IR1_distance).toCharArray(val_str,10);
  String(IR2_distance).toCharArray(val_str2,10);
  //nh.loginfo("IR1: ");
  //nh.loginfo(val_str);
  //nh.loginfo("IR2: ");
  //nh.loginfo(val_str2);

  if(IR1_distance < DISTANCE_THRESHOLD && IR2_distance < DISTANCE_THRESHOLD && !obj_passing){
    obj_passing = true;
    nh.loginfo("OBJ PASSING");
    digitalWrite(LED_PIN, HIGH);
      
    }
  
  if(IR1_distance > DISTANCE_THRESHOLD && obj_passing && !passed){
      nh.loginfo("IR1 TRIGG");
      time_first = millis();
      digitalWrite(LED_PIN, LOW);
      passed = true;
    }
    
  if(IR2_distance > DISTANCE_THRESHOLD && passed && obj_passing){
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




 
