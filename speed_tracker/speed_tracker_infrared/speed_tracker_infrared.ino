#include <ros.h>
#include <std_msgs/Float32.h>

// PIN declarations
#define IR1 2 // first IR pin
#define IR2 3 // second IR pin
#define LED_PIN 52 // led pin, lights up when first sensor active, down when second
#define DISTANCE 15 // disntace in cm between sensors

//velocity 
float velocity;

//helpers
long time_first;
long time_second;
float delta;
bool passed = false;

//rosserial setup
ros::NodeHandle nh;
std_msgs::Float32 velocity_msg;
char val_str[2];

ros::Publisher pub_velocity("/speed_tracker/velocity", &velocity_msg);

void  setup()

{
  nh.initNode();
  nh.advertise(pub_velocity);
  
  pinMode(IR1, INPUT_PULLUP);
  pinMode(IR2, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  attachInterrupt(digitalPinToInterrupt(IR1), infrared_1, FALLING);
  attachInterrupt(digitalPinToInterrupt(IR2), infrared_2, FALLING);
}

void loop()
{
  int val = digitalRead(IR1);
  String(val).toCharArray(val_str,2);
  //nh.loginfo(val_str);

  nh.spinOnce();
  delay(50);
}

void infrared_1()
{
  if(!passed){
    time_first = millis();
    digitalWrite(LED_PIN, HIGH);
    passed = true;
    }
}

void infrared_2()
{
  if(passed){
    time_second = millis();
    Serial.println("IR2 TRIGG");
    delta = time_second - time_first;
    velocity = (DISTANCE/delta)*10; // in m/s
    velocity_msg.data = velocity;
    delay(30);
    digitalWrite(LED_PIN, LOW);
    pub_velocity.publish(&velocity_msg);
    passed = false;
  }
}




 
