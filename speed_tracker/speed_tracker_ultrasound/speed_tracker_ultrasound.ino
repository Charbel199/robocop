#include <ros.h>
#include <std_msgs/Float32.h>

// PIN declarations
#define ECHO1 2 // first ultrasound echo pin
#define ECHO2 3 // second ultrasound echo pin
#define PING_PIN1 50 // first ultrasound ping pin 
#define PING_PIN2 51 // second ultrasound ping pin
#define LED_PIN 52 // led pin, lights up when first sensor active, down when second

// constants
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
  
  pinMode(ECHO1, INPUT_PULLUP);
  pinMode(ECHO2, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  pinMode(PING_PIN1, OUTPUT);
  pinMode(PING_PIN2, OUTPUT);
  attachInterrupt(digitalPinToInterrupt(ECHO1), echo_1, RISING);
  attachInterrupt(digitalPinToInterrupt(ECHO2), echo_2, RISING);
}

void loop()
{
  int val = digitalRead(ECHO1);
  String(val).toCharArray(val_str,2);
  nh.loginfo(val_str);

  // pulse the ping pins constantly
  digitalWrite(PING_PIN1, LOW);
  digitalWrite(PING_PIN2, LOW);
  delayMicroseconds(2);
  digitalWrite(PING_PIN1, HIGH);
  digitalWrite(PING_PIN2, HIGH);
  delayMicroseconds(10);
  digitalWrite(PING_PIN1, LOW);
  digitalWrite(PING_PIN2, LOW);
  
  nh.spinOnce();
  delay(50);
}

void echo_1()
{
  if(!passed){
    time_first = millis();
    digitalWrite(LED_PIN, HIGH);
    passed = true;
    }
}

void echo_2()
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




 
