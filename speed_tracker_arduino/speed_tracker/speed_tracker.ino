
// PIN declarations
const int IR1=2; // first IR pin
const int IR2=3; // second IR pin
int LED_PIN=52; // led pin, lights up when first sensor active, down when second
float DISTANCE=15; // disntace in cm between sensors

// timers 
unsigned  long t1=0;
unsigned long t2=0; 

//velocity 
float velocity;

//helpers
float time_first;
float time_second;
float delta;
bool passed = false;

void  setup()
{
  Serial.begin(9600);
  pinMode(IR1, INPUT_PULLUP);
  pinMode(IR2, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  attachInterrupt(digitalPinToInterrupt(IR1), infrared_1, FALLING);
  attachInterrupt(digitalPinToInterrupt(IR2), infrared_2, FALLING);
}

void loop()
{
  int val = digitalRead(IR1);
}

void infrared_1()
{
  if(!passed){
    time_first = millis();
    Serial.println("IR1 TRIGG");
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
    delay(30);
    digitalWrite(LED_PIN, LOW);
    Serial.print("The velocity is: ");
    Serial.print(velocity);
    Serial.println(" m/s.");
    passed = false;
  }
}




 
