int sen1=A0;
int sen2=A3;
int ledPin=9;
unsigned long t1=0;
unsigned long t2=0;
float velocity;
float velocity_real;
float timeFirst;
float timeScnd;
float diff;
float speedConst=7.5;  //in cm.

void setup()
{
    Serial.begin(9600);
    pinMode(sen1,INPUT);
    pinMode(sen2,INPUT);

    analogWrite(11,LOW);
    analogWrite(10,HIGH);
}

void loop()
{
    if (analogRead(sen1)<500 && analogRead(sen2)>500)
    {
        timeFirst= millis();
        digitalWrite(ledPin, LOW);
        delay(30);
    }


    if (analogRead(sen2)>500 && analogRead(sen1)<500)
    {
        timeScnd = millis();
        diff = timeScnd - timeFirst;
        velocity = speedConst / diff;
        velocity_real = (velocity*360)/100;     //milliseconds to hours and centimetres to kilometeres.
        delay(30);
        digitalWrite(ledPin, HIGH);
        Serial.print("\the velocity is : ");
        Serial.println(velocity_real);
        Serial.print(" km/hr. ");
        delay(500);
        digitalWrite(ledPin,LOW);
        delay(500);
    }

  /*else if(analogRead(sen2)<500 && analogRead(sen1)<500)       //uncomment
  if you want to write it.
  {
      Serial.print("\
 Error:404/ the object
  is tooo big.");
  }*/
 /* else{Serial.print("\
 error:404/no object detected
  ");}     //uncomment if you want to write it.*/
}