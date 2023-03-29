# Robocop
This project is aimed at developing an autonomous rover equipped with sirens, speakers,
and cameras to monitor traffic violations and enforce traffic rules. The rover is designed
to detect speeding cars and automatically follow them until they stop. The rover (robocop) is also able to capture
the speeding car's license plate and fetch the driver's information.

## Hardware List
The following hardware components are used in this project:

- Rover
- Depth camera (+ RGB camera ?)
- Speaker
- Siren lights
- 2 * IR sensors
- Arduino
- Raspberry Pi 4
- Power Bank
- SD Card
- SD Card reader (And adapter if needed)


## Fuzzy Logic Controller
The autonomous rover uses a fuzzy logic controller to determine its behavior when following a speeding car.
The fuzzy logic controller takes inputs from the depth and RGB cameras to determine the appropriate actions to take.

The fuzzy logic system includes input variables such as the distance between the rover and the speeding car, 
and the direction of the speeding car. These input variables are fuzzified
into fuzzy sets, and the fuzzy rules are defined to determine the appropriate output actions.
The output actions of the fuzzy logic controller include the speed of each wheel's motor.


## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Developers

- [Georges Daou](https://github.com/George1044)
- [Charbel Bou Maroun](https://github.com/Charbel199)
- [Khaled Jalloul](https://github.com/khaledjalloul)
