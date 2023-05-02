#! /usr/bin/env python

import skfuzzy as fuzz
import numpy as np
import skfuzzy.control as ctrl
import skfuzzy.control.visualization as vsl
import matplotlib.pyplot as plt


class FuzzyRoverController:
    def __init__(self, auto_mf=True):
        self.MFS = 7
        self.FAM = [
            [1, 2, 3, 5, 5, 5, 5],
            [1, 3, 4, 5, 5, 5, 5],
            [1, 4, 4, 5, 5, 5, 5],
            [2, 4, 5, 7, 7, 7, 7],
            [7, 5, 5, 7, 7, 7, 7],
            [7, 6, 5, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7]
        ]

        # Define input and output variables
        self.distance = ctrl.Antecedent(np.arange(0, 3, 0.01), 'distance')
        self.deviation = ctrl.Antecedent(np.arange(-50, 51, 1), 'deviation')
        # self.r_speed = ctrl.Antecedent(np.arange(0, 13, 0.1), 'r_speed')
        # self.l_speed = ctrl.Antecedent(np.arange(0, 13, 0.1), 'l_speed')

        self.r_motor = ctrl.Consequent(np.arange(0, 51, 1), 'r_motor', defuzzify_method='centroid')
        self.l_motor = ctrl.Consequent(np.arange(0, 51, 1), 'l_motor', defuzzify_method='centroid')
        self.names = []

        # Define membership functions
        if auto_mf:
            for i in range(1, self.MFS + 1):
                self.names.append(str(i))

            distance_limits = [self.distance.universe.min(), self.distance.universe.max()]
            distance_universe_range = distance_limits[1] - distance_limits[0]
            distance_width = distance_universe_range / ((self.MFS - 1) / 2.)
            distance_centers = np.linspace(distance_limits[0], distance_limits[1], self.MFS)
            distance_first_center = distance_centers[0]
            distance_last_center = distance_centers[self.MFS-1]

            self.distance.automf(self.MFS, names=self.names)
            #self.distance['0'].mf = fuzz.trapmf(self.distance.universe, [distance_limits[0], distance_limits[0],distance_first_center,distance_first_center + distance_width/2])
            #self.distance[str(self.MFS - 1)].mf = fuzz.trapmf(self.distance.universe, [distance_last_center - distance_width/2, distance_last_center, distance_limits[1], distance_limits[1]])



            self.deviation.automf(self.MFS, names=self.names)

            # self.r_speed.automf(4, names=['very slow', 'slow', 'average', 'fast'])
            # self.l_speed.automf(4, names=['very slow', 'slow', 'average', 'fast'])

            motor_limits = [self.r_motor.universe.min(), self.r_motor.universe.max()]
            motor_universe_range = motor_limits[1] - motor_limits[0]
            motor_width = motor_universe_range / ((self.MFS - 1) / 2.)
            motor_centers = np.linspace(motor_limits[0], motor_limits[1], self.MFS)
            motor_first_center = motor_centers[0]
            motor_last_center = motor_centers[self.MFS-1]

            self.r_motor.automf(self.MFS, names=self.names)
            # self.r_motor['0'] = fuzz.trapmf(self.r_motor.universe, [motor_limits[0], motor_limits[0],motor_first_center,motor_first_center + motor_width/2])
            # self.r_motor[str(self.MFS - 1)] = fuzz.trapmf(self.r_motor.universe, [motor_last_center - motor_width/2, motor_last_center, motor_limits[1], motor_limits[1]])

            self.l_motor.automf(self.MFS, names=self.names)
            # self.l_motor['0'] = fuzz.trapmf(self.l_motor.universe, [motor_limits[0], motor_limits[0],motor_first_center,motor_first_center + motor_width/2])
            # self.l_motor[str(self.MFS - 1)] = fuzz.trapmf(self.l_motor.universe, [motor_last_center - motor_width/2, motor_last_center, motor_limits[1], motor_limits[1]])

        else:
            self.distance['low'] = fuzz.trimf(self.distance.universe, [0, 0, 5])
            self.distance['medium'] = fuzz.trimf(self.distance.universe, [0, 5, 10])
            self.distance['high'] = fuzz.trimf(self.distance.universe, [5, 10, 10])

            self.deviation['poor'] = fuzz.trimf(self.deviation.universe, [-5, -5, 0])
            self.deviation['average'] = fuzz.trimf(self.deviation.universe, [-5, 0, 5])
            self.deviation['good'] = fuzz.trimf(self.deviation.universe, [0, 5, 5])

            # self.r_speed['very slow'] = fuzz.trimf(self.r_speed.universe, [0, 0, 2])
            # self.r_speed['slow'] = fuzz.trimf(self.r_speed.universe, [1, 4, 7])
            # self.r_speed['average'] = fuzz.trimf(self.r_speed.universe, [5, 8, 11])
            # self.r_speed['fast'] = fuzz.trimf(self.r_speed.universe, [10, 12, 12])

            # self.l_speed['very slow'] = fuzz.trimf(self.l_speed.universe, [0, 0, 2])
            # self.l_speed['slow'] = fuzz.trimf(self.l_speed.universe, [1, 4, 7])
            # self.l_speed['average'] = fuzz.trimf(self.l_speed.universe, [5, 8, 11])
            # self.l_speed['fast'] = fuzz.trimf(self.l_speed.universe, [10, 12, 12])

            self.r_motor['low'] = fuzz.trimf(self.r_motor.universe, [0, 0, 50])
            self.r_motor['medium'] = fuzz.trimf(self.r_motor.universe, [0, 50, 100])
            self.r_motor['high'] = fuzz.trimf(self.r_motor.universe, [50, 100, 100])

            self.l_motor['low'] = fuzz.trimf(self.l_motor.universe, [0, 0, 50])
            self.l_motor['medium'] = fuzz.trimf(self.l_motor.universe, [0, 50, 100])
            self.l_motor['high'] = fuzz.trimf(self.l_motor.universe, [50, 100, 100])

        self.rules = []

        distance_visualizer = vsl.FuzzyVariableVisualizer(self.deviation)
        fig, ax = distance_visualizer.view()
        # plt.show()

    def define_rules(self):
        # Define fuzzy rules
        for i, deviationArr in enumerate(self.FAM):
            for j, speed in enumerate(deviationArr):
                self.rules.append(ctrl.Rule(self.distance[str(j + 1)] & self.deviation[str(i+1)], self.l_motor[str(speed)]))

        for i, deviationArr in enumerate(reversed(self.FAM)):
            for j, speed in enumerate(deviationArr):
                self.rules.append(ctrl.Rule(self.distance[str(j + 1)] & self.deviation[str(i+1)], self.r_motor[str(speed)]))



    def create_control_system(self):
        # Create control system
        self.motors_ctrl = ctrl.ControlSystem(self.rules)
        self.motors = ctrl.ControlSystemSimulation(self.motors_ctrl)

    def compute_output(self, distance, deviation, r_speed, l_speed, sugeno=False):
        if distance < 0.2:
            return 0, 0
        self.motors.input['distance'] = distance
        self.motors.input['deviation'] = deviation
        # self.motors.input['r_speed'] = r_speed
        # self.motors.input['l_speed'] = l_speed
        self.motors.compute()

        if sugeno:
            print(f"Using Takagi-Sugeno-Kang Inference System")
            z = []
            for c in self.motors_ctrl.consequents:
                w = []
                print(f"For consequent {c.label}")
                for key, val in c.terms.items():
                    print(f"Membership of MF: {key} is {val.membership_value[self.motors]}")
                    w.append(val.membership_value[self.motors])

                z1 = 5.0 + 0.2 * distance + 0.2 * deviation + 0.2 * r_speed + 0.2 * l_speed
                z2 = 5.0 + 0.5 * distance + 0.5 * deviation + 0.5 * r_speed + 0.5 * l_speed
                z3 = 5 + 1.0 * distance + 1.0 * deviation + 1.0 * r_speed + 1.0 * l_speed
                z.append((w[0] * z1 + w[1] * z2 + w[2] * z3) / (w[0] + w[1] + w[2]))
            return z

        print(f"Using Mamdani Inference System")
        return self.motors.output['l_motor'], self.motors.output['r_motor']


if __name__ == "__main__":
    fu = FuzzyRoverController()
    fu.define_rules()
    fu.create_control_system()

    # Distance test
    for i in range(0,300):
        l_motor_value, r_motor_value = fu.compute_output(i/100, 30, 11, 11, sugeno=False)
        print(f"for distance {i/100}: l_motor_value = {l_motor_value}, r_motor_value = {r_motor_value}")

    # Deviation test
    # for i in range(-50,51):
    #     l_motor_value, r_motor_value = fu.compute_output(2, i, 11, 11, sugeno=False)
    #     print(f"for deviation {i}: l_motor_value = {l_motor_value}, r_motor_value = {r_motor_value}")
