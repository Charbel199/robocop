import skfuzzy as fuzz
import numpy as np
import skfuzzy.control as ctrl


class FuzzyRoverController:
    def __init__(self, auto_mf=True):
        # Define input and output variables
        self.distance = ctrl.Antecedent(np.arange(0, 11, 1), 'distance')
        self.deviation = ctrl.Antecedent(np.arange(-5, 6, 1), 'deviation')
        self.r_speed = ctrl.Antecedent(np.arange(0, 13, 1), 'r_speed')
        self.l_speed = ctrl.Antecedent(np.arange(0, 13, 1), 'l_speed')

        self.r_motor = ctrl.Consequent(np.arange(0, 101, 1), 'r_motor', defuzzify_method='centroid')
        self.l_motor = ctrl.Consequent(np.arange(0, 101, 1), 'l_motor', defuzzify_method='centroid')

        # Define membership functions
        if auto_mf:
            self.distance.automf(3, names=['low', 'medium', 'high'])
            self.deviation.automf(3, names=['poor', 'average', 'good'])
            self.r_speed.automf(4, names=['very slow', 'slow', 'average', 'fast'])
            self.l_speed.automf(4, names=['very slow', 'slow', 'average', 'fast'])
            self.r_motor.automf(3, names=['low', 'medium', 'high'])
            self.l_motor.automf(3, names=['low', 'medium', 'high'])
        else:
            self.distance['low'] = fuzz.trimf(self.distance.universe, [0, 0, 5])
            self.distance['medium'] = fuzz.trimf(self.distance.universe, [0, 5, 10])
            self.distance['high'] = fuzz.trimf(self.distance.universe, [5, 10, 10])

            self.deviation['poor'] = fuzz.trimf(self.deviation.universe, [-5, -5, 0])
            self.deviation['average'] = fuzz.trimf(self.deviation.universe, [-5, 0, 5])
            self.deviation['good'] = fuzz.trimf(self.deviation.universe, [0, 5, 5])

            self.r_speed['very slow'] = fuzz.trimf(self.r_speed.universe, [0, 0, 2])
            self.r_speed['slow'] = fuzz.trimf(self.r_speed.universe, [1, 4, 7])
            self.r_speed['average'] = fuzz.trimf(self.r_speed.universe, [5, 8, 11])
            self.r_speed['fast'] = fuzz.trimf(self.r_speed.universe, [10, 12, 12])

            self.l_speed['very slow'] = fuzz.trimf(self.l_speed.universe, [0, 0, 2])
            self.l_speed['slow'] = fuzz.trimf(self.l_speed.universe, [1, 4, 7])
            self.l_speed['average'] = fuzz.trimf(self.l_speed.universe, [5, 8, 11])
            self.l_speed['fast'] = fuzz.trimf(self.l_speed.universe, [10, 12, 12])

            self.r_motor['low'] = fuzz.trimf(self.r_motor.universe, [0, 0, 50])
            self.r_motor['medium'] = fuzz.trimf(self.r_motor.universe, [0, 50, 100])
            self.r_motor['high'] = fuzz.trimf(self.r_motor.universe, [50, 100, 100])

            self.l_motor['low'] = fuzz.trimf(self.l_motor.universe, [0, 0, 50])
            self.l_motor['medium'] = fuzz.trimf(self.l_motor.universe, [0, 50, 100])
            self.l_motor['high'] = fuzz.trimf(self.l_motor.universe, [50, 100, 100])

        self.rules = []

    def define_rules(self):
        # Define fuzzy rules
        self.rules.append(ctrl.Rule(self.distance['low'] & self.deviation['poor'] & self.r_speed['fast'], self.r_motor['low']))
        self.rules.append(ctrl.Rule(self.distance['medium'] & self.deviation['average'] & self.r_speed['average'], self.r_motor['medium']))
        self.rules.append(ctrl.Rule(self.distance['medium'] & self.deviation['good'] & self.r_speed['slow'], self.r_motor['high']))

        self.rules.append(ctrl.Rule(self.distance['low'] & self.deviation['average'] & self.l_speed['fast'], self.l_motor['low']))
        self.rules.append(ctrl.Rule(self.distance['low'] & self.deviation['good'] & self.l_speed['average'], self.l_motor['medium']))
        self.rules.append(ctrl.Rule(self.distance['high'] & self.deviation['good'] & self.l_speed['slow'], self.l_motor['high']))

    def create_control_system(self):
        # Create control system
        self.motors_ctrl = ctrl.ControlSystem(self.rules)
        self.motors = ctrl.ControlSystemSimulation(self.motors_ctrl)

    def compute_output(self, distance, deviation, r_speed, l_speed, sugeno=False):
        self.motors.input['distance'] = distance
        self.motors.input['deviation'] = deviation
        self.motors.input['r_speed'] = r_speed
        self.motors.input['l_speed'] = l_speed
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

    for i in range(1):
        l_motor_value, r_motor_value = fu.compute_output(4, 3, 11, 11, sugeno=False)
        print(f"l_motor_value = {l_motor_value}, r_motor_value = {r_motor_value}")
