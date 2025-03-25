# """ Heidi Gwinner
#     CS-481, Artificial Intelligence
#     Dr. Farmer, Winter 2025
#     Assignment 4: Fuzzy Logic Control System    """

# """
# This module implements a fuzzy logic system to adjust speed based on distance and relative velocity.

# Modules:
#     - matplotlib.pyplot:    Used for plotting the fuzzy sets.
#     - fuzzylogic.classes:   Provides Domain, Set, and Rule classes for fuzzy logic.
#     - fuzzylogic.functions: Provides various membership functions like R, S, bounded_sigmoid, gauss, and trapezoid.
# Domains:
#     - distance:          Represents the distance with fuzzy sets for far, moderately_far, moderately_close, close, and very_close.
#     - relative_velocity: Represents the relative velocity with fuzzy sets for closing_fast, closing_slowly, neutral, receding_slowly, and receding_quickly.
#     - adjust_speed:      Represents the speed adjustment with fuzzy sets for decelerate_quickly, decelerate_slowly, maintain_speed, accelerate_slowly, and accelerate_quickly.
# Rules:
#     - far1 to far5:             Rules for when the distance is far.
#     - modfar1 to modfar5:       Rules for when the distance is moderately far.
#     - modclose1 to modclose5:   Rules for when the distance is moderately close.
#     - close1 to close5:         Rules for when the distance is close.
#     - veryclose1 to veryclose5: Rules for when the distance is very close.  """

import numpy as np
from matplotlib import pyplot
from fuzzylogic.classes import Domain, Set, Rule
from fuzzylogic.functions import R, S, bounded_sigmoid
from fuzzylogic.functions import (gauss, trapezoid)


# distance = Domain("distance", 0, 700, res=0.1)
# distance.far = R(500, 550)
# distance.moderately_far = trapezoid(200, 300, 500, 600, c_m=1)
# distance.moderately_close = trapezoid(80, 120, 250, 300, c_m=1)
# distance.close = trapezoid(40, 60, 90, 110, c_m=1)
# distance.very_close = S(35, 50)
# distance.far.plot()
# distance.moderately_far.plot()
# distance.moderately_close.plot()
# distance.close.plot()
# distance.very_close.plot()

# relative_velocity = Domain("relative_velocity", -25, 30, res=0.1)
# relative_velocity.closing_fast = R(13,15)
# relative_velocity.closing_slowly = trapezoid(3, 5, 12, 15)
# relative_velocity.neutral = trapezoid(-5, -2, 3, 5)
# relative_velocity.receding_slowly = trapezoid(-12, -10, -4, -2)
# relative_velocity.receding_quickly = S(-12,-10)
# relative_velocity.closing_fast.plot()
# relative_velocity.closing_slowly.plot()
# relative_velocity.neutral.plot()
# relative_velocity.receding_slowly.plot()
# relative_velocity.receding_quickly.plot()

# adjust_speed = Domain("adjust_speed", -0.5, 0.5, res=0.1)
# adjust_speed.accelerate_quickly = bounded_sigmoid(-0.4,-0.3, inverse=True)
# adjust_speed.accelerate_slowly = gauss(-0.2, 40, c_m=1)
# adjust_speed.maintain_speed = gauss(0, 40, c_m=1)
# adjust_speed.decelerate_slowly = gauss(0.2, 40, c_m=1)
# adjust_speed.decelerate_quickly = bounded_sigmoid(0.3,0.4)
# adjust_speed.decelerate_quickly.plot()
# adjust_speed.decelerate_slowly.plot()
# adjust_speed.maintain_speed.plot()
# adjust_speed.accelerate_slowly.plot()
# adjust_speed.accelerate_quickly.plot()

# far1 = Rule({(distance.far, relative_velocity.receding_quickly): adjust_speed_universe.accelerate_quickly})
# far2 = Rule({(distance.far, relative_velocity.receding_slowly): adjust_speed_universe.accelerate_quickly})
# far3 = Rule({(distance.far, relative_velocity.neutral): adjust_speed_universe.accelerate_quickly})
# far4 = Rule({(distance.far, relative_velocity.closing_slowly): adjust_speed_universe.accelerate_slowly})
# far5 = Rule({(distance.far, relative_velocity.closing_fast): adjust_speed_universe.accelerate_slowly})

# modfar1 = Rule({(distance.moderately_far, relative_velocity.receding_quickly): adjust_speed_universe.accelerate_quickly})
# modfar2 = Rule({(distance.moderately_far, relative_velocity.receding_slowly): adjust_speed_universe.accelerate_quickly})
# modfar3 = Rule({(distance.moderately_far, relative_velocity.neutral): adjust_speed_universe.accelerate_slowly})
# modfar4 = Rule({(distance.moderately_far, relative_velocity.closing_slowly): adjust_speed_universe.accelerate_slowly})
# modfar5 = Rule({(distance.moderately_far, relative_velocity.closing_fast): adjust_speed_universe.maintain_speed})

# modclose1 = Rule({(distance.moderately_close, relative_velocity.receding_quickly): adjust_speed_universe.accelerate_quickly})
# modclose2 = Rule({(distance.moderately_close, relative_velocity.receding_slowly): adjust_speed_universe.accelerate_slowly})
# modclose3 = Rule({(distance.moderately_close, relative_velocity.neutral): adjust_speed_universe.accelerate_slowly})
# modclose4 = Rule({(distance.moderately_close, relative_velocity.closing_slowly): adjust_speed_universe.maintain_speed})
# modclose5 = Rule({(distance.moderately_close, relative_velocity.closing_fast): adjust_speed_universe.decelerate_slowly})

# close1 = Rule({(distance.close, relative_velocity.receding_quickly): adjust_speed_universe.accelerate_slowly})
# close2 = Rule({(distance.close, relative_velocity.receding_slowly): adjust_speed_universe.accelerate_slowly})
# close3 = Rule({(distance.close, relative_velocity.neutral): adjust_speed_universe.maintain_speed})
# close4 = Rule({(distance.close, relative_velocity.closing_slowly): adjust_speed_universe.decelerate_slowly})
# close5 = Rule({(distance.close, relative_velocity.closing_fast): adjust_speed_universe.decelerate_quickly})

# veryclose1 = Rule({(distance.very_close, relative_velocity.receding_quickly): adjust_speed_universe.accelerate_quickly})
# veryclose2 = Rule({(distance.very_close, relative_velocity.receding_slowly): adjust_speed_universe.accelerate_quickly})
# veryclose3 = Rule({(distance.very_close, relative_velocity.neutral): adjust_speed_universe.accelerate_quickly})
# veryclose4 = Rule({(distance.very_close, relative_velocity.closing_slowly): adjust_speed_universe.accelerate_slowly})
# veryclose5 = Rule({(distance.very_close, relative_velocity.closing_fast): adjust_speed_universe.accelerate_slowly})

# # Define the rules object as a combination of all individual rules
# rules = Rule({(distance.far, relative_velocity.receding_quickly): adjust_speed_universe.accelerate_quickly,
#               (distance.far, relative_velocity.receding_slowly): adjust_speed_universe.accelerate_quickly,
#               (distance.far, relative_velocity.neutral): adjust_speed_universe.accelerate_quickly,
#               (distance.far, relative_velocity.closing_slowly): adjust_speed_universe.accelerate_slowly,
#               (distance.far, relative_velocity.closing_fast): adjust_speed_universe.accelerate_slowly,
#               (distance.moderately_far, relative_velocity.receding_quickly): adjust_speed_universe.accelerate_quickly,
#               (distance.moderately_far, relative_velocity.receding_slowly): adjust_speed_universe.accelerate_quickly,
#               (distance.moderately_far, relative_velocity.neutral): adjust_speed_universe.accelerate_slowly,
#               (distance.moderately_far, relative_velocity.closing_slowly): adjust_speed_universe.accelerate_slowly,
#               (distance.moderately_far, relative_velocity.closing_fast): adjust_speed_universe.maintain_speed,
#               (distance.moderately_close, relative_velocity.receding_quickly): adjust_speed_universe.accelerate_quickly,
#               (distance.moderately_close, relative_velocity.receding_slowly): adjust_speed_universe.accelerate_slowly,
#               (distance.moderately_close, relative_velocity.neutral): adjust_speed_universe.accelerate_slowly,
#               (distance.moderately_close, relative_velocity.closing_slowly): adjust_speed_universe.maintain_speed,
#               (distance.moderately_close, relative_velocity.closing_fast): adjust_speed_universe.decelerate_slowly,
#               (distance.close, relative_velocity.receding_quickly): adjust_speed_universe.accelerate_slowly,
#               (distance.close, relative_velocity.receding_slowly): adjust_speed_universe.accelerate_slowly,
#               (distance.close, relative_velocity.neutral): adjust_speed_universe.maintain_speed,
#               (distance.close, relative_velocity.closing_slowly): adjust_speed_universe.decelerate_slowly,
#               (distance.close, relative_velocity.closing_fast): adjust_speed_universe.decelerate_quickly,
#               (distance.very_close, relative_velocity.receding_quickly): adjust_speed_universe.accelerate_quickly,
#               (distance.very_close, relative_velocity.receding_slowly): adjust_speed_universe.accelerate_quickly,
#               (distance.very_close, relative_velocity.neutral): adjust_speed_universe.accelerate_quickly,
#               (distance.very_close, relative_velocity.closing_slowly): adjust_speed_universe.accelerate_slowly,
#               (distance.very_close, relative_velocity.closing_fast): adjust_speed_universe.accelerate_slowly})

# # Combine all individual rules into a list
# all_rules = [far1, far2, far3, far4, far5, modfar1, modfar2, modfar3, modfar4, modfar5,
#              modclose1, modclose2, modclose3, modclose4, modclose5, close1, close2,
#              close3, close4, close5, veryclose1, veryclose2, veryclose3, veryclose4, veryclose5]


import numpy as np
from matplotlib import pyplot

# Define membership functions
def R(x, a, b):
    return np.maximum(0, np.minimum(1, (x - a) / (b - a)))

def S(x, a, b):
    return np.maximum(0, np.minimum(1, (b - x) / (b - a)))

def trapezoid(x, a, b, c, d):
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a), 1), (d - x) / (d - c)))

def bounded_sigmoid(x, a, b, inverse=False):
    if inverse:
        return 1 / (1 + np.exp(-10 * (x - a))) - 1 / (1 + np.exp(-10 * (x - b)))
    else:
        return 1 / (1 + np.exp(-10 * (x - b))) - 1 / (1 + np.exp(-10 * (x - a)))

def gauss(x, mean, sigma, c_m=1):
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) * c_m

# Define domains and membership functions
distance = {
    "far": lambda x: R(x, 500, 550),
    "moderately_far": lambda x: trapezoid(x, 200, 300, 500, 600),
    "moderately_close": lambda x: trapezoid(x, 80, 120, 250, 300),
    "close": lambda x: trapezoid(x, 40, 60, 90, 110),
    "very_close": lambda x: S(x, 35, 50)
}

relative_velocity = {
    "closing_fast": lambda x: R(x, 13, 15),
    "closing_slowly": lambda x: trapezoid(x, 3, 5, 12, 15),
    "neutral": lambda x: trapezoid(x, -5, -2, 3, 5),
    "receding_slowly": lambda x: trapezoid(x, -12, -10, -4, -2),
    "receding_quickly": lambda x: S(x, -12, -10)
}

adjust_speed = {
    "decelerate_quickly": lambda x: bounded_sigmoid(x, 0.3, 0.4),
    "decelerate_slowly": lambda x: gauss(x, 0.2, 0.1),
    "maintain_speed": lambda x: gauss(x, 0, 0.1),
    "accelerate_slowly": lambda x: gauss(x, -0.2, 0.1),
    "accelerate_quickly": lambda x: bounded_sigmoid(x, -0.4, -0.3, inverse=True)
}

# Define rules
rules = [
    (("far", "receding_quickly"), "accelerate_quickly"),
    (("far", "receding_slowly"), "accelerate_quickly"),
    (("far", "neutral"), "accelerate_quickly"),
    (("far", "closing_slowly"), "accelerate_slowly"),
    (("far", "closing_fast"), "accelerate_slowly"),
    (("moderately_far", "receding_quickly"), "accelerate_quickly"),
    (("moderately_far", "receding_slowly"), "accelerate_quickly"),
    (("moderately_far", "neutral"), "accelerate_slowly"),
    (("moderately_far", "closing_slowly"), "accelerate_slowly"),
    (("moderately_far", "closing_fast"), "maintain_speed"),
    (("moderately_close", "receding_quickly"), "accelerate_quickly"),
    (("moderately_close", "receding_slowly"), "accelerate_slowly"),
    (("moderately_close", "neutral"), "accelerate_slowly"),
    (("moderately_close", "closing_slowly"), "maintain_speed"),
    (("moderately_close", "closing_fast"), "decelerate_slowly"),
    (("close", "receding_quickly"), "accelerate_slowly"),
    (("close", "receding_slowly"), "accelerate_slowly"),
    (("close", "neutral"), "maintain_speed"),
    (("close", "closing_slowly"), "decelerate_slowly"),
    (("close", "closing_fast"), "decelerate_quickly"),
    (("very_close", "receding_quickly"), "accelerate_quickly"),
    (("very_close", "receding_slowly"), "accelerate_quickly"),
    (("very_close", "neutral"), "accelerate_quickly"),
    (("very_close", "closing_slowly"), "accelerate_slowly"),
    (("very_close", "closing_fast"), "accelerate_slowly")
]

# Define crisp values for the sensor inputs
crisp_distance = 250
crisp_relative_velocity = -10

# Determine the memberships in the various fuzzy sets
distance_memberships = {key: func(crisp_distance) for key, func in distance.items()}
relative_velocity_memberships = {key: func(crisp_relative_velocity) for key, func in relative_velocity.items()}

# Print the memberships
print("Distance memberships:", distance_memberships)
print("Relative velocity memberships:", relative_velocity_memberships)

# Manually define the universe for adjust_speed
adjust_speed_values = np.arange(-0.5, 0.5, 0.1)

# Aggregate the results from the rules
output_values = np.zeros_like(adjust_speed_values)
for (dist_key, vel_key), speed_key in rules:
    dist_membership = distance_memberships[dist_key]
    vel_membership = relative_velocity_memberships[vel_key]
    rule_strength = min(dist_membership, vel_membership)
    output_values = np.maximum(output_values, rule_strength * adjust_speed[speed_key](adjust_speed_values))

# Defuzzify using the centroid method
numerator = np.sum(output_values * adjust_speed_values)
denominator = np.sum(output_values)
if denominator != 0:
    defuzzified_value = numerator / denominator
else:
    defuzzified_value = 0  # Handle the case where the denominator is zero

print("Defuzzified output (adjust speed):", defuzzified_value)



# Plot the output fuzzy sets
# pyplot.figure()
# pyplot.plot(adjust_speed_values, adjust_speed_universe["accelerate_quickly"](adjust_speed_values), label='Accelerate Quickly')
# pyplot.plot(adjust_speed_values, adjust_speed_universe["accelerate_slowly"](adjust_speed_values), label='Accelerate Slowly')
# pyplot.plot(adjust_speed_values, adjust_speed_universe["maintain_speed"](adjust_speed_values), label='Maintain Speed')
# pyplot.plot(adjust_speed_values, adjust_speed_universe["decelerate_slowly"](adjust_speed_values), label='Decelerate Slowly')
# pyplot.plot(adjust_speed_values, adjust_speed_universe["decelerate_quickly"](adjust_speed_values), label='Decelerate Quickly')
# pyplot.plot(adjust_speed_values, output_values, label='Aggregated Output', linestyle='--', linewidth=2)
# pyplot.xlabel('Adjust Speed')
# pyplot.ylabel('Membership Degree')
# pyplot.title('Output Fuzzy Sets')
# pyplot.legend()
# pyplot.show()