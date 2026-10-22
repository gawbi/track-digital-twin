"""Evaluation-only ordered gates; never supplied to mission control."""

import math


class GateCounter:
    def __init__(self, gates, radius_m=0.13):
        self.gates = [tuple(point) for point in gates]
        if len(self.gates) < 3 or math.dist(self.gates[0], self.gates[-1]) > 1e-6:
            raise ValueError("Mission evaluation gates must form an explicit closed sequence")
        if any(math.dist(a, b) <= 2 * radius_m for a, b in zip(self.gates, self.gates[1:])):
            raise ValueError("Adjacent evaluation gates overlap")
        self.radius, self.next, self.laps, self.distance, self.previous = radius_m, 0, 0, 0.0, None

    def update(self, x, y):
        point = (x, y)
        if self.previous is not None:
            self.distance += math.dist(point, self.previous)
        self.previous = point
        if math.dist(point, self.gates[self.next]) <= self.radius:
            self.next += 1
            if self.next == len(self.gates):
                self.laps += 1
                self.next = 1
        return self.laps
