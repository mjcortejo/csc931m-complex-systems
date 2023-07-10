import simpy
import tkinter as tk
import random
import math

class Car:
    def __init__(self, canvas, x, y):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.radius = 5
        self.destination = None
        self.shape = None

    def draw(self):
        x0 = self.x - self.radius
        y0 = self.y - self.radius
        x1 = self.x + self.radius
        y1 = self.y + self.radius
        self.shape = self.canvas.create_oval(x0, y0, x1, y1, fill="blue")

    def move_to_destination(self):
        if self.destination is None:
            self.destination = (random.randint(0, self.canvas.winfo_width()), 
                                random.randint(0, self.canvas.winfo_height()))

    def move(self):
        if self.destination is not None:
            dx = self.destination[0] - self.x
            dy = self.destination[1] - self.y
            distance = math.sqrt(dx**2 + dy**2)
            if distance > 1:  # Only move if the distance is greater than 1
                speed = 1  # Adjust the speed as needed
                self.x += (dx / distance) * speed
                self.y += (dy / distance) * speed
                self.canvas.coords(self.shape, self.x - self.radius, self.y - self.radius,
                                   self.x + self.radius, self.y + self.radius)
            else:
                print("Car reached its destination!")


class RoadNetwork:
    def __init__(self, env, canvas, width, height):
        self.env = env
        self.canvas = canvas
        self.width = width
        self.height = height
        self.cars = []
        self.destinations = []

    def create_grid(self, rows, cols):
        cell_width = self.width / cols
        cell_height = self.height / rows
        for i in range(rows + 1):
            y = i * cell_height
            self.canvas.create_line(0, y, self.width, y, fill="gray")
        for j in range(cols + 1):
            x = j * cell_width
            self.canvas.create_line(x, 0, x, self.height, fill="gray")

    def place_car(self, x, y):
        car = Car(self.canvas, x, y)
        self.cars.append(car)
        car.draw()

    def draw_destination(self, x, y):
        x0 = x - 3
        y0 = y - 3
        x1 = x + 3
        y1 = y + 3
        self.canvas.create_oval(x0, y0, x1, y1, fill="red")
        self.destinations.append((x, y))

env = simpy.Environment()
root = tk.Tk()
canvas = tk.Canvas(root, width=800, height=600)
canvas.pack()

network = RoadNetwork(env, canvas, 800, 600)
network.create_grid(rows=10, cols=8)

# Place cars at specific positions
network.place_car(x=100, y=100)
network.place_car(x=300, y=200)
network.place_car(x=600, y=400)

# Set random destinations for the cars
for car in network.cars:
    car.move_to_destination()
    network.draw_destination(car.destination[0], car.destination[1])

def simulate_movement():
    while True:
        for car in network.cars:
            car.move()
        if all(car.destination is None for car in network.cars):
            print("All cars reached their destination!")
            break
        yield env.timeout(0.1)  # Adjust the simulation time step as needed

fps = 60
duration = 10

# env = simpy.rt.RealtimeEnvironment(factor=1/fps, strict=False)
env.process(simulate_movement())
# env.run(until=10)  # Adjust the simulation duration as needed
env.run(until=duration * fps + 1)


root.mainloop()