import simpy
import random

def create_road_network(env, width, height, length_range):
  """Creates a grid of road networks with different lengths."""
  road_network = []
  for x in range(width):
    row = []
    for y in range(height):
      length = random.randint(*length_range)
      line = simpy.Line(env, capacity=1)
      row.append(line)
    road_network.append(row)
  return road_network

def main():
  env = simpy.Environment()
  road_network = create_road_network(env, 10, 10, (10, 50))
  for line in road_network:
    env.process(lambda: line.process())
  env.run(until=100)

if __name__ == "__main__":
  main()
