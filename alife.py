"""

alife.py: An animated Artificial Life Simulation

Key concepts:
- Modeling and simulation as a way of understanding complex systems
- Artificial Life: Life as it could be
- Numpy array processing!
- Animation using the matplotlib.animation library

Key life lesson: Let curiosity be your guide.

"""

import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import seaborn as sns
import argparse

# Using argparse to tweak six arguments
parser = argparse.ArgumentParser()
parser.add_argument('--grass_rate', default=0.025)
parser.add_argument('--k_value', default=5)
parser.add_argument('--field_size', default=300)
parser.add_argument('--initial_rabbits', default=100)
parser.add_argument('--initial_foxes', default=100)
parser.add_argument('--generation', default=1000)
args = parser.parse_args()


class Rabbit:
    """ A furry creature roaming a field in search of grass to eat.
    Mr. Rabbit must eat enough to reproduce, otherwise he will starve. """

    def __init__(self):
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.eaten = 0
        self.eaten_by_fox = False

    def reproduce(self):
        """ Make a new rabbit at the same location.
         Reproduction is hard work! Each reproducing
         rabbit's eaten level is reset to zero. """
        self.eaten = 0
        return copy.deepcopy(self)

    def eat(self, amount):
        """ Feed the rabbit some grass """
        self.eaten += amount

    def move(self):
        """ Move up, down, left, right randomly """

        if WRAP:
            self.x = (self.x + rnd.choice([-1,0,1])) % SIZE
            self.y = (self.y + rnd.choice([-1,0,1])) % SIZE
        else:
            self.x = min(SIZE-1, max(0, (self.x + rnd.choice([-1, 0, 1]))))
            self.y = min(SIZE-1, max(0, (self.y + rnd.choice([-1, 0, 1]))))
        return self.x, self.y

class Fox:
    def __init__(self):
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.starving_days = 0

    def reproduce(self):
        self.starving_days = 0
        return copy.deepcopy(self)

    def eat(self):
        # fox is not starving after eating
        self.starving_days = 0

    def move(self):
        if WRAP:
            self.x = (self.x + rnd.choice([-2, -1, 0, 1, 2])) % SIZE
            self.y = (self.y + rnd.choice([-2, -1, 0, 1, 2])) % SIZE
        else:
            self.x = min(SIZE-1, max(0, (self.x + rnd.choice([-2, -1, 0, 1, 2]))))
            self.y = min(SIZE-1, max(0, (self.y + rnd.choice([-2, -1, 0, 1, 2]))))


class Field:
    """ A field is a patch of grass with 0 or more rabbits hopping around
    in search of grass """

    def __init__(self):
        """ Create a patch of grass with dimensions SIZE x SIZE
        and initially no rabbits """
        self.rabbits = []
        self.foxes = []
        self.field = np.ones(shape=(SIZE, SIZE), dtype = int)
        self.rabbit_location_nums = np.zeros(shape=(SIZE, SIZE), dtype = int)


    def add_rabbit(self, rabbit):
        """ A new rabbit is added to the field """
        self.rabbits.append(rabbit)
        self.rabbit_location_nums[rabbit.x, rabbit.y] += 1

    def add_fox(self, fox):
        self.foxes.append(fox)

    def move(self):
        """ Rabbits move """
        for r in self.rabbits:
            new_x, new_y = r.move()
            self.rabbit_location_nums[r.x, r.y] -= 1
            self.rabbit_location_nums[new_x, new_y] += 1
        """ Foxes move"""
        for f in self.foxes:
            f.move()

    def eat(self):
        """ Rabbits eat (if they find grass where they are) """
        for r in self.rabbits:
            r.eat(self.field[r.x, r.y])
            self.field[r.x, r.y] = 0

        """ Foxes eat (if they find rabbit where they are) """
        for f in self.foxes:
            if self.rabbit_location_nums[f.x, f.y] > 0:
                f.eat()
                self.rabbit_location_nums[f.x, f.y] -= 1
            else:
                f.starving_days += 1

    def survive(self):
        """ Rabbits who eat some grass live to eat another day """
        survived_rabbits = []
        for r in self.rabbits:
            if r.eaten > 0 and not r.eaten_by_fox:
                survived_rabbits.append(r)
            else:
                self.rabbit_location_nums[r.x, r.y] -= 1
        self.rabbits = survived_rabbits
        #self.rabbits = [ r for r in self.rabbits if r.eaten > 0]
        """ Foxes who is not starving for K days live to eat another day """
        self.foxes = [f for f in self.foxes if f.starving_days <= K]


    def reproduce(self):
        """ Rabbits reproduce like rabbits. """
        born = []
        for r in self.rabbits:
            for _ in range(rnd.randint(1, OFFSPRING)):
                reproduced_rabbit = r.reproduce()
                born.append(reproduced_rabbit)
                self.rabbit_location_nums[reproduced_rabbit.x, reproduced_rabbit.y] += 1
        self.rabbits += born

        """ Foxes reproduce at most one offspring"""
        fox_born = []
        for f in self.foxes:
            # for _ in range(rnd.randint(0, 1)):
                if f.starving_days == 0:
                    fox_born.append(f.reproduce())
        self.foxes += fox_born

    def grow(self):
        """ Grass grows back with some probability """
        growloc = (np.random.rand(SIZE, SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)


    def generation(self):
        """ Run one generation of rabbits and foxes """
        self.move()
        self.eat()
        self.survive()
        self.reproduce()
        self.grow()

def animate(i, field, im, rabbitDot, foxDot):
    # for _ in range(3):
    #     field.generation()
    field.generation()
    im.set_array(field.field)
    print("rabbits num {}".format(len(field.rabbits)))
    print("foxes num {}".format(len(field.foxes)))
    rabbitDot.set_data([r.y for r in field.rabbits], [r.x for r in field.rabbits])
    foxDot.set_data([f.y for f in field.foxes], [f.x for f in field.foxes])
    plt.title("generation = " + str(i))
    return [im, rabbitDot, foxDot]


def main():
    # Create a field
    field = Field()

    # Then God created rabbits....
    for _ in range(INITIAL_RABBITS):
        field.add_rabbit(Rabbit())
    # And foxes
    for _ in range(INITIAL_FOXES):
        field.add_fox(Fox())
    # Animate the world!

    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(array, cmap='Greens', interpolation='hamming', aspect='auto', vmin=0, vmax=1)
    rabbitDot, = plt.plot(0, 0, 'bo')
    foxDot, = plt.plot(0, 0, 'ro')
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im, rabbitDot, foxDot, ), frames=GENERATION, interval=1, repeat=False)
    plt.show()


if __name__ == '__main__':
    SIZE = int(args.field_size)  # The dimensions of the field
    OFFSPRING = 2  # Max offspring offspring when a rabbit reproduces
    GRASS_RATE = float(args.grass_rate)  # Probability that grass grows back at any location in the next season.
    WRAP = False
    K = int(args.k_value)
    INITIAL_RABBITS = int(args.initial_rabbits)
    INITIAL_FOXES = int(args.initial_foxes)
    GENERATION = int(args.generation)
    main()


