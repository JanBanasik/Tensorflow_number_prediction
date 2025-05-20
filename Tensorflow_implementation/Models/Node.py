import pygame
from Tensorflow_implementation.Models.Colors import Color


class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = Color.WHITE.value
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_colored(self):
        return self.color == Color.BLACK.value

    def reset(self):
        self.color = Color.WHITE.value

    def make_colored(self):
        self.color = Color.BLACK.value

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def __lt__(self, other):
        return False

    def __hash__(self):
        return hash((self.x, self.y))
