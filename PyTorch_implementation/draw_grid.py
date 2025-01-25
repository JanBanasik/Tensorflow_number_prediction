import pygame
from Colors import Color
from Node import Node
from predict_labels import predict
import numpy as np
import torch

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption('Shortest Path Visualization')

pygame.font.init()
FONT = pygame.font.SysFont('Arial', 30)


def draw_label(win, text, x, y):
    label = FONT.render(text, True, Color.PURPLE.value)
    win.blit(label, (x, y))


def make_grid(rows, width) -> list[list[Node]]:
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)
    return grid


def draw_grid(win, rows, width) -> None:
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, Color.BLACK.value, (0, i * gap), (width, i * gap))
        pygame.draw.line(win, Color.BLACK.value, (i * gap, 0), (i * gap, width))


def draw(win, grid, rows, width, currentPrediction):
    win.fill(Color.WHITE.value)
    for row in grid:
        for node in row:
            node.draw(win)
    draw_grid(win, rows, width)
    if currentPrediction != "None":
        draw_label(win, f"Prediction: {currentPrediction}", 10, 10)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    x, y = pos
    return x // gap, y // gap


def make_predictions(labeled_image):

    labeled_image_np = np.array(labeled_image, dtype=np.float32)

    labeled_image_np = np.expand_dims(labeled_image_np, axis=0)

    input_tensor = torch.tensor(labeled_image_np)

    prediction = predict(input_tensor)

    return prediction


def transform_grid(grid):
    result = []
    for j in range(len(grid[0])):
        result.append([])
        for i in range(len(grid)):
            if grid[i][j].is_colored():
                result[j].append(1)
            else:
                result[j].append(-1)
    return result


def main(win, width):
    ROWS = 28
    grid = make_grid(ROWS, width)

    run = True
    currPrediction = "None"
    while run:
        draw(win, grid, ROWS, width, currPrediction)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                if not (0 <= row < ROWS and 0 <= col < ROWS):
                    continue
                node: Node = grid[row][col]

                if not node.is_colored():
                    node.make_colored()

            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]
                node.reset()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    labeled_image = transform_grid(grid)
                    label = make_predictions(labeled_image)
                    draw_label(win, f"Predicted: {label}", 10, 10)
                    currPrediction = label
                    pygame.display.update()

                if event.key == pygame.K_c:
                    grid = make_grid(ROWS, width)
                    win.fill(Color.WHITE.value)
                    currPrediction = "None"
                    pygame.display.update()

    pygame.quit()


main(WIN, WIDTH)
