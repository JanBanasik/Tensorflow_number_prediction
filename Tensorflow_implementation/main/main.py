from Tensorflow_implementation.Utils import draw_grid
import pygame
from Tensorflow_implementation.Models.LabelPredictorClass import LabelPredictor

if __name__ == '__main__':
    WIDTH = 800
    WIN = pygame.display.set_mode((WIDTH, WIDTH))
    pygame.display.set_caption('Number prediction')

    pygame.font.init()
    FONT = pygame.font.SysFont('Arial', 30)

    predictor = LabelPredictor()
    draw_grid.main(WIN, WIDTH, predictor)