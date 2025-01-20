import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from constants import *

file_name = "fix_5_1"
SAVE = True
PLOT = False

file_path = f"./{file_name}.json"
results_folder = f"./games/{file_name}/"
data_from_json = json.load(open(file_path, "r"))
game_states = data_from_json["game_states"]
os.makedirs(results_folder, exist_ok=True)

piece_images = {
    piece: Image.open(f'assets/{piece}.png') for piece in PIECES_TO_NUM.keys() if piece != "square"
}

def draw_chessboard(board, out, title, save=False, plot=False):
    board = np.flipud(board)
    fig, ax = plt.subplots()

    chessboard_pattern = np.zeros((8, 8))
    chessboard_pattern[1::2, ::2] = 1
    chessboard_pattern[::2, 1::2] = 1

    ax.imshow(chessboard_pattern, cmap='gray', interpolation='none')

    for i in range(8):
        for j in range(8):
            piece_value = board[i, j]
            if piece_value != 0:
                piece_name = NUM_TO_PIECE[piece_value]
                piece_image = piece_images[piece_name]
                ax.imshow(piece_image, extent=[j - 0.5, j + 0.5, i - 0.5, i + 0.5])

    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    
    fig.suptitle(title, fontsize=20)

    if save:
        fig.savefig(out)
    if plot:
        plt.show()
    matplotlib.pyplot.close()

for i in range(len(game_states)):
    draw_chessboard(np.array(game_states[i]["gs"]), f"{results_folder}{file_name}_{i}.png", f"Gamestate #{i}", save=SAVE, plot=PLOT)
