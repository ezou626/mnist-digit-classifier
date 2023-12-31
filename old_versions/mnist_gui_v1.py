'''
The basic framework of the GUI was ChatGPTed lol
'''

#imports
import tkinter as tk
import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn
from typing import List
from standard_cnn import get_prediction, ConvoNet

#constants
GRID_SIZE = 28
CELL_SIZE = 10
BRIGHT_COLOR = 0.99
DARK_COLOR = 0.7

#load model
MODEL = torch.load('mnist_classifier.pth')
MODEL.eval()

#transform_image
transform: transforms.Compose = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
        ])

#tkinter app class
class DrawingApp:
    def __init__(self, root):
        self.root = root
        
        #canvas
        self.canvas = tk.Canvas(root, width=GRID_SIZE*CELL_SIZE, height=GRID_SIZE*CELL_SIZE, bg='black')
        self.canvas.pack()
        self.canvas.bind('<B1-Motion>', self.draw)
        
        #data matrix
        self.matrix = torch.Tensor(np.array([np.array([0 for _ in range(GRID_SIZE)]) for _ in range(GRID_SIZE)]))
        
        #response label
        self.label = tk.Label(root, text='Click the Identify button to have your digit identified.')
        self.label.pack()

    #draw method (main brush with tapered edges)
    def draw(self, event):
        x = event.x // CELL_SIZE
        y = event.y // CELL_SIZE
        half_width = 1
        x1 = max(0, (x - half_width)) * CELL_SIZE
        y1 = max(0, (y - half_width)) * CELL_SIZE
        x2 = min(GRID_SIZE, (x + half_width + 1)) * CELL_SIZE
        y2 = min(GRID_SIZE, (y + half_width + 1)) * CELL_SIZE
        for x_i in range(max(0, (x - half_width)), min(GRID_SIZE, (x + half_width + 1))):
            for y_i in range(max(0, (y - half_width)), min(GRID_SIZE, (y + half_width + 1))):
                if (x_i, y_i) == (x, y):
                    self.matrix[y_i][x_i] = BRIGHT_COLOR
                elif not self.matrix[y_i][x_i]: #don't override previous 1s
                    self.matrix[y_i][x_i] = DARK_COLOR
        self.canvas.create_rectangle(x1, y1, x2, y2, fill='gray70', outline='gray70')
        for x_i in range(max(0, (x - half_width)), min(GRID_SIZE, (x + half_width + 1))):
            for y_i in range(max(0, (y - half_width)), min(GRID_SIZE, (y + half_width + 1))):
                cell_x = x_i * CELL_SIZE
                cell_y = y_i * CELL_SIZE
                if self.matrix[y_i][x_i] == BRIGHT_COLOR:
                    self.canvas.create_rectangle(cell_x, cell_y, 
                                                 cell_x + CELL_SIZE, cell_y + CELL_SIZE, 
                                                 fill='white', outline='white')
            

    def clear(self):
        self.canvas.delete('all')
        self.matrix = torch.Tensor(np.array([np.array([0 for _ in range(GRID_SIZE)]) for _ in range(GRID_SIZE)]))
        self.white_units = []
            
    def id(self):
        image = self.matrix.unsqueeze(0).unsqueeze(0)
        normed_image = transform(image)
        prediction = get_prediction(MODEL.forward(normed_image)).item()
        self.label.config(text = f'Your number is {prediction}. Clear to try another!')

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Number Identification App")
    app = DrawingApp(root)

    clear_button = tk.Button(root, text='Clear Drawing', command=app.clear)
    clear_button.pack(side='left')
    
    id_button = tk.Button(root, text='Identify', command=app.id)
    id_button.pack(side = 'left')

    root.mainloop()