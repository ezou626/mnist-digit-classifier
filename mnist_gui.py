'''
The basic framework of the GUI was ChatGPTed lol
'''

#imports
import tkinter as tk
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from typing import List

#evaluate correctness
def get_prediction(tensor):
    return tensor.argmax(dim=-1)

#convolutional neural network
class ConvoNet(nn.Module):
    def __init__(self, inputs: int, outputs: int, features: List[int], pooling: List[int], taps: List[int]):
        super().__init__()
        self.convo_list = nn.ModuleList()
        self.pool_list = nn.ModuleList()
        for i, t in enumerate(taps):
            dim = 2 * t + 1
            self.convo_list.append(nn.Conv2d(features[i], features[i + 1], dim, padding='same'))
            self.pool_list.append(nn.AvgPool2d(pooling[i]))
        
        self.readout = nn.Linear(features[-1] * 7*7, outputs)
        self.softmax = nn.Softmax(dim = 1)

        self.active = nn.ReLU()
        
    def forward(self, x):
        for convo, pool in zip(self.convo_list, self.pool_list):
            x = pool(self.active(convo(x)))
        x = torch.flatten(x, start_dim = 1)
        x = self.readout(x)
        x = self.softmax(x)
        return x

# Constants
GRID_SIZE = 28
CELL_SIZE = 10
MODEL = torch.load('mnist_classifier.pth')
MODEL.eval()

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
                    self.matrix[y_i][x_i] = 0.95
                elif not self.matrix[y_i][x_i]: #don't override previous 1s
                    self.matrix[y_i][x_i] = 0.7
        self.canvas.create_rectangle(x1, y1, x2, y2, fill='gray70', outline='gray70')
        for x_i in range(max(0, (x - half_width)), min(GRID_SIZE, (x + half_width + 1))):
            for y_i in range(max(0, (y - half_width)), min(GRID_SIZE, (y + half_width + 1))):
                cell_x = x_i * CELL_SIZE
                cell_y = y_i * CELL_SIZE
                if self.matrix[y_i][x_i] == 0.95:
                    self.canvas.create_rectangle(cell_x, cell_y, 
                                                 cell_x + CELL_SIZE, cell_y + CELL_SIZE, 
                                                 fill='white', outline='white')
            

    def clear(self):
        self.canvas.delete('all')
        self.matrix = torch.Tensor(np.array([np.array([0 for _ in range(GRID_SIZE)]) for _ in range(GRID_SIZE)]))
        self.white_units = []
            
    def id(self):
        image = self.matrix.unsqueeze(0).unsqueeze(0)
        prediction = get_prediction(MODEL.forward(image)).item()
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