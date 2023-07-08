import tkinter as tk
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw
import io
from torch import Tensor
from torch.nn import AvgPool2d
from standard_cnn import *
from torchvision import transforms

'''
Prototyping app to create a better drawing brush for the real thing
'''

#color constants
WHITE = 1
BLACK = 0

#load model
MODEL = torch.load('mnist_classifier.pth')
MODEL.eval()

#transform_image
transform: transforms.Compose = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
        ])

class PaintApp:
    
    def __init__(self, root: tk.Tk, cell_size: int):
        
        self.root = root
        
        self.CELL_SIZE = cell_size
        self.WIDTH = int(cell_size * 2.5)
        
        self.canvas = tk.Canvas(self.root, 
                                bg = 'black', 
                                width = 28 * self.CELL_SIZE, 
                                height = 28 * self.CELL_SIZE)
        self.canvas.pack()
        
        self.clear_button = tk.Button(self.root, text='Clear', command=self.clear)
        self.clear_button.pack()
        
        self.image: Image.Image = Image.new('I', (28 * self.CELL_SIZE, 28 * self.CELL_SIZE), BLACK)
        self.drawer = ImageDraw.Draw(self.image)
        
        self.canvas.bind('<B1-Motion>', self.paint)
        
        self.save_button = tk.Button(self.root, text='Save', command=self.save)
        self.save_button.pack()
        
        self.classify_button = tk.Button(self.root, text='Classify', command=self.classify)
        self.classify_button.pack()
        
        #response label
        self.label = tk.Label(root, text='Click the Identify button to have your digit identified.')
        self.label.pack()
        
        self.POOL = AvgPool2d((self.CELL_SIZE, self.CELL_SIZE), (self.CELL_SIZE, self.CELL_SIZE))
        self.POOL.eval()
    
    def paint(self, event):
        '''
        Draw on screen (continuous circles)
        '''
        r = self.WIDTH
        x_1, y_1 = event.x, event.y
        x_2, y_2 = x_1 + r, y_1 + r
        self.canvas.create_oval(x_1, y_1, x_2, y_2, fill='white', outline='white')
        self.drawer.ellipse((x_1, y_1, x_2, y_2), fill=WHITE, outline=WHITE)
    
    def clear(self):
        '''
        Clear drawing
        '''
        self.canvas.delete("all")
        self.image: Image.Image = Image.new('I', (28 * self.CELL_SIZE, 28 * self.CELL_SIZE), BLACK)
        self.drawer = ImageDraw.Draw(self.image)
    
    def save(self):
        '''
        Save image (testing only)
        '''
        self.image.save('./test.bmp')
        
    def classify(self):
        '''
        Classify number
        '''
        original_image = Tensor(np.asarray(self.image).copy())
        downsampled_image = self.POOL(original_image.unsqueeze(0).unsqueeze(0))
        normed_image = transform(downsampled_image)
        self.label.config(text = f'Classifying...')
        prediction = get_prediction(MODEL.forward(normed_image)).item()
        self.label.config(text = f'Your number is {prediction}. Clear to try another!')

if __name__ == '__main__':
    
    root = tk.Tk()
    
    PaintApp(root, 10)
    
    root.mainloop()