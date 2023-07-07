import tkinter as tk
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw
import io

'''
Prototyping app to create a better drawing brush for the real thing
'''

#color constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class PaintApp:
    
    def __init__(self, root: tk.Tk, cell_size: int):
        
        self.root = root
        
        self.CELL_SIZE = cell_size
        self.WIDTH = 24
        
        self.canvas = tk.Canvas(self.root, 
                                bg = 'black', 
                                width = 28 * self.CELL_SIZE, 
                                height = 28 * self.CELL_SIZE)
        self.canvas.pack()
        
        self.clear_button = tk.Button(self.root, text='Clear', command=self.clear)
        self.clear_button.pack()
        
        self.image: Image.Image = Image.new('RGB', (28 * self.CELL_SIZE, 28 * self.CELL_SIZE), BLACK)
        self.drawer = ImageDraw.Draw(self.image)
        
        self.canvas.bind('<B1-Motion>', self.paint)
        
        self.save_button = tk.Button(self.root, text='Save', command=self.save)
        self.save_button.pack()
    
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
    
    def save(self):
        self.image.save('./test.bmp')

if __name__ == '__main__':
    
    root = tk.Tk()
    
    PaintApp(root, 10)
    
    root.mainloop()