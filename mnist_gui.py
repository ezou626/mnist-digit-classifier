import tkinter as tk
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw
import io
from torch import Tensor
from torch.nn import AvgPool2d
from standard_cnn import *
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

# color constants
WHITE = 1
BLACK = 0

# load model
MODEL = torch.load('mnist_classifier.pth')
MODEL.eval()

# transform_image
transform: transforms.Compose = transforms.Compose([
    transforms.Normalize((0.1307,), (0.3081,))
])


class PaintApp:

    def __init__(self, root: tk.Tk, cell_size: int):
        self.root = root

        self.CELL_SIZE = cell_size
        self.WIDTH = int(cell_size * 2.5)

        self.canvas = tk.Canvas(self.root,
                                bg='black',
                                width=28 * self.CELL_SIZE,
                                height=28 * self.CELL_SIZE)
        self.canvas.pack()

        self.clear_button = tk.Button(self.root, text='Clear', command=self.clear)
        self.clear_button.pack()

        self.image: Image.Image = Image.new('I', (28 * self.CELL_SIZE, 28 * self.CELL_SIZE), BLACK)
        self.drawer = ImageDraw.Draw(self.image)

        self.canvas.bind('<B1-Motion>', self.paint)

        # self.save_button = tk.Button(self.root, text='Save', command=self.save)
        # self.save_button.pack()

        self.classify_button = tk.Button(self.root, text='Classify', command=self.classify)
        self.classify_button.pack()

        # response label
        self.label = tk.Label(root, text='Click the Identify button to have your digit identified.')
        self.label.pack()

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

    # def save(self):
    #     '''
    #     Save image (testing only)
    #     '''
    #     self.image.save('./test.bmp')
    
    def classify(self):
        '''
        Classify number
        '''
        # Comment out if you don't want to see differences between sampling modes
        self.visualize_downsampling()

        original_image = Tensor(np.asarray(self.image).copy())
        original_image = Image.fromarray(original_image.numpy())

        downsampled_image = F.resize(original_image, [28, 28], interpolation=F.InterpolationMode.BICUBIC)
        downsampled_image = torch.Tensor(np.array(downsampled_image)).unsqueeze(0)
        normed_image = transform(downsampled_image)
        self.label.config(text='Classifying...')
        prediction = torch.argmax(MODEL.forward(normed_image.unsqueeze(0))).item()
        self.label.config(text=f'Your number is {prediction}. Clear to try another!')

    def visualize_downsampling(self):
        '''
        Visualize bicubic interp vs average pooling (previous)
        '''
        original_image = Tensor(np.asarray(self.image).copy())
        avgpool = nn.AvgPool2d(kernel_size=(10, 10))
        pooled_image = avgpool(original_image.unsqueeze(0).unsqueeze(0))
        pooled_image = pooled_image.squeeze(0).squeeze(0).numpy()

        original_image = Image.fromarray(original_image.numpy())

        bicubic_image = F.resize(original_image, [28, 28], interpolation=F.InterpolationMode.BICUBIC)
        bicubic_image = np.array(bicubic_image)

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(pooled_image, cmap='gray')
        axes[0].set_title('Average Pooling')
        axes[1].imshow(bicubic_image, cmap='gray')
        axes[1].set_title('Bicubic Interpolation')
        plt.show()


if __name__ == '__main__':
    root = tk.Tk()

    PaintApp(root, 10)

    root.mainloop()
