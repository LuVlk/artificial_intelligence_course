from keras.models import load_model
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np


class DigitClassifier():

    def __init__(self, model, img=None):
        if img is not None:
            self.__img = img

        self.__model = load_model(model)

    def predict(self, img=None):
        if img is not None:
            self.__img = img

        # resizing the img to 28 x 28 Pixels
        self.__img = self.__img.resize((28, 28))

        # convert rgb to grayscale
        self.__img = self.__img.convert('L')
        self.__img = np.array(self.__img)

        # reshaping to suppert model input and normalizing
        self.__img = self.__img.reshape(1, 28, 28, 1)
        self.__img = self.__img/255.0

        # predicting the digit
        res = self.__model.predict([self.__img])[0]

        # returning predicted digit and respective probability
        return np.argmax(res), max(res)


class App(tk.Tk):

    def __init__(self) -> None:
        tk.Tk.__init__(self)

        # loading the model
        self.__classifier = DigitClassifier('mnist.h5')

        # initial cursor position
        self.__x = self.__y = 0

        # Creating elements
        self.__canvas = tk.Canvas(
            self,
            width=300,
            height=300,
            bg='white',
            cursor='cross'
        )

        self.__label = tk.Label(
            self,
            text='Thinking..',
            font=('Helvetica', 48)
        )
        self.__classify_btn = tk.Button(
            self,
            text='Recognise',
            command=self.classify_handwriting
        )
        self.__button_clear = tk.Button(
            self,
            text='Clear',
            command=self.clear_all
        )

        # Grid structure
        self.__canvas.grid(row=0, column=0, pady=2, sticky=tk.W, )
        self.__label.grid(row=0, column=1, pady=2, padx=2)
        self.__button_clear.grid(row=1, column=0, pady=2)
        self.__classify_btn.grid(row=1, column=1, pady=2, padx=2)

        # adding an event callback
        self.__canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self) -> None:
        self.__canvas.delete("all")

    def classify_handwriting(self) -> None:
        HWND = self.__canvas.winfo_id()  # get the canvas handle
        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        a, b, c, d = rect
        rect = (a + 4, b + 4, c - 4, d - 4)
        img = ImageGrab.grab(rect)

        digit, acc = self.__classifier.predict(img)
        self.__label.configure(
            text=('My guess:\n' + str(digit) + ',  ' + str(int(acc*100)) + '%')
        )

    def draw_lines(self, event):
        self.__x = event.x
        self.__y = event.y
        self.__canvas.create_oval(
            self.__x - 8,
            self.__y - 8,
            self.__x + 8,
            self.__y + 8,
            fill='black'
        )


def main():
    app = App()
    tk.mainloop()


if __name__ == '__main__':
    main()
