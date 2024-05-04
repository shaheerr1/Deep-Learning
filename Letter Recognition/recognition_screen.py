import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import numpy as np
import mysql.connector


import cv2
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

import login_screen
import database_screen


class RecognitionWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Recognition Window")
        self.geometry("800x700")
        self.resizable(0, 0)
        self['background'] = 'gray6'

        self.canvas = tk.Canvas(self, width=800, height=500, bg='gray8')
        self.canvas.grid(row=0, column=2, columnspan=2, sticky='e',)
        self.canvas.bind('<B1-Motion>', self.event_function, )

        self.label_status = tk.Label(
            self, text='Predicted Value - None', bg='white', font=('Helvetica', 16, 'bold'), fg='black')
        self.label_status.place(x=20, y=600)

        style = ttk.Style()
        style.configure('TButton', borderwidth=0, relief='solid',
                        padding=10, background='black', foreground='black')
        style.configure('Rounded.TButton', borderwidth=0, relief='flat', padding=5, background='black',
                        foreground='black', bordercolor='gray8', focuscolor='none', highlightthickness=0, borderradius=15)

        self.btn_save = ttk.Button(
            text="Back", command=self.move_back, style='Rounded.TButton')
        self.btn_save.place(x=20, y=50)

        self.btn_database = ttk.Button(
            text="Records", command=self.show_records, style='Rounded.TButton')
        self.btn_database.place(x=20, y=150)

        self.btn_predict = ttk.Button(
            text="Predict", command=self.predict, style='Rounded.TButton')
        self.btn_predict.place(x=20, y=300)

        self.btn_clear = ttk.Button(
            text="Clear", command=self.clear, style='Rounded.TButton')
        self.btn_clear.place(x=20, y=370)

        self.btn_exit = ttk.Button(
            self, text='Exit', command=self.destroy, style='Rounded.TButton')
        self.btn_exit.place(x=20, y=440)

        self.img = Image.new('RGB', (700, 700), (0, 0, 0))
        self.img_draw = ImageDraw.Draw(self.img)

        self.attributes('-alpha', 0.9)
        self.attributes('-topmost', True)

    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (5, 5), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())
    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dense(36, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    model.load_weights("C:/Datasets/best_val_loss_model.h5")

    def event_function(self, event):
        x = event.x
        y = event.y

        x1 = x - 20
        y1 = y - 20

        x2 = x + 20
        y2 = y + 20

        self.canvas.create_oval((x1, y1, x2, y2), fill='black')
        self.img_draw.ellipse((x1, y1, x2, y2), fill='white')

    def clear(self):
        self.canvas.delete('all')
        self.img = Image.new('RGB', (700, 500), (0, 0, 0))
        self.img_draw = ImageDraw.Draw(self.img)

        self.label_status.config(text='Predictio : None')

    def save_prediction(self, predictions):
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='shaheer',
            database='new_schema')

        cursor = conn.cursor()
        insert_query = "INSERT INTO recog (label, probability) VALUES (%s, %s)"
        for label, probability in predictions.items():
            cursor.execute(insert_query, (label, float(probability)))
        conn.commit()
        cursor.close()
        conn.close()

    def predict(self):
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                  'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        min_x, min_y, max_x, max_y = self.canvas.bbox('all')
        drawn_image = self.img.crop((min_x, min_y, max_x, max_y))
        drawn_image = drawn_image.resize((28, 28))
        drawn_image = drawn_image.convert('L')
        image_array = np.array(drawn_image) / 255.0
        image_array = np.reshape(
            image_array, (1, image_array.shape[0], image_array.shape[1], 1))
        prediction = self.model.predict(image_array)
        best_predictions = {}
        record_predictions = {}

        for i in range(3):
            max_i = np.argmax(prediction[0])
            acc = round(prediction[0][max_i], 1)
            if acc > 0:
                label = labels[max_i]
                best_predictions[label] = acc
                prediction[0][max_i] = 0
            else:
                break

        self.label_status.config(
            text='Predicted Value - ' + str(best_predictions))
        self.save_prediction(best_predictions)

    def show_records(self):
        self.destroy()
        db_window = database_screen.DatabaseWindow()
        db_window.mainloop()

    def move_back(self):
        self.destroy()
        login_window = login_screen.LoginWindow()
        login_window.mainloop()


if __name__ == "__main__":
    recognition_window = RecognitionWindow()
    recognition_window.mainloop()
