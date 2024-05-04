import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import ImageTk, Image, ImageDraw
import mysql.connector


import recognition_screen


class LoginWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Login Form")
        self.geometry("800x700")
        self.resizable(0, 0)
        self['background'] = 'gray6'
        self.attributes('-alpha', 0.9)
        self.attributes('-topmost', True)

        def login():
            username = "admin"
            password = "12345"
            if username_entry.get() == username and password_entry.get() == password:
                messagebox.showinfo(title="Login Success",
                                    message="You successfully logged in.")
                self.destroy()
                recognition_screen.RecognitionWindow()
            else:
                messagebox.showerror(title="Error", message="Invalid login.")

        frame = tk.Frame(self, bg='gray6')

        login_label = tk.Label(frame, text="Login", bg='gray6',
                               fg="dark slate gray", font=("Arial", 30))
        username_label = tk.Label(frame, text="Username",
                                  bg='gray6', fg="#FFFFFF", font=("Arial", 16))
        username_entry = tk.Entry(frame, font=("Arial", 16))
        password_entry = tk.Entry(frame, show="*", font=("Arial", 16))
        password_label = tk.Label(frame, text="Password",
                                  bg='gray6', fg="#FFFFFF", font=("Arial", 16))

        style = ttk.Style()
        style.configure('TButton', borderwidth=0, relief='solid',
                        padding=10, background='black', foreground='black')
        style.configure('Rounded.TButton', borderwidth=0, relief='flat', padding=5, background='black',
                        foreground='black', bordercolor='gray8', focuscolor='none', highlightthickness=0, borderradius=15)

        login_button = ttk.Button(
            frame, text="Login", command=login, style='Rounded.TButton')

        login_label.grid(row=0, column=0, columnspan=2, sticky="news", pady=40)
        username_label.grid(row=1, column=0)
        username_entry.grid(row=1, column=1, pady=20)
        password_label.grid(row=2, column=0)
        password_entry.grid(row=2, column=1, pady=20)
        login_button.grid(row=3, column=0, columnspan=2, pady=30)

        frame.pack()
        self.mainloop()
