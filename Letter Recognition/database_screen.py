import tkinter as tk
from tkinter import ttk
import mysql.connector

import recognition_screen


class DatabaseWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Database Window")
        self.geometry("800x700")
        self.resizable(0, 0)
        self['background'] = 'gray6'
        self.attributes('-alpha', 0.9)
        self.attributes('-topmost', True)

        self.label_records = tk.Label(
            self, text="Records", font=('Helvetica', 16, 'bold'), bg='gray6', fg='dark slate gray')
        self.label_records.pack(pady=20)

        self.records_text = tk.Text(
            self, width=70, height=23, bg='gray8', fg='white')
        self.records_text.pack(pady=10)

        style = ttk.Style()
        style.configure('TButton', borderwidth=0, relief='solid',
                        padding=10, background='black', foreground='black')
        style.configure('Rounded.TButton', borderwidth=0, relief='flat', padding=5, background='black',
                        foreground='black', bordercolor='gray8', focuscolor='none', highlightthickness=0, borderradius=15)

        self.btn_save = ttk.Button(
            text="Back", command=self.move_back, style='Rounded.TButton')
        self.btn_save.place(x=20, y=20)

        self.btn_database = ttk.Button(
            text="Delete", command=self.delete_records, style='Rounded.TButton')
        self.btn_database.place(x=20, y=587)

        self.btn_predict = ttk.Button(
            text="Exit", command=self.destroy, style='Rounded.TButton')
        self.btn_predict.place(x=20, y=650)

        self.show_records()

    def show_records(self):
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='shaheer',
            database='new_schema'
        )

        cursor = conn.cursor()
        select_query = "SELECT label, probability FROM recog"
        cursor.execute(select_query)
        records = cursor.fetchall()

        for record in records:
            label, probability = record
            self.records_text.insert(tk.END, f"Label {label}\n")
            self.records_text.insert(tk.END, f"Probability {probability}\n")
            self.records_text.insert(tk.END, "-" * 20 + "\n")

        cursor.close()
        conn.close()

    def delete_records(self):
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='shaheer',
            database='new_schema'
        )

        cursor = conn.cursor()
        delete_query = "DELETE FROM recog"
        cursor.execute(delete_query)
        conn.commit()
        cursor.close()
        conn.close()

        self.records_text.delete(1.0, tk.END)
        self.records_text.insert(tk.END, "Records deleted.")

    def move_back(self):
        self.destroy()
        recog_window = recognition_screen.RecognitionWindow()
        recog_window.mainloop()


if __name__ == "__main__":
    db_window = DatabaseWindow()
    db_window.mainloop()
