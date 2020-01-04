from tkinter import *
 
window = Tk()
 
window.title("Welcome to LikeGeeks app")
 

lbl = Label(window, text="Visual Symbol Evaluation", font=("Arial Bold", 50))

def clicked():
 
    lbl.configure(text="Button was clicked !!")
lbl.grid(column=0, row=0)
btn_train = Button(window, text="Train Model", bg="orange", fg="red", command=clicked)
 
btn_train.grid(column=0, row=10)



window.geometry('800x600')





window.mainloop()

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:24:49 2019

@author: Alexander Hofmüller
"""

