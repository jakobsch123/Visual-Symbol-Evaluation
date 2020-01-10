# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:57:54 2019

@author: jakob
"""
import eel

eel.init('web')

@eel.expose
def my_python_method(pic):
	pic = pic * -1
	return pic

@eel.expose                         # Expose this function to Javascript
def say_hello_py(x):
    print('Hello from %s' % x)

say_hello_py('Python World!')
#eel.say_hello_js('Python World!')   # Call a Javascript function

eel.start('index.php', size=(650, 612))
