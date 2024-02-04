#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:07:21 2020

@author: macbookair
"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

from front import AnalyseWindow
from operator.operator import OperatorWindow

class MainWindow(BoxLayout):
    front_widget = AnalyseWindow()
    operator_widget = OperatorWindow()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
                
class MainApp(App):
    def build(self):
        return MainWindow()

if __name__ == "__main__":
    sa = MainApp()
    sa.run()
