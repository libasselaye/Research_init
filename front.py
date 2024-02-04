#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:07:21 2020

@author: macbookair
"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

class AnalyseWindow(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def validate_user(self):
        user = self.ids.username_field
        pwd = self.ids.pwd_field
        info = self.ids.info
        
        uname = user.text
        password = pwd.text
        
        if uname =='' or password == '':
            info.text ='[color=#FF0000]username and or password required[/color]' 
        else:
            if uname == 'admin' and password == 'admin':
                info.text ='[color=#00FF00] Logged successfull!![/color]' 
            else:
                info.text ='[color=#FF0000]Invalid username and/or password[/color]'
                
class AnalyseApp(App):
    def build(self):
        return AnalyseWindow()

if __name__ == "__main__":
    sa = AnalyseApp()
    sa.run()
