# -*- coding: utf-8 -*-
"""
@author: Alina Dima
"""

import socket

def get_machine():
    if socket.gethostname() == 'mbk-12-72':
        return 'alina'
    else:
        return 'daniel'

if get_machine() == 'alina':
    data_path = '/Users/alina/Documents/code/lauzhack/data/input'