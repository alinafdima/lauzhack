# -*- coding: utf-8 -*-
"""
@author: Alina Dima
"""

import socket

def get_machine():
    if socket.gethostname() == 'mbk-12-72':
        return 'alina'
    elif socket.gethostname() == 'VAZUS':
        return 'daniel'
    else:
        return ""

if get_machine() == 'alina':
    data_path = '/Users/alina/Documents/code/lauzhack/data/input'
elif get_machine() == 'daniel':
    data_path = '../../local-data/receipts'
