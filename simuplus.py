import numpy as np
import pandas as pd
from pathlib import Path
import TN
import loaddata as ld
import random


standard_speed = 60 #km/h
t = 1 #min
T = 30 #min
csv_net_path = 'data/SF/SiouxFalls_net.csv'
csv_od_path = 'data/SF/SiouxFalls_od.csv'
num_nodes = 24