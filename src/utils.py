import sys
import os

import pickle

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from src.exception import CustomException
from src.logger import logging


