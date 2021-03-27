from numpy.random import seed
seed(1017)
import tensorflow as tf
#tf.random.set_seed(1017)

import os
from glob import glob
from collections import OrderedDict

import mne
from mne.io import RawArray
from mne import read_evokeds, read_source_spaces, compute_covariance
from mne import channels, find_events, concatenate_raws
from mne import pick_types, viz, io, Epochs, create_info
from mne import pick_channels, concatenate_epochs
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_raw
from mne.time_frequency import tfr_morlet

import numpy as np
from numpy import genfromtxt

import pandas as pd
pd.options.display.precision = 4
pd.options.display.max_columns = None

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,12)

import keras
from keras import regularizers
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Flatten, Conv2D, MaxPooling2D, LSTM
from keras.layers import BatchNormalization, Conv3D, MaxPooling3D

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
