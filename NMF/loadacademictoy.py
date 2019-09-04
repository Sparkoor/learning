import pickle
import re
from six.moves import cPickle
import sys

path = r'D:\work\learning\NMF\datasets\academic_toy.pickle'

data = pickle.load(open(path, 'rb'), encoding='iso-8859-1')
