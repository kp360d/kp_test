# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:43:51 2018

@author: kamalesh.pradhan
"""

import pandas as pd
import numpy as np

model_data = data_aftr_promo_1[~data_aftr_promo_1['ps_Grade'].isin(['Missing','Non Exempt'])]

