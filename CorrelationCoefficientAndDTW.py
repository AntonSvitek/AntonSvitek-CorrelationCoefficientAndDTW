
pip install dtw-python

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path_file = '/content/Data7.csv'

data = pd.read_csv(path_file, index_col = 0)

# Выведем первые 5 строк таблицы
display(data.head())

# Выведем графики
display(data.plot())

# Подсчитаем коэффициенты корреляции
correlation = data.corr()
display( sns.heatmap(correlation , annot = True) )

from dtw import *
import numpy as np


y_data_CostPerm = np.array( data['CostPerm'] , dtype=np.float64)
y_data_CountObjectPerm = np.array( data['CountObjectPerm'] , dtype=np.float64)
y_data_CostRussia = np.array( data['CostRussia'] , dtype=np.float64)

dtw(y_data_CostPerm, y_data_CountObjectPerm, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")).plot(type="twoway")
dtw(y_data_CostPerm, y_data_CostRussia, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")).plot(type="twoway")

one_and_two = dtw(y_data_CostPerm, y_data_CountObjectPerm, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")).distance
one_and_three = dtw(y_data_CostPerm, y_data_CostRussia, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")).distance

print('Значение dtw =',one_and_two, ' у 1 и 2'  )
print('Значение dtw =',one_and_three, ' у 1 и 3'  )