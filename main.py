from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import pandas as pd
import numpy as np

# http://www.retrosheet.org/gamelogs/index.html

df = pd.read_csv('hoge1.csv', header=None, dtype='object')
tl = len(df.index)


def to_int(d):
    if d.isdigit():
        return int(d)
    else:
        return 0


to_csv = []

for n in range(tl):
    visitor7 = sum(map(to_int, [x for x in str(df[19][n][0:7])]))
    visitor = sum(map(to_int, [x for x in str(df[19][n])]))
    home7 = sum(map(to_int, [x for x in str(df[20][n])][0:7]))
    home = sum(map(to_int, [x for x in str(df[20][n])]))
    if home != visitor:
        if home > visitor:
            win = 0
        else:
            win = 1

        to_csv.append([win, home7, visitor7])

df = pd.DataFrame(to_csv)
# pd.read_csv('unko.csv', header=None)
cls = df[0]
features = df.loc[:, 1:3]

lr = LogisticRegression()
lr.fit(features, cls)

# 確率を確認
print("0-0 : {0}".format(lr.predict_proba([[0, 0]])))
print("3-2 : {0}".format(lr.predict_proba([[3, 2]])))
print("2-3 : {0}".format(lr.predict_proba([[2, 3]])))
print("1-9 : {0}".format(lr.predict_proba([[1, 9]])))
print("3-3 : {0}".format(lr.predict_proba([[3, 3]])))
print("4-3 : {0}".format(lr.predict_proba([[4, 3]])))

"""
# 結果
0-0 : [[ 0.52888896  0.47111104]]
3-2 : [[ 0.73751883  0.26248117]]
2-3 : [[ 0.27074463  0.72925537]]
1-9 : [[  2.83146993e-04   9.99716853e-01]]
3-3 : [[ 0.50055542  0.49944458]]
4-3 : [[ 0.73013178  0.26986822]]
"""

hist = np.histogram(cls, bins=[0, 1, 2])[0]
win_home = hist[0]
win_visi = hist[1]
total = win_home + win_visi
home_result = win_home / total
vis_result = win_visi / total
print(home_result)
print(vis_result)
"""
HOME
0.530284301607

VISIT
0.469715698393
"""

joblib.dump(lr, 'mlb_stats.pkl', compress=9)

clf = joblib.load('mlb_stats.pkl')

print("0-0 : {0}".format(clf.predict_proba([[0, 0]])))
