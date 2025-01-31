import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from learners_discrete import *

day_dict={1:55, 2:37, 3:32, 4:48, 5:49, 6:48, 7:51, 8:49, 9:39, 10:34}
np.random.seed(2024)
torch.manual_seed(0)
DAY = 1
K = 5
print('DAY:', DAY)
NUM_EXPS = day_dict[DAY]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

flagged = set()

def parse_line(line, skip = False):
    feat1 = float(line[line.find('2:')+2:line.find('3:')-1])
    feat2 = float(line[line.find('3:')+2:line.find('4:')-1])
    feat3 = float(line[line.find('4:')+2:line.find('5:')-1])
    feat4 = float(line[line.find('5:')+2:line.find('6:')-1])
    feat5 = float(line[line.find('6:')+2:line.find('1:')-1])
    split = line.split(' ')
    if skip:
        article = split[1]
        if split[1] in flagged:
            return False,'', '', '', '', '', ''
        clicked = split[2] == '1'
        return article, clicked, feat1,feat2,feat3,feat4,feat5
    else:
        article = split[0]
        return article, feat1,feat2,feat3,feat4,feat5
    

def parse_articles(line):
    articles = line.split('|')
    dicto = {'Article': [], 'Feat 1': [], 'Feat 2': [], 'Feat 3': [], 'Feat 4': [], 'Feat 5': []}
    for article in articles[2:]:
        if '7:' in article:
            flagged.add(article[:6])
            continue
        article, feat1, feat2, feat3, feat4, feat5 = parse_line(article)
        dicto['Article'].append(article) 
        dicto['Feat 1'].append(feat1)
        dicto['Feat 2'].append(feat2)
        dicto['Feat 3'].append(feat3)
        dicto['Feat 4'].append(feat4)
        dicto['Feat 5'].append(feat5)
    return pd.DataFrame(dicto)


filename_prefix = 'data/2009050{date}_{idx}.txt' #change for day 10

filenames = [filename_prefix.format(date = DAY, idx = i) for i in range(NUM_EXPS)]

vals_our = np.zeros(NUM_EXPS)
vals_s = np.zeros(NUM_EXPS)
vals_i = np.zeros(NUM_EXPS)
vals_b1 = np.zeros(NUM_EXPS)
vals_b2 = np.zeros(NUM_EXPS)
vals_b3 = np.zeros(NUM_EXPS)
vals_b4 = np.zeros(NUM_EXPS)
vals_b5 = np.zeros(NUM_EXPS)
vals_p1 = np.zeros(NUM_EXPS)
vals_p2 = np.zeros(NUM_EXPS)
vals_p3 = np.zeros(NUM_EXPS)
vals_p4 = np.zeros(NUM_EXPS)

for enum,filename in enumerate(tqdm(filenames)):

    df_dict = {'Article': [], 'Clicked': [], 'Feat 1': [], 'Feat 2': [], 'Feat 3': [], 'Feat 4': [], 'Feat 5': []}

    with open(filename, 'r') as f:
        for e, line in enumerate(f):
            if e == 0:
                df_art = parse_articles(line)
            article, clicked, feat1,feat2,feat3,feat4,feat5 = parse_line(line, True)
            if article:
                df_dict['Article'].append(article)
                df_dict['Clicked'].append(clicked)
                df_dict['Feat 1'].append(feat1)
                df_dict['Feat 2'].append(feat2)
                df_dict['Feat 3'].append(feat3)
                df_dict['Feat 4'].append(feat4)
                df_dict['Feat 5'].append(feat5)

    df = pd.DataFrame(df_dict)

    arts = df.Article.unique()

    df = df[df.Article.isin(arts)]

    df['Treat'] = df['Article'].apply(lambda d: list(arts).index(d))
    df = df.drop('Article', axis = 1)

    counts = df.groupby(['Feat 1', 'Feat 2', 'Feat 3', 'Feat 4', 'Feat 5']).agg(['sum', 'count'])
    counts.columns = [ '_'.join(str(i) for i in col) for col in counts.columns]
    counts.reset_index(inplace=True)

    users_train, users_test = train_test_split(counts, test_size = 0.3)

    df_train = df.merge(users_train, on = ['Feat 1', 'Feat 2', 'Feat 3', 'Feat 4', 'Feat 5'], how = 'inner')

    df_test = df.merge(users_test, on = ['Feat 1', 'Feat 2', 'Feat 3', 'Feat 4', 'Feat 5'], how = 'inner')

    y_train = torch.tensor(df_train.Clicked.astype(np.float64).values)

    X_train = torch.tensor(df_train.drop(['Clicked', 'Clicked_sum', 'Clicked_count', 'Treat_sum', 'Treat_count', 'Treat'], axis = 1).values)

    treats_train = torch.tensor(df_train.Treat.astype(np.float64).values)

    y_test = torch.tensor(df_test.Clicked.astype(np.float64).values)

    df_test = df_test.drop(['Clicked_sum', 'Clicked_count', 'Treat_sum', 'Treat_count'], axis = 1)

    df_test_all = df_test.drop(['Clicked', 'Treat'], axis = 1).drop_duplicates()

    df_tests_all = []

    for i in treats_train.unique():
        tmp = df_test_all.copy()
        tmp['Treat'] = i.item()
        df_tests_all.append(tmp)

    df_tests_all = pd.concat(df_tests_all, ignore_index = True)

    vals_our[enum] = reg_learner(X_train, treats_train, y_train, df_test, df_tests_all, k = K, schedule_k = 'increase')
    vals_s[enum] = s_learner(X_train, treats_train, y_train, df_test, df_tests_all)
    vals_i[enum] = bandit_net(X_train, treats_train, y_train, df_test, 0)
    vals_b1[enum] = bandit_net(X_train, treats_train, y_train, df_test, 0.65)
    vals_b2[enum] = bandit_net(X_train, treats_train, y_train, df_test, 0.75)
    vals_b3[enum] = bandit_net(X_train, treats_train, y_train, df_test, 0.85)
    vals_b4[enum] = bandit_net(X_train, treats_train, y_train, df_test, 0.95)
    vals_b5[enum] = bandit_net(X_train, treats_train, y_train, df_test, 1.05)
    vals_p1[enum] = pseudo_loss(X_train, treats_train, y_train, df_test, 1e-3)
    vals_p2[enum] = pseudo_loss(X_train, treats_train, y_train, df_test, 1e-2)
    vals_p3[enum] = pseudo_loss(X_train, treats_train, y_train, df_test, 1e-1)
    vals_p4[enum] = pseudo_loss(X_train, treats_train, y_train, df_test, 1)
    # For supplement
    # vals_ls[enum] = log_smooth(X_train, treats_train, y_train, df_test, 1/np.sqrt(len(y_train)))
    # vals_lse[enum] = log_sum_exp(X_train, treats_train, y_train, df_test, 1/np.sqrt(len(y_train)))
    # vals_dr_0[enum] = dr_shrink(X_train, treats_train, y_train, df_test, 0)
    # vals_dr_01[enum] = dr_shrink(X_train, treats_train, y_train, df_test, 0.1)
    # vals_dr_1[enum] = dr_shrink(X_train, treats_train, y_train, df_test, 1)
    # vals_dr_10[enum] = dr_shrink(X_train, treats_train, y_train, df_test, 10)
    # vals_dr_100[enum] = dr_shrink(X_train, treats_train, y_train, df_test, 100)
    # vals_dr_1000[enum] = dr_shrink(X_train, treats_train, y_train, df_test, 1000)
    # vals_dr_inf[enum] = dr_shrink(X_train, treats_train, y_train, df_test, torch.inf)

results_dict = {'Filename': [], 'Val_Our': [], 'Val_S': [] 'Val_IPW': [], 'Val_Bandit_65': [], 'Val_Bandit_75': [], 'Val_Bandit_85': [], 'Val_Bandit_95': [], 'Val_Bandit_105': [], 'Val_PL_e3': [], 'Val_PL_e2': [], 'Val_PL_e1': [], 'Val_PL_e0': []}

for i in range(len(filenames)):
    results_dict['Filename'].append(filenames[i])
    results_dict['Val_Our'].append(vals_our[i])
    results_dict['Val_S'].append(vals_s[i])
    results_dict['Val_IPW'].append(vals_i[i])
    results_dict['Val_Bandit_65'].append(vals_b1[i])
    results_dict['Val_Bandit_75'].append(vals_b2[i])
    results_dict['Val_Bandit_85'].append(vals_b3[i])
    results_dict['Val_Bandit_95'].append(vals_b4[i])
    results_dict['Val_Bandit_105'].append(vals_b5[i])
    results_dict['Val_PL_e3'].append(vals_p1[i])
    results_dict['Val_PL_e2'].append(vals_p2[i])
    results_dict['Val_PL_e1'].append(vals_p3[i])
    results_dict['Val_PL_e0'].append(vals_p4[i])

df_results = pd.DataFrame(results_dict)

df_results.to_csv('day{date}.csv'.format(date = DAY, k = K), index = False)
