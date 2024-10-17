import pandas as pd
import subprocess
from tqdm import tqdm

days = list(range(2,5)) #already did 1 manually

for day in days:
    print('Starting day: May ', day)
    if day<10:
        filename = 'ydata-fp-td-clicks-v1_0.2009050' + str(day)
    else:
        filename = 'ydata-fp-td-clicks-v1_0.20090510'
    date = filename.split('.')[1]

    subprocess.run('gzip -d {file}.gz'.format(file = filename), shell = True)

    pool_dict = {}

    with open(filename, 'r') as f:
        for e, line in enumerate(f):
            key = line[93:]
            articles = [s[:6] for s in key.split('|')]
            pool = tuple(sorted(articles))
            if pool not in pool_dict:
                pool_dict[pool] = {}
                pool_dict[pool]['start'] = e
                pool_dict[pool]['end'] = e
            else:
                pool_dict[pool]['end'] += 1

    for e,key in enumerate(tqdm(pool_dict)):
        subprocess.run('tail -n +{start} {file} | head -n {num} > data/{day}_{idx}.txt'.format(start = pool_dict[key]['start']+1, file = filename, num = pool_dict[key]['end'] - pool_dict[key]['start']+1, day = date, idx =  e), shell = True)


    length = subprocess.check_output('wc -l {file}'.format(file = filename), shell = True)
    length = str(length)
    length = length[length.index('b')+2:length.index(' ')]
    length = int(length)

    counter = 0
    for key in pool_dict:
        counter += (pool_dict[key]['end'] + 1 - pool_dict[key]['start'])

    assert(counter == length)

    subprocess.run('rm {file}'.format(file = filename), shell = True)
# pool_set = set()

# for key in pool_dict:
#     articles = [s[:6] for s in key.split('|')]
#     art_tup = tuple(sorted(articles))
#     if art_tup not in pool_set:
#         pool_set.add(art_tup)
#     else:
#         print('WARNING')




