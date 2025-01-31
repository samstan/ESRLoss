import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

days = np.arange(1,11)

filenames = ['day{num}.csv'.format(num = day) for day in days]

df_all = pd.DataFrame(columns = ['Filename' , 'Val_MSE','Val_Our', 'Day'])

for e,file in enumerate(filenames):
    df = pd.read_csv(file)
    df['Day'] = e+1
    df_all = pd.concat([df_all, df])

df_agg = df_all.groupby(['Day']).agg(['mean', 'std', 'count'])
df_agg.columns = [ '_'.join(str(i) for i in col) for col in df_agg.columns]
df_agg.reset_index(inplace=True)

df_agg['Val_MSE_halfwidth'] = df_agg['Val_MSE_std']*1.96/np.sqrt(df_agg['Val_MSE_count'])
df_agg['Val_Our_halfwidth'] = df_agg['Val_Our_std']*1.96/np.sqrt(df_agg['Val_Our_count'])


df_agg['Val_MSE_low'] = df_agg['Val_MSE_mean'] - df_agg['Val_MSE_halfwidth']
df_agg['Val_MSE_high'] = df_agg['Val_MSE_mean'] + df_agg['Val_MSE_halfwidth']

df_agg['Val_Our_low'] = df_agg['Val_Our_mean'] - df_agg['Val_Our_halfwidth']
df_agg['Val_Our_high'] = df_agg['Val_Our_mean'] + df_agg['Val_Our_halfwidth']


overall_MSE = df_all.Val_MSE.mean()
overall_Our = df_all.Val_Our.mean()
overall_MSE_half = df_all.Val_MSE.std()*1.96/np.sqrt(len(df_all))
overall_Our_half = df_all.Val_Our.std()*1.96/np.sqrt(len(df_all))

print(overall_MSE - overall_MSE_half, overall_MSE + overall_MSE_half)
print(overall_Our - overall_Our_half, overall_Our + overall_Our_half)

plt.errorbar(df_agg.Day, df_agg.Val_MSE_mean, yerr = df_agg.Val_MSE_halfwidth, label = 'MSE')
plt.errorbar(df_agg.Day, df_agg.Val_Our_mean, yerr = df_agg.Val_Our_halfwidth, label = 'Paired')
plt.legend()
plt.xlabel('Day')
plt.ylabel('CTR')
plt.title('CTR from Training on MSE vs. Paired Loss')

plt.show()

