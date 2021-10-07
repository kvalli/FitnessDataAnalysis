import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("FitnessData.csv")

plt.xlabel("Age")
plt.hist(df["Age"])
plt.show()

sleep = []
# plot Sleep hours
names = ['sri', 'krishna', 'hari','venkat']
for name in names:
    mean_sleep = df[df.loc[:,'Name'] ==name]
    sleep.append(np.mean(mean_sleep["Sleep_hours"]))

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(names, sleep)
plt.subplot(132)
plt.scatter(names, sleep)
plt.subplot(133)
plt.plot(names, sleep)
plt.suptitle('Sleep Categorical Plotting')
plt.show()

#plot heart rate
heart_beat = []
# plot HeartBeat hours
for name in names:
    mean_heart_beat = df[df.loc[:,'Name'] ==name]
    heart_beat.append(np.mean(mean_heart_beat["HeartBeat"]))

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(names, heart_beat)
plt.subplot(132)
plt.scatter(names, heart_beat)
plt.subplot(133)
plt.plot(names, heart_beat)
plt.suptitle('HeartBeat Categorical Plotting')
plt.show()

Steps = []
for name in names:
    mean_steps = df[df.loc[:,'Name'] == name]
    Steps.append(np.mean(mean_steps["Steps"]))

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(names, Steps)
plt.subplot(132)
plt.scatter(names, Steps)
plt.subplot(133)
plt.plot(names, Steps)
plt.suptitle('Steps Categorical Plotting')
plt.show()

#plot bmi 
bmi = []
for name in names:
    bmi_df = df[df.loc[:,'Name'] ==name]
    bmi.append(np.mean(bmi_df["Bmi"]))

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(names, bmi)
plt.subplot(132)
plt.scatter(names, bmi)
plt.subplot(133)
plt.plot(names, bmi)
plt.suptitle('Bmi Categorical Plotting')
plt.show()

#steps vs bmi
plt.plot(Steps,bmi)
plt.xlabel("Steps vs BMI")
plt.show()