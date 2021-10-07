import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import uuid


ts = datetime.now()
ts_list = []
for i in range(10080):
    ts +=timedelta(minutes=i)
    ts_list.append(ts)

#temperature generation

#For Uniform Distribution
temp_uniform = np.random.uniform(35.5,37.5,10080)

# name,age,device-id,timestamp,temperature,date,heartbeat,sleep hours,steps,height,weight

names_list = ["sri","krishna","hari","venkat","gopal","krithi","amrutha"]
age_list = [37,45,25,55,65,15,32]
device_id  = [str(uuid.uuid1()) for x in range(0,7)]
height = [1,1.55,1.70,1.60,1.42,1.30,1.65] #in metres
weight = [45,50,55,60,48,35,65] # in kg

ts_list = []
for j in range(0,7):
    for i in range(1440):
        ts +=timedelta(minutes=i)
        ts_list.append(ts)
    ts +=timedelta(days=j)

sri_heart_rate = np.random.logistic(loc=75,size=10080)
sri_sleep_list = np.random.uniform(6,10,size = 10080)
sri_steps = np.random.rayleigh(scale=5000, size = 10080)

sri_healthdata_dict = {'Name':names_list[0],'Age':age_list[0],'Device_Id':device_id[0],'Timestamp':ts_list,
'Temperature':temp_uniform,'HeartBeat':sri_heart_rate,'Sleep_hours':sri_sleep_list,'Steps':sri_steps,'Height':height[0],'Weight':weight[0]}

sri_df = pd.DataFrame(sri_healthdata_dict)

krishna_heart_rate = np.random.logistic(loc=72,size=10080)
krishna_sleep_list = np.random.uniform(6,10,size = 10080)
krishna_steps = np.random.rayleigh(scale=2000, size = 10080)

krishna_healthdata_dict = {'Name':names_list[1],'Age':age_list[1],'Device_Id':device_id[1],'Timestamp':ts_list,
'Temperature':temp_uniform,'HeartBeat':krishna_heart_rate,'Sleep_hours':krishna_sleep_list,'Steps':krishna_steps,'Height':height[1],'Weight':weight[1]}

krishna_df = pd.DataFrame(krishna_healthdata_dict)
hari_heart_rate = np.random.logistic(loc=64,size=10080)
hari_sleep_list = np.random.uniform(6,10,size = 10080)
hari_steps = np.random.rayleigh(scale=8000, size = 10080)

hari_healthdata_dict = {'Name':names_list[2],'Age':age_list[2],'Device_Id':device_id[2],'Timestamp':ts_list,
'Temperature':temp_uniform,'HeartBeat':hari_heart_rate,'Sleep_hours':hari_sleep_list,'Steps':hari_steps,'Height':height[2],'Weight':weight[2]}

hari_df = pd.DataFrame(hari_healthdata_dict)

venkat_heart_rate = np.random.logistic(loc=76,size=10080)
venkat_sleep_list = np.random.uniform(6,10,size = 10080)
venkat_steps = np.random.rayleigh(scale=5000, size = 10080)

venkat_healthdata_dict = {'Name':names_list[3],'Age':age_list[3],'Device_Id':device_id[3],'Timestamp':ts_list,
'Temperature':temp_uniform,'HeartBeat':venkat_heart_rate,'Sleep_hours':venkat_sleep_list,'Steps':venkat_steps,'Height':height[3],'Weight':weight[3]}

venkat_df = pd.DataFrame(venkat_healthdata_dict)

amrutha_heart_rate = np.random.logistic(loc=61.5,size=10080)
amrutha_sleep_list = np.random.uniform(6,10,size = 10080)
amrutha_steps = np.random.rayleigh(scale=8000, size = 10080)

amrutha_healthdata_dict = {'Name':names_list[6],'Age':age_list[6],'Device_Id':device_id[6],'Timestamp':ts_list,
'Temperature':temp_uniform,'HeartBeat':amrutha_heart_rate,'Sleep_hours':amrutha_sleep_list,'Steps':amrutha_steps,'Height':height[6],'Weight':weight[6]}

amrutha_df = pd.DataFrame(amrutha_healthdata_dict)

krithi_heart_rate = np.random.logistic(loc=67.5,size=10080)
krithi_sleep_list = np.random.uniform(6,10,size = 10080)
krithi_steps = np.random.rayleigh(scale=5000, size = 10080)

krithi_healthdata_dict = {'Name':names_list[5],'Age':age_list[5],'Device_Id':device_id[5],'Timestamp':ts_list,
'Temperature':temp_uniform,'HeartBeat':krithi_heart_rate,'Sleep_hours':krithi_sleep_list,'Steps':krithi_steps,'Height':height[5],'Weight':weight[5]}

krithi_df = pd.DataFrame(krithi_healthdata_dict)

gopal_heart_rate = np.random.logistic(loc=77,size=10080)
gopal_sleep_list = np.random.uniform(6,10,size = 10080)
gopal_steps = np.random.rayleigh(scale=2000, size = 10080)

gopal_healthdata_dict = {'Name':names_list[4],'Age':age_list[4],'Device_Id':device_id[4],'Timestamp':ts_list,
'Temperature':temp_uniform,'HeartBeat':gopal_heart_rate,'Sleep_hours':gopal_sleep_list,'Steps':gopal_steps,'Height':height[4],'Weight':weight[4]}

gopal_df = pd.DataFrame(gopal_healthdata_dict)

df = pd.concat([sri_df,krishna_df,hari_df,venkat_df],axis = 0)
df = df.reset_index(drop=True)
df["Bmi"] = (df["Weight"]/(df["Height"] * df["Height"]))
df["Ideal_weight"] = 2.2 * df["Bmi"] + 3.5 * df["Bmi"] * (df["Height"]-1.5)
#write generated data to FitnessData CSV
df.to_csv('FitnessData.csv',index=False)

test_df = pd.concat([gopal_df,krithi_df,amrutha_df],axis = 0)
test_df = test_df.reset_index(drop=True)
test_df["Bmi"] = (test_df["Weight"]/(test_df["Height"] * test_df["Height"]))
test_df["Ideal_weight"] = 2.2 * test_df["Bmi"] + 3.5 * test_df["Bmi"] * (test_df["Height"]-1.5)
#Write generated data to FitnessTestData.csv
test_df.to_csv('FitnessTestData.csv',index=False)
print("Fitness Tracking  System Data Generated Successfully")

