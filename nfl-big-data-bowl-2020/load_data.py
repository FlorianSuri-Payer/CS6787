
# coding: utf-8

# In[83]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[84]:


data = pd.read_csv('train.csv', low_memory = False)


# In[85]:


data.head()


# In[103]:


data.shape[0]


# In[316]:


#split into training and labels
train = data.drop(['Yards'], axis=1).copy()
label = data['Yards'].copy()


# In[317]:


train.isna().sum().sort_values(ascending=False)


# In[318]:


#train['WindSpeed'].value_counts()


# In[319]:


#TODO
# clean up missing data
# preprocess features and standardize them
# encode categorical features
#https://www.kaggle.com/shahules/how-about-some-nn-keras-starter


# In[356]:


#WindDirection
# This function has been updated to reflect what Subin An (https://www.kaggle.com/subinium) mentioned in comments below.
# WindDirection is indicated by the direction that wind is flowing FROM - https://en.wikipedia.org/wiki/Wind_direction

def clean_wind_direction(wind_direction):
    wd = str(wind_direction).upper()
    if wd == 'N' or 'FROM N' in wd:
        return 'north'
    if wd == 'S' or 'FROM S' in wd:
        return 'south'
    if wd == 'W' or 'FROM W' in wd:
        return 'west'
    if wd == 'E' or 'FROM E' in wd:
        return 'east'
    
    if 'FROM SW' in wd or 'FROM SSW' in wd or 'FROM WSW' in wd:
        return 'south west'
    if 'FROM SE' in wd or 'FROM SSE' in wd or 'FROM ESE' in wd:
        return 'south east'
    if 'FROM NW' in wd or 'FROM NNW' in wd or 'FROM WNW' in wd:
        return 'north west'
    if 'FROM NE' in wd or 'FROM NNE' in wd or 'FROM ENE' in wd:
        return 'north east'
    
    if 'NW' in wd or 'NORTHWEST' in wd:
        return 'north west'
    if 'NE' in wd or 'NORTH EAST' in wd:
        return 'north east'
    if 'SW' in wd or 'SOUTHWEST' in wd:
        return 'south west'
    if 'SE' in wd or 'SOUTHEAST' in wd:
        return 'south east'

    return 'none'

train['WindDirection'] = train['WindDirection'].apply(clean_wind_direction)


# In[321]:


#Windspeed
def windspeed(x):
    x=str(x)
    if x.isdigit():
        return int(x)
    elif (x.isalpha()):
        return np.nan
    elif (x.isalnum()):
        return int(x.upper().split('M')[0])                             #return 12 incase of 12mp or 12 MPH
    elif '-' in x:
        return int((int(x.split('-')[0])+int(x.split('-')[1]))/2)   # return average windspeed incase of 11 - 20 etc..
    else:
        return np.nan
    

train['WindSpeed']=train['WindSpeed'].apply(windspeed)

train['WindSpeed'].fillna(train['WindSpeed'].median(),inplace=True)


# In[322]:


# Humidity and Temperature
train['Humidity'].astype('float32').fillna(method='ffill', inplace=True)
train['Temperature'].astype('float32').fillna(0, inplace=True)


# In[323]:


# Orientation, Direction, Defenders, Offense
print(np.nanmean(train['Orientation'].values))
na_map = {
    'Orientation': train['Orientation'].mean(),
    'Dir': train['Dir'].mean(),
    'DefendersInTheBox': (train['DefendersInTheBox'].median()),
    'OffenseFormation': 'blank'
}

train.fillna(na_map, inplace=True)
train['DefendersInTheBox'].isna().sum()


# In[324]:


#Field Position, undefined when yard line = 50
print(train['FieldPosition'].isna().sum())
train['FieldPosition'] = np.where(train['YardLine'] == 50, train['PossessionTeam'], train['FieldPosition'])
print(train['FieldPosition'].isna().sum())


# In[325]:


#Game weather
def group_game_weather(weather):
    rain = [
        'Rainy', 'Rain Chance 40%', 'Showers',
        'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',
        'Scattered Showers', 'Cloudy, Rain', 'Rain shower', 'Light Rain', 'Rain'
    ]
    overcast = [
        'Cloudy, light snow accumulating 1-3"', 'Party Cloudy', 'Cloudy, chance of rain',
        'Coudy', 'Cloudy, 50% change of rain', 'Rain likely, temps in low 40s.',
        'Cloudy and cold', 'Cloudy, fog started developing in 2nd quarter',
        'Partly Clouidy', '30% Chance of Rain', 'Mostly Coudy', 'Cloudy and Cool',
        'cloudy', 'Partly cloudy', 'Overcast', 'Hazy', 'Mostly cloudy', 'Mostly Cloudy',
        'Partly Cloudy', 'Cloudy'
    ]
    clear = [
        'Partly clear', 'Sunny and clear', 'Sun & clouds', 'Clear and Sunny',
        'Sunny and cold', 'Sunny Skies', 'Clear and Cool', 'Clear and sunny',
        'Sunny, highs to upper 80s', 'Mostly Sunny Skies', 'Cold',
        'Clear and warm', 'Sunny and warm', 'Clear and cold', 'Mostly sunny',
        'T: 51; H: 55; W: NW 10 mph', 'Clear Skies', 'Clear skies', 'Partly sunny',
        'Fair', 'Partly Sunny', 'Mostly Sunny', 'Clear', 'Sunny'
    ]
    snow  = ['Heavy lake effect snow', 'Snow']
    indoor  = ['N/A Indoor', 'Indoors', 'Indoor', 'N/A (Indoors)', 'Controlled Climate']
    
    if weather in rain:
        return 'rain'
    elif weather in overcast:
        return 'overcast'
    elif weather in clear:
        return 'clear'
    elif weather in snow:
        return 'snow'
    elif weather in indoor:
        return 'indoor'
    
    return 'unspecified'

train['GameWeather'] = train['GameWeather'].apply(group_game_weather)


# In[326]:


#Stadiums
def group_stadium_types(stadium):
    outdoor  = [
        'Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 
        'Outdor', 'Ourdoor', 'Outside', 'Outddors', 
        'Outdoor Retr Roof-Open', 'Oudoor', 'Bowl'
    ]
    indoor_closed = [
        'Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed', 
        'Retractable Roof', 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed',
    ]
    indoor_open   = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']
    dome_closed   = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']
    dome_open     = ['Domed, Open', 'Domed, open']
    
    if stadium in outdoor:
        return 'outdoor'
    elif stadium in indoor_closed:
        return 'indoor closed'
    elif stadium in indoor_open:
        return 'indoor open'
    elif stadium in dome_closed:
        return 'dome closed'
    elif stadium in dome_open:
        return 'dome open'
    else:
        return 'unknown'
    
train['StadiumType'] = train['StadiumType'].apply(group_stadium_types)


# In[327]:


#BirthDate, GameHour and Time
import datetime
train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

#train['BirthYear']=train['PlayerBirthDate'].apply(lambda x : int(x.split('/')[2]))
def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans
train['GameClock']=train['GameClock'].apply(strtoseconds)


# In[328]:


train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

seconds_in_year = 60*60*24*365.25
train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
train = train.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], axis=1)


# In[329]:


#Height
train['PlayerHeight'] = z
train['PlayerHeight'].value_counts()
train['PlayerHeight']=train['PlayerHeight'].apply(lambda x : 30*int(x.split('-')[0]) + 3*int(x.split('-')[1]))


# In[330]:


#DefensePersonnel
def process_defense(x):
    num=[]
    num=x.split(',')
    dl=int(num[0].split(' ')[0])
    lb=int(num[1].split(' ')[1])
    db=int(num[2].split(' ')[1])
    if(len(num)>3):
         ol=int(num[3].split(' ')[1])
    else:
         ol=0
    return [dl,lb,db,ol]

values=train['DefensePersonnel'].apply(process_defense)
u,v,x,y=list(map(list,zip(*values)))




# In[331]:


train['DL']=u
train['LB']=v
train['BL']=x
train['OL']=y
train.drop(['DefensePersonnel'],axis=1,inplace=True)


# In[332]:


train['PlayDirection'] = train['PlayDirection'].apply(lambda x: x.strip() == 'right')

train['Team'] = train['Team'].apply(lambda x: x.strip()=='home')


# In[333]:


train['X'] = train.apply(lambda row: row['X'] if row['PlayDirection'] else 120-row['X'], axis=1)
#from https://www.kaggle.com/scirpus/hybrid-gp-and-nn
def new_orientation(angle, play_direction):
    if play_direction == 0:
        new_angle = 360.0 - angle
        if new_angle == 360.0:
            new_angle = 0.0
        return new_angle
    else:
        return angle
    
train['Orientation'] = train.apply(lambda row: new_orientation(row['Orientation'], row['PlayDirection']), axis=1)
train['Dir'] = train.apply(lambda row: new_orientation(row['Dir'], row['PlayDirection']), axis=1)


# In[334]:


#Remove irrelevant features
train = train.drop(['GameId','PlayId','NflId','NflIdRusher'], axis=1)


# In[335]:


#Additional features
train['PlayerBMI'] = (0.5*train['PlayerWeight']/(0.01*train['PlayerHeight'])**2)


# In[373]:


def get_cats(train):
    cat_features = []
    for col in train.columns:
        if train[col].dtype =='object':
            cat_features.append(col)
    #print(cat_features)
    return cat_features


# In[368]:


train['WindDirection'].value_counts()


# In[359]:


# encoding categorical values

from sklearn.preprocessing import LabelEncoder

cats = get_cats(train)

lbdic={}
for c in cats:
    lb=LabelEncoder()
    lb=lb.fit(train[c].values)
    lbdic[c]=lb
    train[c]=lb.transform(train[c].values)


# In[362]:


train.head()


# In[371]:


#TODO: Refactor all pipeline steps as function so Test can be processed the same way.

def pipeline(df):
    #print(df.head())
    df['WindDirection'] = df['WindDirection'].apply(clean_wind_direction)
    df['WindSpeed']=df['WindSpeed'].apply(windspeed)

    df['WindSpeed'].fillna(df['WindSpeed'].median(),inplace=True)
    
    df['Humidity'].astype('float32').fillna(method='ffill', inplace=True)
    df['Temperature'].astype('float32').fillna(0, inplace=True)
    df.fillna(na_map, inplace=True)
    df['FieldPosition'] = np.where(df['YardLine'] == 50, df['PossessionTeam'], df['FieldPosition'])
    df['GameWeather'] = df['GameWeather'].apply(group_game_weather)
    df['StadiumType'] = df['StadiumType'].apply(group_stadium_types)
    
    
    
    df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    df['GameClock']=df['GameClock'].apply(strtoseconds)
    df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

    seconds_in_year = 60*60*24*365.25
    df['PlayerAge'] = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
    df = df.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], axis=1)
    df['PlayerHeight']=df['PlayerHeight'].apply(lambda x : 30*int(x.split('-')[0]) + 3*int(x.split('-')[1]))

    values=df['DefensePersonnel'].apply(process_defense)
    u,v,x,y=list(map(list,zip(*values)))
    df['DL']=u
    df['LB']=v
    df['BL']=x
    df['OL']=y
    df.drop(['DefensePersonnel'],axis=1,inplace=True)

    df['PlayDirection'] = df['PlayDirection'].apply(lambda x: x.strip() == 'right')

    df['Team'] = df['Team'].apply(lambda x: x.strip()=='home')

    df['X'] = df.apply(lambda row: row['X'] if row['PlayDirection'] else 120-row['X'], axis=1)

    df['Orientation'] = df.apply(lambda row: new_orientation(row['Orientation'], row['PlayDirection']), axis=1)
    df['Dir'] = df.apply(lambda row: new_orientation(row['Dir'], row['PlayDirection']), axis=1)

    #Remove irrelevant features
    df = df.drop(['GameId','PlayId','NflId','NflIdRusher'], axis=1)
    #Additional features
    df['PlayerBMI'] = (0.5*df['PlayerWeight']/(0.01*df['PlayerHeight'])**2)

    #Encode categorical features
    cats = get_cats(df)

    lbdic={}
    for c in cats:
        lb=LabelEncoder()
        lb=lb.fit(df[c].values)
        lbdic[c]=lb
        df[c]=lb.transform(df[c].values)
        
    #print(df.head)
    features = df.drop(['Yards'], axis=1).copy()
    label = df['Yards'].copy()
    return features, label


# In[372]:


data = pd.read_csv('train.csv', low_memory = False)
features, label = pipeline(data)


# In[ ]:


#split into custom validation/test set - or can we use their test set as black box? 


# In[388]:


#turns yardage into categorical format to match softmax output. We will aggregate the NN PDF into a CDF afterwards.
y_train = tf.keras.utils.to_categorical(y_train, 199)
y_test = tf.keras.utils.to_categorical(y_test, 199)

#Turns yardage gained into cumulative function: 000..111111. Flips to 1 at the index of the "yards gained"
def transform_label(y):
    Y_train=np.zeros((y.shape[0],199))
    for i,yard in enumerate(y):
        Y_train[i, yard+99:] = np.ones(shape=(1, 100-yard))
    
    return Y_train

Y = transform_label(label)

from sklearn.preprocessing import StandardScaler

#optional scaling of X
#scaler = StandardScaler()
#X = scaler.fit_transform(features)


# In[ ]:


# baseline model Copy from Hw2 - tune accordingly

#redefine an optimizer that inherits from SGD or any optimizer but introduces synchronization
def optim_SGD(lr = 0.001, mom=0.99):
    #Declare Optimizer
    optim = tf.keras.optimizers.SGD(learning_rate = lr, momentum = mom)
    return optim

#set up layers for baseline model: remove convolutions and set dense layers... Softmax output with 199 dim
def new_model(optim, c1 =32, c2=32, d=128):

    models = tf.keras.models
    layers = tf.keras.layers
    #Instantiate an empty model
    model = models.Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv2D(c1, kernel_size =(3, 3), activation="relu", input_shape=(28,28,1)))

    # S2 Pooling Layer
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # C3 Convolutional Layer
    model.add(layers.Conv2D(c2, kernel_size =(3, 3), activation="relu"))

    # S4 Pooling Layer
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())

    # C5 Dense Layer
    model.add(layers.Dense(d, activation="relu"))

    #Output Layer with softmax activation
    model.add(layers.Dense(10, activation="softmax"))

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optim, metrics=["accuracy"])

    print(model.summary())

    return model

