 #!/usr/bin/env python3 -W ignore::DeprecationWarning

import numpy as np
import pandas as pd
import tensorflow as tf
#from kaggle.competitions import nflrush
import time
import datetime
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import tensorflow.keras.backend as K
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

models = tf.keras.models
layers = tf.keras.layers
#Instantiate an empty model
model = models.Sequential()
features = pd.DataFrame()
label = pd.DataFrame()
X = 0
Y = 0
# Cleaning functions

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

def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans

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

def new_orientation(angle, play_direction):
    if play_direction == 0:
        new_angle = 360.0 - angle
        if new_angle == 360.0:
            new_angle = 0.0
        return new_angle
    else:
        return angle

def get_cats(train):
    cat_features = []
    for col in train.columns:
        if train[col].dtype =='object':
            cat_features.append(col)
    #print(cat_features)
    return cat_features

def pipeline(df, train=False):
    #print(df.head())
    df['WindDirection'] = df['WindDirection'].apply(clean_wind_direction)
    df['WindSpeed']=df['WindSpeed'].apply(windspeed)

    df['WindSpeed'].fillna(df['WindSpeed'].median(),inplace=True)

    df['Humidity'].fillna(method='ffill', inplace=True)
    df['Temperature'].fillna(method='ffill', inplace=True)


    na_map = {
        'Orientation': df['Orientation'].mean(),
        'Dir': df['Dir'].mean(),
        'DefendersInTheBox': (df['DefendersInTheBox'].median()),
        'OffenseFormation': 'blank'
    }

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

    df = df.sample(frac=1)  #could this lead to overfitting of the validation set?
    #print(df.head)
    if(train):
        features = df.drop(['Yards'], axis=1).copy()
        label = df['Yards'].copy()
        return features, label
    else:
        return features

def transform_label(label):
    inter = np.apply_along_axis(lambda x: x+99, 0, label)
    return tf.keras.utils.to_categorical(inter, 199)

def Heavi_transform_label(y):
    Y_train=np.zeros((y.shape[0],199))
    for i,yard in enumerate(y):
        Y_train[i, yard+99:] = np.ones(shape=(1, 100-yard))

    return Y_train

def CDF_transform_label(y):
    return transform_label(y).cumsum(axis =1)

def normalizeData(features):
    x = features.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return x_scaled


def process_data(test_size = 1000):
    data = pd.read_csv('train.csv', low_memory = False)
    print(data.head())
    features, labels = pipeline(data, train=True)

    print(features.head())
    X = normalizeData(features)
    Y_heavi = Heavi_transform_label(labels)
    Y_cat = transform_label(labels)

    pd.DataFrame(data=X).to_csv(path_or_buf='features.csv', index=False)
    pd.DataFrame(data=Y_heavi).to_csv(path_or_buf='labelsHeavi.csv', index=False)
    pd.DataFrame(data=Y_cat).to_csv(path_or_buf='labelsCat.csv', index=False)
    return X, Y


def import_data(config):
    features = pd.read_csv(config['features_path'], low_memory = False)
    #labels = pd.read_csv('labelsCat.csv', low_memory = False)
    labels = pd.read_csv(config['labels_path'], low_memory = False)
    #print(features.head())
    #X = normalizeData(features)
    #Y = Heavi_transform_label(labels)
    X = features.as_matrix()
    Y = labels.as_matrix()
    return X, Y



#from https://www.kaggle.com/davidcairuz/nfl-neural-network-w-softmax
def crps(y_true, y_pred):
    return K.mean(100*K.square(y_true - K.cumsum(y_pred, axis=1)), axis=1)

def crpsB(y_true, y_pred):
    return K.mean(100*K.square(K.cumsum(y_true, axis=1) - K.cumsum(y_pred, axis=1)), axis=1)

def optim_SGD(lr = 0.001, mom=0.99):
    #Declare Optimizer
    optim = tf.keras.optimizers.SGD(learning_rate = lr, momentum = mom) #, clipnorm=1.)
    return optim

def optim_Adam(lr = 0.001, mom=0.99):
    #Declare Optimizer
    optim = tf.keras.optimizers.Adam(learning_rate = lr, momentum = mom) #, clipnorm=1.)
    return optim

#set up layers for baseline model: remove convolutions and set dense layers... Softmax output with 199 dim
def new_model(optim, d1= 256, d2 =256, d3 =256, d4=256, d5 =256, lo=keras.losses.categorical_crossentropy):

    #models = tf.keras.models
    #layers = tf.keras.layers
    #Instantiate an empty model
    #model = models.Sequential()


    model.add(layers.Dense(d1, activation="relu", input_shape=(47,))) #[features.shape[1]]))
    #model.add(layers.BatchNormalization())

    model.add(layers.Dense(d2, activation="relu"))
    #model.add(layers.BatchNormalization())

    model.add(layers.Dense(d3, activation="relu"))
    #model.add(layers.BatchNormalization())

    model.add(layers.Dense(d4, activation="relu"))
    #model.add(layers.BatchNormalization())

    model.add(layers.Dense(d5, activation="relu"))
    #model.add(layers.BatchNormalization())

    #Output Layer with softmax activation
    model.add(layers.Dense(199, activation="softmax"))
    #model.add(layers.Flatten())
    # Compile the model
    model.compile(loss=lo, optimizer=optim, metrics=["accuracy"])
    #keras.losses.categorical_crossentropy
    print(model.summary())

    return model

def new_modelB(optim, lo=keras.losses.categorical_crossentropy):
    models = tf.keras.models
    layers = tf.keras.layers
    #Instantiate an empty model
    model = models.Sequential()


    model.add(layers.Dense(450, activation="relu", input_shape=(47,))) #[features.shape[1]]))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.05))
    # model.add(layers.Flatten())
    model.add(layers.Dense(600, activation="relu"))
    model.add(layers.Dropout(0.05))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Flatten())
    model.add(layers.Dense(450, activation="relu"))
    model.add(layers.Dropout(0.05))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dense(199, activation="softmax"))
    #model.add(layers.Flatten())
    # Compile the model
    #model.compile(loss=lo, optimizer=optim, metrics=["accuracy"])
    model.compile(loss=lo, optimizer=optim, metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model

def eval(X, Y):
    optim = optim_SGD(lr = 0.05, mom=0)

    #Heaviside

    #model = new_model(optim, lo=crps)
    #model.fit(x=X,y=Y, epochs=1, validation_split = 0.1, verbose=1) # batch_size=32

    #Catecorical
    model = new_model(optim)
    model.fit(x=X,y=Y, epochs=1, validation_split = 0.1, verbose=1)

#dont use anymore.
def sample_test(test_size = 50, alt=False):
    test_indices = np.random.choice(X.shape[0], test_size)
    if alt:
        return alt_C(test_indices)
    return C(test_indices)

def sq_heaviside(test_pred, test_label):
    #print(test_pred)
    test_cdf = test_pred.cumsum()
    #print(test_cdf)
    sum = 0
    for i in range(test_cdf.size):
        sum += np.square(test_cdf[i] - np.heaviside((i-99) - test_label, 1))
    return sum

def alt_sq_heaviside(test_pred, test_label):
    #print(test_pred)
    #print(test_label)
    test_cdf = test_pred.cumsum()
    test_cdf = np.clip(test_cdf, 0, 1)
    #print(test_cdf)
    sum = 0
    for i in range(test_cdf.size):
        sum += np.square(test_cdf[i] - np.heaviside((i-99) - test_label, 1))
    return sum
    #test_cdf.apply(lambda x: np.sq(x - np.heaviside(x, 0)) axis=1)

def C(test_indices):
    C = 0
    for index in test_indices:
        test_pred = model.predict(np.array([X[index]]))
        test_label = label[index]
        #print("i:", index, "label:", test_label)
        C += sq_heaviside(test_pred, test_label)

    C *= 1/(199*test_indices.size)
    return C

def alt_C(test_indices):
    C = 0
    for index in test_indices:
        test_pred = model.predict(np.array([X[index]])).flatten()  #orignially alt model
        test_label = label[index]
        #print("i:", index)
        C += alt_sq_heaviside(test_pred, test_label)

    C *= 1/(199*test_indices.size)
    return C

def model_prediction(model, play_idx, X, Y):
    inter = np.array([X[play_idx]])
    print("True yardage:", np.argmax(Y[play_idx])-99)
    print("Predicted yardage:", np.argmax(model.predict(inter))-99)


if __name__== "__main__":
    X, Y = process_data()
#    eval(X,Y)
    #loss = sample_test(test_size=1000, alt=True)
    #print(loss)
    #Just use evaluate from model for test set.
#    test_indices = np.random.choice(X.shape[0], 100)
#    test_score = model.evaluate(X[test_indices], Y[test_indices])
#    print(test_score)
