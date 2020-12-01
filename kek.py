# %%

import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")
df_orig = df

df.info()
# "Item_Weight" and "Outlet_Size" has some missing values,
# but I still need to look into the other columns for consistent data

# %%

# Item Weight
from tqdm import tqdm

# Some items does not have weight, but they have the same ID as other items that have weight.
# I expect an item with the same ID to also have the weight.


unique_item_ids = pd.unique(df['Item_Identifier'])
# Some weights are not present in the train dataset, so I will look for values in the test set.
test = pd.read_csv('test.csv')
# since this takes a little while, I use the tqdm library to add a progress bar

for id in tqdm(unique_item_ids):
    group = df[df['Item_Identifier'] == id]
    test_group = test[test['Item_Identifier'] == id]

    missing = group[pd.isna(group['Item_Weight'])]
    if len(missing) == 0:
        continue

    not_missing = group[pd.notna(group['Item_Weight'])]

    fill_val = 0
    if len(not_missing) > 0:
        fill_val = not_missing['Item_Weight'].iloc[0]
    elif len(test_group) > 0:
        fill_val = test_group['Item_Weight'].iloc[0]
    else:
        continue

    df.loc[df['Item_Identifier'] == id, ['Item_Weight']] = df.loc[df['Item_Identifier'] == id, ['Item_Weight']].fillna(
        fill_val)

# %%

len(df[df['Item_Weight'].isna()])

# %%

df[df['Outlet_Size'].isna()].head(50)

# %%

# After filling "Item_Weight" with help from the test set,
# there is only 8 unique items without weight
# FDS09, FDN52, FDZ50, FDZ50, FDH52, FDH52, FDE52, NCT53
# for now I will drop them

# %%

# Item Fat Content
df['Item_Fat_Content'] = df['Item_Fat_Content'].map({
    'Low Fat': 'Low Fat',
    'low fat': 'Low Fat',
    'LF': 'Low Fat',
    'Regular': 'Regular',
    'reg': 'Regular'
})
unique_item_fat_content = pd.unique(df['Item_Fat_Content'])
unique_item_fat_content

# %%

# Item Visibility
visibility = df[['Item_Visibility', 'Outlet_Identifier']].groupby('Outlet_Identifier').sum()

# I see that some items have a visibility of 0.0
# I must believe that these items are indeed not visible.

# %%

# Item Type
# Seems to be good quality data and is consistent, does not need preprocessing

# %%

# Item MRP
# Since we dont usually deal with more than 2 decimal places in prices, I remove excess decimals
df['Item_MRP'] = df['Item_MRP'].round(2)

# %%

# Outlet Establishment Year
# Seems to be good quality data and is consistent. Does not need preprocessing

# Outlet Location Type
# Seems to be good quality data and is consistent. Does not need preprocessing

# Outlet Type
# Seems to be good quality data and is consistent. Does not need preprocessing

# %%

# Outlet Size
# 'OUT010', 'OUT045' and 'OUT017' is missing size
# These are not presint in the test set
# I want to predict them using a simple KNN
# Even though the dataset is tiny,
# I think this is better than just putting in average or median values,
# or even dropping 3 stores (even though they are not present in the test set)
# since they account for more that 1/4 of the training data

# Its impossible to perform classification on nominal data, so will expect
# Location type to be ordnial, with Type 1 < Type 2 < Type 3
# and type with "Groceries Store" < "Super Market 1" < 2 < 3
# This will not completely represent the data, but its still better than dropping 1/4 of the data.

outlets = df[['Outlet_Identifier',
              'Outlet_Establishment_Year',
              'Outlet_Location_Type',
              'Outlet_Type',
              'Outlet_Size']]

outlets = outlets.drop_duplicates()

outlets['Outlet_Location_Type'] = outlets['Outlet_Location_Type'].replace({'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2})
outlets['Outlet_Type'] = outlets['Outlet_Type'].replace(
    {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3})
outlets['Outlet_Size'] = outlets['Outlet_Size'].replace({'Small': 0, 'Medium': 1, 'High': 2})

missing = outlets[pd.isna(outlets['Outlet_Size'])]

missing = missing[['Outlet_Identifier',
                   'Outlet_Establishment_Year',
                   'Outlet_Location_Type',
                   'Outlet_Type']]

missing_X = missing[['Outlet_Establishment_Year',
                     'Outlet_Location_Type',
                     'Outlet_Type']]

outlets = outlets[pd.notna(outlets['Outlet_Size'])]

X = outlets[['Outlet_Establishment_Year',
             'Outlet_Location_Type',
             'Outlet_Type']]

y = outlets['Outlet_Size']
from sklearn.model_selection import train_test_split

# I have tested with a test train split, and tried different combinations.
# Its hard to learn anything from such a small dataset, but since KNN "learns"
# the data, I think its ok to drop the test set in this case, and evaluate myself.
# the result I ended up with is close to what I would predict myself.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/7, random_state=3)

# %%

from sklearn.neighbors import KNeighborsClassifier

# uses brute force because of the tiny dataset
neighs = [KNeighborsClassifier(n_neighbors=i, algorithm='brute') for i in range(1, 6)]

for neigh in neighs:
    neigh = neigh.fit(X, y)

# Im happiest with prediction number 3, with 3 neighbours. I will use its output to fill missing store sizes
pred = neighs[2].predict(missing_X)

# %%

missing['Outlet_Size'] = pred
missing['Outlet_Size'] = missing['Outlet_Size'].replace({0: 'Small', 1: 'Medium', 2: 'High'})

for i in range(3):
    Oid = missing.iloc[i]['Outlet_Identifier']
    size = missing.iloc[i]['Outlet_Size']
    df.loc[df['Outlet_Identifier'] == Oid, ['Outlet_Size']] = size
df[df['Outlet_Size'].isna()]
df = df.dropna()


# %%

def one_hot_column(col):
    global df
    one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = df.drop(columns=col)
    df = df.join(one_hot)


one_hot_column('Item_Fat_Content')
one_hot_column('Item_Type')
one_hot_column('Outlet_Location_Type')
one_hot_column('Outlet_Type')
one_hot_column('Outlet_Size')

# %%

df = df.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

# %%

X = df.loc[:, df.columns != 'Item_Outlet_Sales']

y = df['Item_Outlet_Sales']

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(x_scaled)

X = X.to_numpy()
y = y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)
# Using a dev set to better evaluate the models
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, random_state=1, test_size=0.1)

# %%

# Since making model architectures are hard, I dont want to do it myself.
# I will set the features of the model as hyper parameters.

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
import datetime
from tensorboard.plugins.hparams import api as hp
from sklearn.metrics import r2_score

logdir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

HP_LAYERS = hp.HParam('layers', hp.Discrete([1, 2, 3, 4]))
HP_LAYER_WIDTH = hp.HParam('layers_width', hp.Discrete([4, 8, 16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['relu']))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([8, 16, 32, 64, 128, 256]))


METRIC = 'mae'

with tf.summary.create_file_writer('logs\\hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_LAYERS, HP_LAYER_WIDTH, HP_DROPOUT, HP_OPTIMIZER, HP_ACTIVATION],
        metrics=[hp.Metric(METRIC, display_name='mae')],
    )


def train_test_model(hparams):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    for layer in range(hparams[HP_LAYERS]):
        model.add(tf.keras.layers.Dense(hparams[HP_LAYER_WIDTH], activation=hparams[HP_ACTIVATION]))
        model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT]))
    model.add(tf.keras.layers.Dense(1, activation=hparams[HP_ACTIVATION]))

    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss=METRIC
    )

    model.fit(X_train, y_train, hparams[HP_BATCH_SIZE], epochs=1000, callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10),
        tf.keras.callbacks.TensorBoard(logdir),  # log metrics
        hp.KerasCallback(logdir, hparams),  # log hparams
        # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    ])  # Run with 1 epoch to speed things up for demo purposes
    _, mae = model.evaluate(X_test, y_test)
    return mae


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        mae = train_test_model(hparams)
        tf.summary.scalar(METRIC, mae, step=1)


session_num = 1 # has to be unique for each run

for layers in HP_LAYERS.domain.values:
    for layer_width in HP_LAYER_WIDTH.domain.values:
        for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
            for optimizer in HP_OPTIMIZER.domain.values:
                for activation in HP_ACTIVATION.domain.values:
                    for batch_size in HP_BATCH_SIZE.domain.values:
                        hparams = {
                            HP_LAYERS: layers,
                            HP_LAYER_WIDTH: layer_width,
                            HP_DROPOUT: dropout_rate,
                            HP_OPTIMIZER: optimizer,
                            HP_ACTIVATION: activation,
                            HP_BATCH_SIZE: batch_size,
                        }
                        run_name = "run-%d" % session_num
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        run('logs/hparam_tuning/' + run_name, hparams)
                        session_num += 1
