# %%



# Import statement

import json

from typing import List, Dict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

## Keras
import keras
from keras.models import Model, Input
from keras.layers import LSTM, Dense, Concatenate
from keras.losses import categorical_crossentropy
from keras.activations import sigmoid, softmax, relu, tanh
from keras.metrics import Accuracy
from keras.optimizers import Adam

import pickle



def import_csv(filename: str, separate_rate: float=0.75):
    data = pd.read_csv(filename)

    x_df = data.iloc[:, :-1]
    y_df = data.iloc[:, -1:]

    separate_rate = separate_rate
    n_training = int(separate_rate * x_df.shape[0])

    x_train = x_df.iloc[:n_training,:].values
    y_train = y_df.iloc[:n_training,:].values
    x_val = x_df.iloc[n_training:,:].values
    y_val = y_df.iloc[n_training:,:].values
    n_match = data.shape[0]
    n_match_train = y_train.shape[0]
    n_match_val = y_val.shape[0]

    return (x_train, y_train, n_match_train), (x_val, y_val, n_match_val), n_match


def build_model():
    '''
    input: in
    output: out
    '''
    inp = Input(shape=(5,))

    tensor = Dense(64, activation='relu', kernel_initializer='random_normal') (inp)

    tensor = Dense(32, activation='relu', kernel_initializer='random_normal') (tensor)

    output = Dense(1, activation='sigmoid', kernel_initializer='random_normal')(tensor)
    
    model = Model(inp, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    model.summary()
    return model

# %%

# (x_train_1, y_train_1, n_train_1), (x_val_1, y_val_1, n_val_1), n_match_1 = \
#     import_csv("./GinRummyMavenProject/data_knock_2_simple.csv", 0.95)

# (x_train_2, y_train_2, n_train_2), (x_val_2, y_val_2, n_val_2), n_match_2 = \
#     import_csv("./GinRummyMavenProject/data_knock_undercut.csv", 0.95)


# x_train = np.append(x_train_1, x_train_2, axis=0)
# y_train = np.append(y_train_1, y_train_2, axis=0)
# x_val = np.append(x_val_1, x_val_2, axis=0)
# y_val = np.append(y_val_1, y_val_2, axis=0)
# n_train = n_train_1 + n_train_2
# n_val = n_val_1 + n_val_2
# n_match = n_match_1 + n_match_2


# %%

(x_train, y_train, n_train), (x_val, y_val, n_val), n_match = \
    import_csv("../java/MavenProject/dataset/data_knock_v2.csv", 0.9)


# %%

model = build_model()


# %%
size = y_train.shape[0]
batch_size = 100
history = model.fit(x = x_train[:size], \
    y = y_train[:size],\
    batch_size = batch_size,\
    epochs=35,\
    verbose=1,\
    validation_split=0.75)


# %%

MODEL_PATH = '../java/MavenProject/src/main/java/model/'

model.save(MODEL_PATH + 'knocking_100_v2.h5')  # save everything in HDF5 format
model_json = model.to_json()  # save just the config. replace with "to_yaml" for YAML serialization
with open(MODEL_PATH + "knocking_100_v2_config.json", "w") as f:
    f.write(model_json)
model.save_weights(MODEL_PATH + 'knocking_100_v2_weights.h5') # save just the weights.

# %%
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# or save to csv: 
hist_csv_file = 'knock_100_v2_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# %%

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# %%

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# %%
batch = 100
r = np.random.randint(0,y_val.shape[0] - batch)
y_pred = model.predict(x_val[r:r+batch])
print(y_pred.reshape((-1,)))
print(y_val[r:r+batch].reshape((-1,)))

def find_melds(hand):
    if not hand:
        return []
        
    melds = []
    ranks = {}
    suits = {}
    
    for card in hand:
        rank = card % 13
        suit = card // 13
        ranks.setdefault(rank, []).append(card)
        suits.setdefault(suit, []).append(card)
        
    for cards in ranks.values():
        if len(cards) >= 3:
            melds.append(cards[:])
            
    for cards in suits.values():
        cards.sort(key=lambda x: x % 13)
        i = 0
        while i < len(cards) - 2:
            run = [cards[i]]
            j = i + 1
            while j < len(cards) and cards[j] % 13 == (run[-1] % 13) + 1:
                run.append(cards[j])
                j += 1
            if len(run) >= 3:
                melds.append(run)
            i = j if j > i + 1 else i + 1
            
    return melds

def get_deadwood(hand, melds=None):
    if not hand:
        return float('inf')
        
    if melds is None:
        melds = find_melds(hand)
        
    if not melds:
        return sum(min(10, x % 13 + 1) for x in hand)
        
    min_deadwood = float('inf')
    
    for i in range(1 << len(melds)):
        used = set()
        valid = True
        
        for j in range(len(melds)):
            if i & (1 << j):
                meld = melds[j]
                if any(card in used for card in meld):
                    valid = False
                    break
                used.update(meld)
                
        if valid:
            unmatched = [card for card in hand if card not in used]
            deadwood = sum(min(10, x % 13 + 1) for x in unmatched)
            min_deadwood = min(min_deadwood, deadwood)
            
    return min_deadwood

def can_knock(hand):
    return get_deadwood(hand) <= 10

def can_gin(hand):
    return get_deadwood(hand) == 0

def get_best_discard(hand):
    if not hand:
        return None
        
    best_card = None
    min_deadwood = float('inf')
    
    for card in hand:
        test_hand = hand.copy()
        test_hand.remove(card)
        deadwood = get_deadwood(test_hand)
        if deadwood < min_deadwood:
            min_deadwood = deadwood
            best_card = card
            
    return best_card

def get_best_draw(hand, discard_top):
    if not hand or discard_top is None:
        return None
        
    old_deadwood = get_deadwood(hand)
    test_hand = hand + [discard_top]
    new_deadwood = get_deadwood(test_hand)
    
    return new_deadwood < old_deadwood

def get_best_action(hand, discard_top, phase):
    if phase == 'draw':
        if discard_top is not None and get_best_draw(hand, discard_top):
            return 1  # Draw from discard
        return 0  # Draw from stock
        
    deadwood = get_deadwood(hand)
    if deadwood == 0:
        return 109  # Gin
    if deadwood <= 10:
        return 108  # Knock
        
    card = get_best_discard(hand)
    if card is not None:
        return card + 2  # Discard
        
    return None
