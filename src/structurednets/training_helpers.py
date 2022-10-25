import torch
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from structurednets.asset_helpers import load_features

def get_accuracy(y_true: torch.tensor, y_pred: torch.tensor):
    return (torch.sum(y_pred.argmax(axis=1) == y_true) / len(y_true)).cpu().detach().numpy()

def get_loss(y_true: torch.tensor, y_pred: torch.tensor):
    cross_entropy = torch.nn.CrossEntropyLoss()
    return cross_entropy(y_pred, target=y_true).cpu().detach().numpy()

def get_train_data(features_path: str):
    X, y, _ = load_features(features_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    return X_train, X_val, y_train, y_val

def get_batch(X: np.ndarray, y: np.ndarray, batch_size: int, batch_i: int):
    X_batch = X[batch_i*batch_size:min((batch_i+1)*batch_size, X.shape[0])]
    y_batch = y[batch_i*batch_size:min((batch_i+1)*batch_size, y.shape[0])]

    X_batch_t, y_batch_t = map(torch.tensor, (X_batch, y_batch))
    X_batch_t = X_batch_t.float()
    y_batch_t = y_batch_t.long()

    return X_batch_t, y_batch_t

def get_full_batch(X: np.ndarray, y: np.ndarray):
    return get_batch(X=X, y=y, batch_size=X.shape[0], batch_i=0)

def train_with_features(model: torch.nn.Module, features_path: str, patience=10, batch_size=1000, verbose=False, lr=1e-6, restore_best_model=True):
    X_train, X_val, y_train, y_val = get_train_data(features_path)
    return train(
        model=model, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
        patience=patience, batch_size=batch_size, verbose=verbose, lr=lr, restore_best_model=restore_best_model
    )

def train(model: torch.nn.Module, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, patience=10, batch_size=1000, verbose=False, lr=1e-6, restore_best_model=True):
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()
    nb_batches_per_epoch = np.ceil(X_train.shape[0] / batch_size).astype("int")

    X_train_t, y_train_t = get_full_batch(X_train, y_train)
    y_train_pred = model(X_train_t)
    start_train_loss = get_loss(y_train_t, y_train_pred)
    start_train_accuracy = get_accuracy(y_train_t, y_train_pred)
    X_val_t, y_val_t = get_full_batch(X_val, y_val)
    y_val_pred = model(X_val_t)
    start_val_loss = get_loss(y_val_t, y_val_pred)
    start_val_accuracy = get_accuracy(y_val_t, y_val_pred)
    if verbose:
        print("*"*20)
        print("Start train Loss: " + str(start_train_loss))
        print("Start train accuracy: " + str(start_train_accuracy))
        print("Start val loss: " + str(start_val_loss))
        print("Start val accuracy: " + str(start_val_accuracy))
        print("*"*20)

    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    best_val_accuracy = -1
    best_model = None

    continue_training = True
    epoch = 1
    while continue_training:
        X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train)

        for batch_i in range(nb_batches_per_epoch):
            X_batch_t, y_batch_t = get_batch(X_train_shuffled, y_train_shuffled, batch_size=batch_size, batch_i=batch_i)
            outputs_train = model(X_batch_t)
            loss_train = loss_function(outputs_train, target=y_batch_t)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        
        X_train_t, y_train_t = get_full_batch(X_train_shuffled, y_train_shuffled)
        y_train_pred = model(X_train_t)
        train_loss_history.append(get_loss(y_train_t, y_train_pred))
        train_accuracy_history.append(get_accuracy(y_train_t, y_train_pred))

        X_val_t, y_val_t = get_full_batch(X_val, y_val)
        y_val_pred = model(X_val_t)
        val_loss_history.append(get_loss(y_val_t, y_val_pred))
        val_accuracy_history.append(get_accuracy(y_val_t, y_val_pred))

        if verbose:
            print("--- Epoch " + str(epoch) + " ---")
            print("Train Acc: " + str(train_accuracy_history[-1]))
            print("Train Loss: " + str(train_loss_history[-1]))
            print("Val Acc: " + str(val_accuracy_history[-1]))
            print("Val Loss: " + str(val_loss_history[-1]))

        if len(val_loss_history) > 2*patience \
            and np.max(val_accuracy_history[-patience:]) < np.max(val_accuracy_history[:-patience]):
            continue_training = False

        if val_accuracy_history[-1] > best_val_accuracy:
            best_val_accuracy = val_accuracy_history[-1]
            best_model = pickle.loads(pickle.dumps(model))

        epoch += 1
    
    if restore_best_model:
        return best_model, start_train_loss, start_train_accuracy, start_val_loss, start_val_accuracy, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history
    else:
        return model, start_train_loss, start_train_accuracy, start_val_loss, start_val_accuracy, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history