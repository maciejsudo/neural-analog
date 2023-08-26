import torch.nn as nn
from model_architestures import GuitarLSTM, ESRLoss
from scipy.io import wavfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader,  TensorDataset
import soundfile as sf
import os

'''
class WindowArrayDataset(Dataset):
    def __init__(self, x, y, window_len):
        self.x = x
        self.y = y[window_len - 1 :]
        self.window_len = window_len

    def __len__(self):
        return len(self.x) - self.window_len + 1

    def __getitem__(self, index):
        x_out = torch.stack([torch.tensor(self.x[idx: idx + self.window_len]) for idx in range(index, index + 1)]).view(-1, self.window_len)
        y_out = torch.tensor(self.y[index : index + self.window_len]).view(-1, self.window_len)
        return x_out, y_out
'''



class WindowArrayDataset(Dataset):
    def __init__(self, x, y, window_len, batch_size):
        self.x = x
        self.y = y[window_len - 1 :]
        self.window_len = window_len
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.x) - self.window_len + 1) // batch_size

    def __getitem__(self, index):
        temp = [torch.tensor(self.x[idx: idx+self.window_len]) for idx in range(index*self.batch_size, (index+1)*self.batch_size)]
        x_out = torch.stack(temp)
        y_out = torch.tensor(self.y[index*self.batch_size:(index+1)*self.batch_size])
        return x_out, y_out

def my_collate(data_in):
    X= []
    Y =[]
    Z =[]

    for seq in data_in:
        X.append(seq[0])
        Y.append(seq[1])

        X = torch.stack(X,dim=1)
        Y = torch.stack(Y,dim=1)
        return X,Y
#class WindowArrayLoader(DataLoader):
#    def __init__(self, x, y, window_len, batch_size=32, shuffle=False):
#        dataset = WindowArrayDataset(x, y, window_len)
#        super(WindowArrayLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)

def normalize(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max,abs(data_min))
    return data / data_norm

def audio_converter(audio):

    if audio.dtype == 'float32':
        return audio
    elif audio.dtype == 'int16':
        return audio.astype(np.float32, order='C') / 32768.0
    elif audio.dtype == 'int32':
        return audio.astype(np.float32, order='C') / 2147483648.0
    else:
        raise RuntimeError('unimplemented audio data type conversion...')


def prepare_WAVs(path):
    in_file = path + '-input.wav'
    out_file = path + '-target.wav'

    in_rate, in_data = wavfile.read(in_file)
    out_rate, out_data = wavfile.read(out_file)

    #print("IN type =",in_data.dtype, "\nsamplerate =",in_rate,"\ndatalength =",in_data.shape)
    #print("\nOUT type =", out_data.dtype, "\nsamplerate =", out_rate, "\ndatalength =", out_data.shape)

    in_data = audio_converter(in_data)
    out_data = audio_converter(out_data)

    X_all = in_data.astype(np.float32).flatten()
    X_all = normalize(X_all).reshape(len(X_all), 1)

    y_all = out_data.astype(np.float32).flatten()
    y_all = normalize(y_all).reshape(len(y_all), 1)

    return [X_all, y_all]

def save_wav(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))

if __name__ == "__main__":

    name = 'test1'

    path = "D:\\repos\MGR\\neural-analog\data\\test\\combined"
    data_test = prepare_WAVs(path)

    path = "D:\\repos\MGR\\neural-analog\data\\train\\combined"
    data_train = prepare_WAVs(path)

    path = "D:\\repos\MGR\\neural-analog\data\\validation\\combined"
    data_val = prepare_WAVs(path)


    #LSTM#


    epochs = 1
    input_size = 150
    batch_size = 4096
    test_size = 0.2
    train_mode = 0

    if train_mode == 0:  # Speed Training
        learning_rate = 0.01
        conv1d_strides = 12
        conv1d_filters = 16
        hidden_units = 36
    elif train_mode == 1:  # Accuracy Training (~10x longer than Speed Training)
        learning_rate = 0.01
        conv1d_strides = 4
        conv1d_filters = 36
        hidden_units = 64
    else:  # Extended Training (~60x longer than Accuracy Training)
        learning_rate = 0.0005
        conv1d_strides = 3
        conv1d_filters = 36
        hidden_units = 96

    # Define model:
    model = GuitarLSTM(conv1d_filters, conv1d_strides, input_size, hidden_units)
    print('Model architcture: \n',model)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loader:
    train_arr = WindowArrayDataset(data_train[0], data_train[1], input_size, batch_size= batch_size)
    train_loader = DataLoader(train_arr, batch_size=1, shuffle=False)

    #validation loader:
    val_arr = WindowArrayDataset(data_val[0], data_val[1], input_size, batch_size= batch_size)
    val_loader = DataLoader(val_arr, batch_size=1, shuffle=False)




    print_interval = 100
    # Train Model ###################################################
    for epoch in range(epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):

            inputs = inputs.view(1 *4096, 150, 1)
            targets = targets.view(1 *4096, 1)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (i + 1) % print_interval == 0: # every each 100epochs print and run validation!:

                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_arr)}], Loss: {loss.item():.4f}")

        # for each epoch calculate validation loss:
        mse_sum = 0.0
        num_batches = 0
        model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.view(1 * 4096, 150, 1)
                targets = targets.view(1 * 4096, 1)

                outputs = model(inputs)
                val_loss = loss_fn(outputs,targets)

                mse_sum += val_loss
                num_batches +=1
            average_mse = mse_sum / num_batches
            print(f"Average validation MSE: {average_mse:.4f}")



    # Save trained model
    new_dir = os.path.join('models',name)
    os.makedirs(new_dir, exist_ok=True)

    torch.save(model.state_dict(), f'models/{name}/{name}.pth')

    #running prediction on test dataset:

    test_arr = WindowArrayDataset(data_test[0], data_test[1], input_size, batch_size=204800)

    model.load_state_dict(torch.load(f'models/{name}/{name}.pth'))
    model.eval()

    # Convert test_arr to a PyTorch tensor
    #test_loader = DataLoader(test_arr, batch_size=batch_size, shuffle=False)

    index = 0
    sample_item = test_arr[index]

    # Make predictions
    with torch.no_grad():
        prediction = model(sample_item[0])

    # Convert PyTorch tensor to numpy array
    prediction = prediction.numpy()


    # Save predictions as WAV
    sf.write(f'models/{name}/y_pred.wav', prediction, samplerate=44100)




