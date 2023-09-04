import torch.nn as nn
from model_architestures import GuitarLSTM, ESRLoss
from scipy.io import wavfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader,  TensorDataset
import soundfile as sf
import os
from matplotlib import pyplot as plt



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

def concat_WAV(data, value):
    '''
    # data = audio data tensor (23456,1)
    # value = tine. drive or level param
    '''
    audio_len = list(data.size())[0]
    param = torch.full((audio_len,1), value)
    return torch.cat((data,param), dim =1)


def return_n_data(full_data, div_factor,exact_number):
    '''
    # full data = the tensor (audio_values, params)
    # div_factor = the value dividing the full data size
    # exact number = [0...div_factor-1]
    '''
    full_size = list(full_data.size())[0]
    a = int(full_size/div_factor)
    return full_data[a*exact_number : a*(exact_number+1)]


def save_wav(name, data):
      wavfile.write(name, 44100, data.flatten().astype(np.float32))



if __name__ == "__main__":

  name = 'test1'

  path = "D:\\repos\MGR\\neural-analog\data\\test\\t20k_l100k_d100k_fs44100_bd16_time180s"
  data_test = prepare_WAVs(path)

  path = "D:\\repos\MGR\\neural-analog\data\\train\\t20k_l100k_d100k_fs44100_bd16_time180s"
  data_train = prepare_WAVs(path)

  path = "D:\\repos\MGR\\neural-analog\data\\validation\\t20k_l100k_d100k_fs44100_bd16_time180s"
  data_val = prepare_WAVs(path)


  #LSTM#


  epochs = 20
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

  # architecture:
  # input posklejany + wektor cyfr => output posklejany
  #
  # generate parameters [0...9]
  # inputs = inputs.view(1 * 4096, input_size, 1) + 1 dimension!
  # targets = targets.view(1 * 4096, 1)
  #
  print(data_train[0].shape)
  val = 1.0
  param = torch.full((150,1), val)
  print(param.shape)
  print(data_train[0].shape)

  # zaimplementowanie wstepnie w pliku: https://colab.research.google.com/drive/1dzlsXIMGQc_Vr2g8VlR8uwKCL8_uCPHN







'''


    print_interval = 100
    plot_interval = 10
    ###################
    # for plotting purposes:
    train_loss_plt =[]
    val_loss_plt =[]
    mse_sum_train = 0.0
    num_batches_train = 0
    ###################
    # Train Model ###################################################
    for epoch in range(epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):

            inputs = inputs.view(1 *4096, input_size , 1)
            targets = targets.view(1 *4096, 1)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            mse_sum_train += loss
            num_batches_train += 1

            # plot last validation loss:

            # if(epoch == epochs):
            #   val_loss_plt.append(val_loss.item())



            if (i + 1) % print_interval == 0: # every each 100 batches print and run validation!:

                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_arr)}], Loss: {loss.item():.4f}")

        #train_loss_plt.append(loss.item()) # add last loss value to plot
        average_mse_train = mse_sum_train / num_batches_train
        train_loss_plt.append(average_mse_train.item())



        # for each epoch calculate validation loss:
        mse_sum_val = 0.0
        num_batches_val = 0
        model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.view(1 * 4096, input_size , 1)
                targets = targets.view(1 * 4096, 1)

                outputs = model(inputs)
                val_loss = loss_fn(outputs,targets)

                mse_sum_val += val_loss
                num_batches_val +=1

                #plot last validation loss:

                #if(epoch == epochs):
                #   val_loss_plt.append(val_loss.item())
            average_mse_val = mse_sum_val / num_batches_val
            print(f"Average validation MSE: {average_mse_val:.4f}")
            #val_loss_plt.append(val_loss.item())
            val_loss_plt.append(average_mse_val.item())

    #AFTER ALL EPOCHS:
    epochs_axis = [x for x  in range(1, epochs+1)]


    plt.plot(epochs_axis, train_loss_plt,label = 'train_loss')
    plt.plot(epochs_axis, val_loss_plt, label='validation_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss [MSE]')
    plt.legend()
    plt.show()






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
    mse = loss_fn(prediction,sample_item[1])

    print(f"Test file  MSE: {mse:.4f}")

    prediction = prediction.numpy()


    # Save predictions as WAV
    sf.write(f'models/{name}/y_pred.wav', prediction, samplerate=44100)




'''