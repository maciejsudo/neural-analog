import torch.nn as nn
from model_architestures import GuitarLSTM, ESRLoss
from scipy.io import wavfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import soundfile as sf
import os
from itertools import repeat
from matplotlib import pyplot as plt


class WindowArrayDataset(Dataset):
    def __init__(self, x, y, window_len, batch_size):
        self.x = x
        self.y = y[window_len - 1:]
        self.window_len = window_len
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.x) - self.window_len + 1) // batch_size

    def __getitem__(self, index):
        temp = [(self.x[idx: idx + self.window_len]).clone().detach() for idx in
                range(index * self.batch_size, (index + 1) * self.batch_size)]
        # temp = [torch.tensor(self.x[idx: idx+self.window_len]) for idx in range(index*self.batch_size, (index+1)*self.batch_size)]
        x_out = torch.stack(temp)
        y_out = (self.y[index * self.batch_size:(index + 1) * self.batch_size]).clone().detach()
        # y_out = torch.tensor(self.y[index*self.batch_size:(index+1)*self.batch_size])
        return x_out, y_out


def normalize(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max, abs(data_min))
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

    in_data = audio_converter(in_data)
    out_data = audio_converter(out_data)

    X_all = in_data.astype(np.float32).flatten()
    X_all = normalize(X_all).reshape(len(X_all), 1)

    y_all = out_data.astype(np.float32).flatten()
    y_all = normalize(y_all).reshape(len(y_all), 1)

    return [X_all, y_all]


def read_WAVs(path_to_single_file, part):
    in_rate = None
    in_data = None

    # train/val/test  mode !!!
    # zeby dobrze dzielic na grupy!!

    in_file = path_to_single_file
    in_rate, in_data = wavfile.read(in_file)

    length = int(part * in_data.shape[0])

    in_data = audio_converter(in_data[:length])

    X_all = in_data.astype(np.float32).flatten()
    X_all = normalize(X_all).reshape(len(X_all), 1)

    X_all = torch.from_numpy(X_all)

    return X_all


def concat_WAV(data, value):
    '''
    # data = audio data tensor (23456,1)
    # value = tine. drive or level param
    '''
    audio_len = len(data)
    param = torch.full((audio_len, 1), value)
    return torch.cat((data, param), dim=1)


def return_n_data(full_data, div_factor, exact_number):
    '''
    # full data = the tensor (audio_values, params)
    # div_factor = the value dividing the full data size
    # exact number = [0...div_factor-1]
    '''
    full_size = list(full_data.size())[0]
    a = int(full_size / div_factor)
    return full_data[a * exact_number: a * (exact_number + 1)]


def return_all_filenames(path_to_dir):
    file_list = os.listdir(path_to_dir)
    return file_list


def parse_filename(filename):
    x = filename
    if (x[0] == 't'):  # for t1k_l1k_d1k_fs44100_bd16_time180s-target.wav
        idx1 = x.find("_l")
        idx2 = x.find("_d")
        idx3 = x.find("_fs")
        t = float(x[1:idx1 - 1])
        l = float(x[idx1 + 2:idx2 - 1])
        d = float(x[idx2 + 2:idx3 - 1])
    else:  # for gru-input.wav file:
        t = 0.0
        l = 0.0
        d = 0.0
    return t, l, d


def data_splitter(tensr, train_size, val_size):
    len = tensr.shape[0]
    train_len = int(train_size * len)
    val_len = int(val_size * len)

    return tensr[:train_len], tensr[train_len:train_len + val_len], tensr[train_len + val_len:]


def whole_dataset(path_to_dir, feature,data_size, train_size, val_size):

    train_list_tgt =[]
    train_list_inp =[]

    val_list_tgt =[]
    val_list_inp =[]

    test_list_tgt =[]
    test_list_inp =[]


    filelist = return_all_filenames(path_to_dir)

    tensor_list_tgt = []
    tensor_list_inp = []

    param =[]

    for x in filelist:
        t,l,d = parse_filename(x)

        if (feature == 'l'):
            param.append(l)
        elif (feature == 'd'):
            param.append(d)
        elif (feature == 't'):
            param.append(t)



    for x in filelist:

        t,l,d = parse_filename(x)

        #outputs are all files (with input also)

        data = read_WAVs(path_to_dir + str(x), data_size)
        train,val,test = data_splitter(data, train_size, val_size)

        train_list_tgt.append(train)
        val_list_tgt.append(val)
        test_list_tgt.append(test)


        #old# tensor_list_tgt.append(data)

        if(not(t) and not(l) and not(d)):
            for i in range(0, len(filelist)):
                data = read_WAVs(path_to_dir + str(x),data_size) # zaczytanie inputu o wartosci "0"
                train, val, test = data_splitter(data, train_size, val_size)

                train = concat_WAV(train, param[i])
                val = concat_WAV(val, param[i])
                test = concat_WAV(test, param[i])

                train_list_inp.append(train)
                val_list_inp.append(val)
                test_list_inp.append(test)

                #new_tnsr_2 = concat_WAV(data, param[i])
                #tensor_list_inp.append(new_tnsr_2)


    rslt_train_tgt = torch.cat(train_list_tgt, dim=0)
    rslt_train_inp = torch.cat(train_list_inp, dim=0)

    rslt_val_tgt = torch.cat(val_list_tgt, dim=0)
    rslt_val_inp = torch.cat(val_list_inp, dim=0)

    rslt_test_tgt = torch.cat(test_list_tgt, dim=0)
    rslt_test_inp = torch.cat(test_list_inp, dim=0)


    return [rslt_train_inp,rslt_train_tgt] , [rslt_val_inp,rslt_val_tgt], [rslt_test_inp,rslt_test_tgt]


def whole_dataset2(path_to_dir, feature, data_size, train_size, val_size):
    train_list_tgt = []
    train_list_inp = []

    val_list_tgt = []
    val_list_inp = []

    test_list_tgt = []
    test_list_inp = []

    filelist = return_all_filenames(path_to_dir)

    tensor_list_tgt = []
    tensor_list_inp = []

    param = []

    for x in filelist:
        t, l, d = parse_filename(x)

        if (feature == 'l'):
            param.append(l)
        elif (feature == 'd'):
            param.append(d)
        elif (feature == 't'):
            param.append(t)

    for x in filelist:

        t, l, d = parse_filename(x)

        # outputs are all files (with input also)
        print("values:",t,l,d)

        if(t!=0.0 and l!=0.0 and d!=0.0):
            data = read_WAVs(path_to_dir + str(x), data_size)
            train, val, test = data_splitter(data, train_size, val_size)

            train_list_tgt.append(train)
            val_list_tgt.append(val)
            test_list_tgt.append(test)

        # old# tensor_list_tgt.append(data)

        if (not (t) and not (l) and not (d)):
            for i in range(1, len(filelist)):
                data = read_WAVs(path_to_dir + str(x), data_size)  # zaczytanie inputu o wartosci "0"
                train, val, test = data_splitter(data, train_size, val_size)

                print('wartosci=',param[i])
                # tutaj pominą dla i=0 !!!

                train = concat_WAV(train, param[i])
                val = concat_WAV(val, param[i])
                test = concat_WAV(test, param[i])

                train_list_inp.append(train)
                val_list_inp.append(val)
                test_list_inp.append(test)

                # new_tnsr_2 = concat_WAV(data, param[i])
                # tensor_list_inp.append(new_tnsr_2)

    rslt_train_tgt = torch.cat(train_list_tgt, dim=0)
    rslt_train_inp = torch.cat(train_list_inp, dim=0)

    rslt_val_tgt = torch.cat(val_list_tgt, dim=0)
    rslt_val_inp = torch.cat(val_list_inp, dim=0)

    rslt_test_tgt = torch.cat(test_list_tgt, dim=0)
    rslt_test_inp = torch.cat(test_list_inp, dim=0)

    return [rslt_train_inp, rslt_train_tgt], [rslt_val_inp, rslt_val_tgt], [rslt_test_inp, rslt_test_tgt]


def save_wav(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))

def return_testfiles(path_to_dir):

    filelist = return_all_filenames(path_to_dir)

    filelist2 =[]

    for x in filelist:
        t, l, d = parse_filename(x)
        filelist2.append([x,t,l,d])
    return filelist2



def randomize(seg_len, test_tnsr, plot=False):
    # czas 1 sekundy!:
    #seg_len = 44100
    original_tensor = test_tnsr[0]

    length = test_tnsr[1].shape[0]

    tensor1 = test_tnsr[1].reshape(int(length / seg_len), seg_len)
    tensor2 = original_tensor[:, 0].unsqueeze(dim=1).reshape(int(length / seg_len), seg_len)
    tensor3 = original_tensor[:, 1].unsqueeze(dim=1).reshape(int(length / seg_len), seg_len)

    concatenated_tensor = torch.stack([tensor1, tensor2, tensor3], dim=1)
    permutation = torch.randperm(concatenated_tensor.size(0))
    shuffled_tensor = concatenated_tensor[permutation]

    # Split the shuffled tensor back into three tensors
    shuffled_tensor1, shuffled_tensor2, shuffled_tensor3 = torch.unbind(shuffled_tensor, dim=1)

    new_tensr = torch.stack([shuffled_tensor2, shuffled_tensor3], dim=2)

    #number_of_segments = tensor1.shape[0]


    if(plot==True):
        unique_values, _ = torch.unique(shuffled_tensor3, dim=1, return_inverse=True)
        plot_values = unique_values.squeeze()
        input = plot_values.to(torch.int)
        plt.plot(input.numpy(), label='training parameters')
        plt.xlabel('file length')
        plt.ylabel('parameter value')
        plt.legend()
        plt.show()

    new_tensr = new_tensr.reshape(-1,2)
    shuffled_tensor1 = shuffled_tensor1.reshape(-1,1)

    return [new_tensr,shuffled_tensor1]

def perform_tests(path_to_dir,feature,number_in_dir, model, name):
    # test_tnsr = test_tnsr[1]!!!
    # feature = 'level',number_in_dir=1,test_tnsr =
    database_names = return_testfiles(path_to_dir)
    y = number_in_dir
    x = database_names[y][0]

    if feature == 'tone':
        value = database_names[y][1]
    elif feature == 'level':
        value = database_names[y][2]
    elif feature == 'drive':
        value = database_names[y][3]

    print(f"Performing testing for: No.{y}. {x} recording, with {feature}=",value)
    # this data must be input0 !!! # database_names[0][0] = input.wav

    data = read_WAVs(path_to_dir + database_names[0][0], data_size)
    output_param =  read_WAVs(path_to_dir + x, data_size)

    input0 = concat_WAV(data, value)
    print(input0)
    print(output_param)


    test_arr = WindowArrayDataset(input0, output_param, 150, batch_size=204800)

    model.load_state_dict(torch.load(f'models/{name}/{name}.pth'))
    model.eval()

    index = 0
    sample_item = test_arr[index]

    # Make predictions
    with torch.no_grad():
        prediction = model(sample_item[0])

    # Convert PyTorch tensor to numpy array
    mse = loss_fn(prediction, sample_item[1])

    print(f"Test file  MSE: {mse:.4f}")

    prediction = prediction.numpy()

    original_wav = sample_item[1].numpy()

    return [original_wav, prediction]



if __name__ == "__main__":

    name = 'test1'

    #path = "D:\\repos\MGR\\neural-analog\data\\test\\t20k_l100k_d100k_fs44100_bd16_time180s"
    #data_test = prepare_WAVs(path)

    #path = "D:\\repos\MGR\\neural-analog\data\\train\\t20k_l100k_d100k_fs44100_bd16_time180s"
    #data_train = prepare_WAVs(path)

    #path = "D:\\repos\MGR\\neural-analog\data\\validation\\t20k_l100k_d100k_fs44100_bd16_time180s"
    #data_val = prepare_WAVs(path)

    '''
    LSTM
    Parameters:
    '''
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
    print('Model architcture: \n', model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # training loader:
    # train_arr = WindowArrayDataset(data_train[0], data_train[1], input_size, batch_size= batch_size)
    # train_loader = DataLoader(train_arr, batch_size=1, shuffle=False)
    # #validation loader:
    # val_arr = WindowArrayDataset(data_val[0], data_val[1], input_size, batch_size= batch_size)
    # val_loader = DataLoader(val_arr, batch_size=1, shuffle=False)

    path_to_dir = "D:\\repos\MGR\\neural-analog\data\\parametric\\"



    data_size = 0.1

    train_size = 0.5
    val_size = 0.25
    # test size is what is left! in this situation its 0.25 of each file

    train_tnsr, val_tnsr, test_tnsr = whole_dataset(path_to_dir, feature='l', data_size=data_size,
                                                    train_size=train_size, val_size=val_size)



    

    print(f'train_size = {int(train_size * 100)} % that means {train_tnsr[0].shape[0]} samples')

    print(f'val_size = {int(val_size * 100)}% that means {val_tnsr[0].shape[0]} samples')

    print(f'test_size = {int((1 - train_size - val_size) * 100)} % that means {test_tnsr[0].shape[0]} samples')

    #train_tnsr = randomize(seg_len=44100, test_tnsr=train_tnsr, plot=False)



    # 50 % stanowi zbiór trenignowy!:
    train_arr = WindowArrayDataset(train_tnsr[0], train_tnsr[1], input_size, batch_size=batch_size)
    train_loader_param = DataLoader(train_arr, batch_size=1, shuffle=True)

    # 30 % stanowi zbiór walidacyjny!:
    print(val_tnsr[0].shape, val_tnsr[1].shape)

    val_arr = WindowArrayDataset(val_tnsr[0], val_tnsr[1], input_size, batch_size=batch_size)
    val_loader_param = DataLoader(val_arr, batch_size=1, shuffle=True)

    print_interval = 100
    plot_interval = 10
    ###################
    # for plotting purposes:
    train_loss_plt = []
    val_loss_plt = []
    mse_sum_train = 0.0
    num_batches_train = 0
    ###################
    # Train Model ###################################################
    for epoch in range(epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader_param):

            inputs = inputs.view(1 * 4096, input_size, 2)
            targets = targets.view(1 * 4096, 1)

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

            if (i + 1) % print_interval == 0:  # every each 100 batches print and run validation!:

                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_arr)}], Loss: {loss.item():.4f}")

        # train_loss_plt.append(loss.item()) # add last loss value to plot
        average_mse_train = mse_sum_train / num_batches_train
        train_loss_plt.append(average_mse_train.item())

        # for each epoch calculate validation loss:
        mse_sum_val = 0.0
        num_batches_val = 0
        model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader_param):
                inputs = inputs.view(1 * 4096, input_size, 2)
                targets = targets.view(1 * 4096, 1)

                outputs = model(inputs)
                val_loss = loss_fn(outputs, targets)

                mse_sum_val += val_loss
                num_batches_val += 1

                # plot last validation loss:

                # if(epoch == epochs):
                #   val_loss_plt.append(val_loss.item())
            average_mse_val = mse_sum_val / num_batches_val
            print(f"Average validation MSE: {average_mse_val:.4f}")
            # val_loss_plt.append(val_loss.item())
            val_loss_plt.append(average_mse_val.item())

    # AFTER ALL EPOCHS:
    epochs_axis = [x for x in range(1, epochs + 1)]

    plt.plot(epochs_axis, train_loss_plt, label='train_loss')
    plt.plot(epochs_axis, val_loss_plt, label='validation_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss [MSE]')
    plt.legend()
    plt.show()

    # Save trained model
    new_dir = os.path.join('models', name)
    os.makedirs(new_dir, exist_ok=True)

    torch.save(model.state_dict(), f'models/{name}/{name}.pth')

    original_wav, prediction = perform_tests(path_to_dir,'level',1, model, name)
    original_wav, prediction = perform_tests(path_to_dir, 'level', 2, model, name)
    #original_wav, prediction = perform_tests(path_to_dir, 'level', 3, model, name)

    sf.write(f'models/{name}/y_pred.wav', prediction, samplerate=44100)
    sf.write(f'models/{name}/y_original.wav', original_wav, samplerate=44100)



