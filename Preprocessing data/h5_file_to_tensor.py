import torch
import numpy as np
import h5py

def preprocessing(filename,group):
    h5file = h5py.File(filename, 'r')
    dataset = h5file.get(group)
    y = dataset['hsALT']
    spectrum = dataset['spectrum']

    names = ['cwave', 'dxdt', 'latlonSARcossin', 'todSAR', 'incidence', 'satellite']
    features = []
    for name in names:
        if name in dataset:
            temp = dataset[name]
        elif name == 'dxdt':
            temp = np.zeros_like(dataset['incidence'])
        else:
            raise Exception
        features.append(temp)
    features = np.hstack(features)

    del_idx = np.argwhere(np.isnan(np.mean(features,axis=1)))
    y = np.delete(y,del_idx,0)
    spectrum = np.delete(spectrum,del_idx,0)
    features = np.delete(features,del_idx,0)

    assert features.shape[1] == 32, features.shape
    assert not np.any(np.isnan(features))
    assert not np.any(features > 1000), features.max()

    assert y.shape[1] == 1
    assert not np.any(np.isnan(y))
    assert not np.any(y > 100), y
    
    assert not np.any(np.isnan(spectrum))

    # return [spectrum[:],features[:]],y[:]
    return [torch.from_numpy(np.transpose(spectrum[:],(0,3,1,2))),torch.from_numpy(features[:]).float()], torch.from_numpy(y[:])


x, y = preprocessing(filename='/content/drive/MyDrive/sar_img/aggregated_grouped_final.h5', group='2018')
torch.save({"x1": x[0], "x2":x[1], "y": y}, '/content/drive/MyDrive/sar_img/tensors_2018.pt')


x, y = preprocessing(filename='/content/drive/MyDrive/sar_img/aggregated_grouped_final.h5', group='2017')
torch.save({"x1": x[0], "x2":x[1], "y": y}, '/content/drive/MyDrive/sar_img/tensors_2017.pt')

x, y = preprocessing(filename='/content/drive/MyDrive/sar_img/aggregated_grouped_final.h5', group='2015_2016')
torch.save({"x1": x[0], "x2":x[1], "y": y}, '/content/drive/MyDrive/sar_img/tensors_2015_2016.pt')