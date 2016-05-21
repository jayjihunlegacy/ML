import pickle
import numpy as np

def get_cifar10():
    folder='E:/Datasets/CIFAR-10/'
    trainnames=['data_batch_%i'%(i+1,) for i in range(5)]
    testname='test_batch'

    datas=tuple()
    labels=tuple()
    for filename in trainnames:
        full=folder+filename
        with open(full,'rb') as f:
            u=pickle._Unpickler(f)
            u.encoding='latin1'
            input=u.load()

        datas+=(input['data'],)
        labels+=(input['labels'],)
    data = np.concatenate(datas,axis=0)
    label = np.concatenate(labels,axis=0)

    num_of_data = data.shape[0]
    num_of_train=num_of_data * 9 //10
    train_x = data[0:num_of_train]
    train_y = label[0:num_of_train]
    trainset = (train_x, train_y)

    valid_x = data[num_of_train:]
    valid_y = label[num_of_train:]
    validset = (valid_x, valid_y)

    full=folder+testname
    with open(full,'rb') as f:
        u=pickle._Unpickler(f)
        u.encoding='latin1'
        input=u.load()
    test_x=input['data']
    test_y=input['labels']
    testset = (test_x, test_y)
    print('CIFAR10 imported')
    return trainset, validset, testset

def get_cifar100():
    pass


def get_cifar(version):
    if version==10:
        return get_cifar10()
    elif version==100:
        return get_cifar100()
    else:
        return None