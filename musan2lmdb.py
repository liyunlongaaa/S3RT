import torch, numpy, random, os, math, glob, soundfile, sys
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import lmdb
import pickle

import tqdm
import pyarrow as pa

import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np



class musanDataset(data.Dataset):
    def __init__(self, musan_path):
        self.data_list = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
        self.prefixLen = len(musan_path) + 1
                
    def __getitem__(self, index):
        waveform, sr = soundfile.read(self.data_list[index]) # Load one utterance
        return waveform, self.data_list[index][self.prefixLen:]

    def __len__(self):
        return len(self.data_list)
    
def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer() #pyarrow.serialize creates an intermediate object which can be converted to a buffer (the to_buffer method) or written directly to an output stream.

def folder2lmdb(dpath, lmdb_path=None, write_frequency=500, max_num=3000, num_workers=5, **kw):

    if lmdb_path is None:
        lmdb_path = dpath
    ds = musanDataset(**kw)
    #dataloader = DataLoader(ds,num_workers=num_workers,shuffle=True)

    if len(ds) > max_num:
        lmdb_split = 0
        lmdb_path = os.path.join(dpath, "data_{}.lmdb".format(lmdb_split))
        lmdb_split += 1
    else:
        lmdb_path = os.path.join(dpath, "data.lmdb")

    if os.path.exists(lmdb_path):
        print("{} already exists".format(lmdb_path))
        exit(0)

    isdir = os.path.isdir(lmdb_path) #False

    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 // 4, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    keys = []
    for idx, data in enumerate(ds):
        print(idx)
        waveform, name = data[0], data[1]
        keys.append(u'{}'.format(name).encode('ascii'))
        txn.put(key=u'{}'.format(name).encode('ascii'), value=dumps_pyarrow(waveform))
        if idx >0 and idx % max_num ==0: #超过最大文件数，另外建一个lmdb文件

            txn.commit()
            print("commited")
            with db.begin(write=True) as txn:
                txn.put(b'__keys__', dumps_pyarrow(keys))
                txn.put(b'__len__', dumps_pyarrow(len(keys))) #文件长度
            print("Flushing database to {} ...".format(lmdb_path))
            db.sync()
            db.close()

            lmdb_path = os.path.join(dpath, "data_{}.lmdb".format(lmdb_split))
            lmdb_split += 1
            isdir = os.path.isdir(lmdb_path)

            db = lmdb.open(lmdb_path, subdir=isdir,
                        map_size=1099511627776 // 18, readonly=False,
                        meminit=False, map_async=True)

            txn = db.begin(write=True)
            keys = []

        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(ds)))
            txn.commit()
            txn = db.begin(write=True)
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

    return ds

class musanLMDBDataset(data.Dataset):
    def __init__(self, db_path):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length =pa.deserialize(txn.get(b'__len__'))


    def __getitem__(self, filename):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(filename.encode())
        unpacked = pa.deserialize(byteflow)
        waveform = unpacked
        return waveform

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')' #定义输出类时的信息
    

if __name__ == '__main__':
    #folder2lmdb(dpath = '/home/yoos/Downloads/vox1_train_lmdb', train_list='/data/voxceleb/voxceleb1_train_list', train_path='/data/voxceleb/voxceleb1')

    #dataset = LMDBDataset('/home/yoos/Downloads/vox1_train_lmdb/data.lmdb')
    #print(len(dataset))
    #wa, sr = soundfile.read('/data/voxceleb/voxceleb1/id10001/1zcIwhmdeo4/00001.wav')
    #print(dataset[0].shape, wa.shape, wa == dataset[0])
    #folder2lmdb(dpath = '/home/yoos/Downloads/musan_lmdb', musan_path='/data/musan')
    dataset = musanLMDBDataset('/home/yoos/Downloads/musan_lmdb/data.lmdb')
    print(len(dataset))
    wa, sr = soundfile.read('/data/musan/music/fma/music-fma-0010.wav')
    print(dataset['music/fma/music-fma-0010.wav'].shape, wa.shape, wa == dataset['music/fma/music-fma-0010.wav'])
