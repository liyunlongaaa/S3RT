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

import musan2lmdb

class train_dataset(Dataset):
    def __init__(self, max_frames, musan_path, vox_lmdb_path, musan_lmdb_path, **kwargs):
        self.max_frames = max_frames
        self.voxdataset = voxLMDBDataset(vox_lmdb_path)
        self.musan_dataset = musan2lmdb.musanLMDBDataset(musan_lmdb_path)
        self.data_list = []
        self.noisetypes = ['noise','speech','music'] # Type of noise
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]} # The range of SNR
        self.noiselist = {} 
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav')) # All noise files in list
        for file in augment_files:
            if not file.split('/')[-3] in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file[12:]) # All noise files in dic
        self.rir_files = numpy.load('rir.npy') # Load the rir file
                
    def __getitem__(self, index):
        audio = self.voxdataset[index]
        audio = self.loadWAVSplit(audio, self.max_frames).astype(numpy.float) # Load one utterance
        augment_profiles, audio_aug = [], []
        for ii in range(0,2): # Two segments of one utterance
            rir_gains = numpy.random.uniform(-7,3,1)
            rir_filts = random.choice(self.rir_files)
            noisecat    = random.choice(self.noisetypes)
            noisefile   = random.choice(self.noiselist[noisecat].copy()) # Augmentation information for each segment
            snr = [random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])]
            p = random.random()
            if p < 0.25:  # Add rir only
                augment_profiles.append({'rir_filt':rir_filts, 'rir_gain':rir_gains, 'add_noise': None, 'add_snr': None})
            elif p < 0.50: # Add noise only
                augment_profiles.append({'rir_filt':None, 'rir_gain':None, 'add_noise': noisefile, 'add_snr': snr})
            else: # Add both
                augment_profiles.append({'rir_filt':rir_filts, 'rir_gain':rir_gains, 'add_noise': noisefile, 'add_snr': snr})
        audio_aug.append(self.augment_wav(audio[0],augment_profiles[0])) # Segment 0 with augmentation method 0
        audio_aug.append(self.augment_wav(audio[1],augment_profiles[0])) # Segment 1 with augmentation method 0, used for AAT
        audio_aug.append(self.augment_wav(audio[1],augment_profiles[1])) # Segment 1 with augmentation method 1
        audio_aug = numpy.concatenate(audio_aug,axis=0) # Concate and return
        return torch.FloatTensor(audio_aug)

    def __len__(self):
        return self.voxdataset.__len__()

    def augment_wav(self,audio, augment):
        if augment['rir_filt'] is not None:
            rir     = numpy.multiply(augment['rir_filt'], pow(10, 0.1 * augment['rir_gain']))    
            audio   = signal.convolve(audio, rir, mode='full')[:len(audio)]
        if augment['add_noise'] is not None:
            noiseaudio  = self.loadWAV(augment['add_noise'], self.max_frames, load_type="aug").astype(numpy.float)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
            noise = numpy.sqrt(10 ** ((clean_db - noise_db - augment['add_snr']) / 10)) * noiseaudio
            audio = audio + noise
        else:
            audio = numpy.expand_dims(audio, 0)
        return audio

    def loadWAV(self, filename, max_frames, load_type="aug"):
        max_audio = max_frames * 160 + 240 # 240 is for padding, for 15ms since window is 25ms and step is 10ms.
        if load_type == "aug":
            audio = self.musan_dataset[filename]
        else:
            audio = self.voxdataset[filename]
        audiosize = audio.shape[0]
        if audiosize <= max_audio: # Padding if the length is not enough
            shortage    = math.floor( ( max_audio - audiosize + 1 ) / 2 )
            audio       = numpy.pad(audio, (shortage, shortage), 'wrap')
            audiosize   = audio.shape[0]
        startframe = numpy.int64(random.random()*(audiosize-max_audio)) # Randomly select a start frame to extract audio
        feat = numpy.stack([audio[int(startframe):int(startframe)+max_audio]],axis=0)
        return feat

    def loadWAVSplit(self, audio, max_frames): # Load two segments
        max_audio = max_frames * 160 + 240
        audiosize = audio.shape[0]
        if audiosize <= max_audio:
            shortage    = math.floor( ( max_audio - audiosize) / 2 )
            audio       = numpy.pad(audio, (shortage, shortage), 'wrap')
            audiosize   = audio.shape[0]
        randsize = audiosize - (max_audio*2) # Select two segments
        startframe = random.sample(range(0, randsize), 2)
        startframe.sort()
        startframe[1] += max_audio # Non-overlapped two segments
        startframe = numpy.array(startframe)
        numpy.random.shuffle(startframe)
        feats = []
        for asf in startframe: # Startframe[0] means the 1st segment, Startframe[1] means the 2nd segment
            feats.append(audio[int(asf):int(asf)+max_audio])
        feat = numpy.stack(feats,axis=0)
        return feat

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

    
def get_loader(args): # Define the data loader
    dataset = train_dataset(**vars(args))
    trainLoader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=5,
    )
    return trainLoader

class vox1Dataset(data.Dataset):
    def __init__(self, train_list, train_path):
        self.data_list, self.name_suffix = [], []
        for line in open(train_list).read().splitlines():
            filename = os.path.join(train_path, line.split()[1])
            self.name_suffix.append(line.split()[1])
            self.data_list.append(filename) # Load the training data list
                
    def __getitem__(self, index):
        waveform, sr = soundfile.read(self.data_list[index]) # Load one utterance
        return waveform, self.name_suffix[index] 

    def __len__(self):
        return len(self.data_list)
    

def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer() #pyarrow.serialize creates an intermediate object which can be converted to a buffer (the to_buffer method) or written directly to an output stream.

def folder2lmdb(dpath, lmdb_path=None, write_frequency=5000, max_num=200000, num_workers=5, **kw):

    if lmdb_path is None:
        lmdb_path = dpath
    ds = vox1Dataset(**kw)
    #dataloader = DataLoader(ds,num_workers=num_workers,shuffle=True)

    if len(ds) > max_num:
        lmdb_split = 0
        lmdb_path = os.path.join(lmdb_path, "data_{}.lmdb".format(lmdb_split))
        lmdb_split += 1
    else:
        lmdb_path = os.path.join(lmdb_path, "data.lmdb")

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
        waveform, name = data[0], data[1]
        keys.append(u'{}'.format(name).encode('ascii'))
        txn.put(key=u'{}'.format(name).encode('ascii'), value=dumps_pyarrow(waveform))
        if idx >0 and idx % max_num ==0: #超过最大文件数，另外建一个lmdb文件

            txn.commit()
            with db.begin(write=True) as txn:
                txn.put(b'__keys__', dumps_pyarrow(keys))
                txn.put(b'__len__', dumps_pyarrow(len(keys))) #文件长度
            print("Flushing database to {} ...".format(lmdb_path))
            db.sync()
            db.close()

            lmdb_path = os.path.join(lmdb_path, "{}.lmdb".format(lmdb_split))
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


class voxLMDBDataset(data.Dataset):
    def __init__(self, db_path):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length =pa.deserialize(txn.get(b'__len__'))
            self.keys= pa.deserialize(txn.get(b'__keys__'))


    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        waveform = unpacked

        return waveform

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')' #定义输出类时的信息
    

if __name__ == '__main__':
    #make lmdb dataset and test 

    # folder2lmdb(dpath = '/home/yoos/Downloads/vox1_train_lmdb', train_list='/data/voxceleb/voxceleb1_train_list', train_path='/data/voxceleb/voxceleb1')

    # dataset = voxLMDBDataset('/home/yoos/Downloads/vox1_train_lmdb/data.lmdb')
    # print(len(dataset))
    # wa, sr = soundfile.read('/data/voxceleb/voxceleb1/id10001/1zcIwhmdeo4/00001.wav')
    # print(dataset[0].shape, wa.shape, wa == dataset[0])

    dataset = train_dataset(max_frames=300, train_list='/data/voxceleb/voxceleb1_train_list', train_path='/data/voxceleb/voxceleb1', musan_path='/data/musan', vox_lmdb_path='/home/yoos/Downloads/vox1_train_lmdb/data.lmdb', musan_lmdb_path="/home/yoos/Downloads/musan_lmdb/data.lmdb")
    print(dataset[0])