import time
import torch, numpy, random, os, math, glob, soundfile
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from torchvision import datasets, transforms
import torchaudio
import torch.nn.functional as F

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class CustomAudioTransform:
    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomCrop(CustomAudioTransform):
    def __init__(self, size:int, pad:bool = True):
        self.size = size
        self.pad = pad

    def __call__(self, signal):
        # if signal.shape[0] < self.size :
        #     if self.pad:
        #         signal = F.pad(signal, (0, self.size-signal.shape[0]))     #pad 0是否有损伤？ 但不会有需要pad的情况，因为load时已经检查过长度
        #     return signal
        start = numpy.random.randint(0, signal.shape[-1] - self.size + 1)
        return signal[start: start + self.size]


#class S3RTTransform: 不好单独写muti_crop + 数据增强，因为还要读取声音文件加噪


#相当于把transform直接写在dataset里了 又叫做在线数据争强，可能会有性能瓶颈
class train_dataset(Dataset):
    def __init__(self, train_list, train_path, musan_path, input_fdim=80, max_frames=300, global_crops_scale=3, local_crops_scale=2, local_crops_number=0, **kwargs):
        self.max_frames = max_frames  # 300就是3s，输入网络的最大长度,最大长度其实就是teacher的输入
        self.data_list = []
        self.noisetypes = ['noise','speech', 'music'] # Type of noise
        #self.noisetypes = ['noise']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]} # The range of SNR
        #self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}   #[3,8]表示在3个到8个之间采样k个noise files
        self.noiselist = {} 
        #self.i_time, self.i = 0, 0

        self.local_crops_number = local_crops_number
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav')) # All noise files in list

        # fbank
        self.mel_feature = torch.nn.Sequential(
            PreEmphasis(),        #预加重 no large influence     
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=input_fdim),
            ) ## (channel, n_mels, time)
        #注意， torchaudio.compliance.kaldi.fbank 得到的是 （time, n_mels）并且通道融合了？？
        # global crop    
        self.global_transfo = transforms.Compose(
                                            [RandomCrop(16000 * global_crops_scale),
                                             #self.mel_feature
                                            ]) #3s
        # transformation for the local small crops
        self.local_transfo = transforms.Compose(
                                            [RandomCrop(16000 * local_crops_scale),
                                             #self.mel_feature
                                            ]) #2s

        for file in augment_files:
            if not file.split('/')[-3] in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file) # All noise files in dic
        self.rir_files = numpy.load('rir.npy') # Load the rir file (1000, 11200)
        for line in open(train_list).read().splitlines():
            filename = os.path.join(train_path, line.split()[1])
            self.data_list.append(filename) # Load the training data list
        
    def __getitem__(self, index): #返回列表
        audio = self.loadWAV(self.data_list[index], self.max_frames)
        # Data Augmentation

        crops = []

        for ii in range(0, 2 + self.local_crops_number): # 2 + self.local_crops_number segments , muticrop   这个每次load的数据增强最花时间，即使local_crops_number为0
            augment_profiles = {}
            rir_gains = numpy.random.uniform(-7,3,1)
            rir_filts = random.choice(self.rir_files)
            noisecat    = random.choice(self.noisetypes)
            noisefile   = random.choice(self.noiselist[noisecat].copy()) # Augmentation information for each segment
            snr = [random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])]
            p = random.random()
            if p < 0.25:  # Add rir only
                augment_profiles = {'rir_filt':rir_filts, 'rir_gain':rir_gains, 'add_noise': None, 'add_snr': None}
            elif p < 0.50: # Add noise only
                augment_profiles = {'rir_filt':None, 'rir_gain':None, 'add_noise': noisefile, 'add_snr': snr}
            else: # Add both
                augment_profiles = {'rir_filt':rir_filts, 'rir_gain':rir_gains, 'add_noise': noisefile, 'add_snr': snr}
            
            if ii <= 1:
                crop = self.global_transfo(audio)
            else:
                crop = self.local_transfo(audio)
            aug_audio = self.augment_wav(crop, augment_profiles) #(1,len)  #主要的时间开销！！！
            crops.append(aug_audio)
            # with torch.no_grad():  
            #     fbank = self.mel_feature(aug_audio) + 1e-6
            #     #normalize
            #     fbank = fbank.log()   
            #     fbank = fbank - torch.mean(fbank, dim=-1, keepdim=True) # (1, n_mel, t)
            #     #print("fbank.shape", fbank.shape)
            #     crops.append(fbank)
        return crops

    def __len__(self):
        return len(self.data_list)

    def augment_wav(self, audio, augment):
        if augment['rir_filt'] is not None:
            rir     = numpy.multiply(augment['rir_filt'], pow(10, 0.1 * augment['rir_gain']))    
            #print(rir.shape, audio.shape)
            audio   = signal.convolve(audio, rir, mode='full')[:len(audio)]
        if augment['add_noise'] is not None:
            #print(audio.shape)
            noiseaudio  = self.loadWAV(augment['add_noise'], self.max_frames)    #这个最慢，为啥？ 因为有一些噪声音频很长，还要完整读入，是不是没必要？
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
            noise = numpy.sqrt(10 ** ((clean_db - noise_db - augment['add_snr']) / 10)) * noiseaudio
            audio = audio + noise[:len(audio)]    #不同crop用的noise长度不一样
        
        audio = numpy.expand_dims(audio, 0)
        return torch.FloatTensor(audio) #(1，len) 

    # #上下这两个数据增强实现有啥区别？ 可以换一下试试后面
    # def add_noise(self, audio, noisecat):
    #     clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)  #计算原声平均分贝，audio ** 2是声音的能量
    #     numnoise    = self.numnoise[noisecat]   
    #     noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
    #     noises = []
    #     for noise in noiselist:
    #         noiseaudio, _ = soundfile.read(noise)
    #         length = self.num_frames * 160 + 240 #num_frames 200就是裁减两秒，因为默认采样率是16000，hop_len是160,即每次平移10ms，即1s就有100个frame
    #         if noiseaudio.shape[0] <= length:
    #             shortage = length - noiseaudio.shape[0]
    #             noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
    #         start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
    #         noiseaudio = noiseaudio[start_frame:start_frame + length]
    #         noiseaudio = numpy.stack([noiseaudio],axis=0)

    #         #调整noiseaudio的分贝
    #         noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
    #         noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
    #         noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
    #     noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
    #     return noise + audio

    # def add_rev(self, audio):
    #     rir_file    = random.choice(self.rir_files)
    #     rir, sr     = soundfile.read(rir_file)
    #     rir         = numpy.expand_dims(rir.astype(numpy.float),0)
    #     rir         = rir / numpy.sqrt(numpy.sum(rir**2))
    #     return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]  #卷积

    def loadWAV(self, filename, max_frames):
        max_audio = max_frames * 160 + 240  # 240 is for padding, for 15ms since window is 25ms and step is 10ms.
        #s = time.time()
        audio, _ = soundfile.read(filename)
        #self.i_time += (time.time() - s)
        audiosize = audio.shape[0]

        if audiosize <= max_audio: # Padding if the length is not enough
            shortage    = math.floor( ( max_audio - audiosize + 1 ) / 2 )
            audio       = numpy.pad(audio, (shortage, shortage), 'wrap')
            audiosize   = audio.shape[0]
        startframe = numpy.int64(random.random() * (audiosize - max_audio)) # Randomly select a start frame to extract audio
        wavform = audio[int(startframe):int(startframe) + max_audio]
        return wavform #(len,)

def eval_transform(filename):
    wavform, _ = soundfile.read(filename)
    wavform = torch.FloatTensor(numpy.stack([wavform], axis=0)) #(1, len)
    # mel_feature = torch.nn.Sequential(
    #         PreEmphasis(),        #预加重 no large influence     
    #         torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
    #                                              f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=input_fdim),
    #         )
    # with torch.no_grad():
    #     fbank = mel_feature(wavform) + 1e-6
    #     #normalize
    #     fbank = fbank.log()   
    #     fbank = fbank - torch.mean(fbank, dim=-1, keepdim=True) # (1, n_mel, t)
    #return fbank.unsqueeze(0)
    return wavform.unsqueeze(0) # (1, 1, len)
def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

def get_loader(args): # Define the data loader
    tdataset = train_dataset(**vars(args))
    print(f"Data loaded: there are {len(tdataset)} audios.")

    sampler = torch.utils.data.DistributedSampler(tdataset, shuffle=True)

    trainLoader = torch.utils.data.DataLoader(
        tdataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=5,
    )
    return trainLoader

if __name__ == '__main__':
    datasets = train_dataset(train_list='list/voxceleb1_train_list', train_path='/data/voxceleb', musan_path='/data/musan_split')
    #,'music'
    # tt1 = tt2 = 0
    # for i in range(200):
    #     s = time.time()
    #     a1, _ = soundfile.read("/data/musan/music/fma/music-fma-0121.wav")
    #     tt1 += (time.time() - s)

    #     s = time.time()
    #     a1, _ = soundfile.read("/data/musan/noise/free-sound/noise-free-sound-0000.wav")
    #     tt2 += (time.time() - s)

    # print(tt1, tt2)

    t = 0
    s = time.time()
    for i in range(2000):
        a1 = datasets[i]
    t += (time.time() - s)

    print(t)