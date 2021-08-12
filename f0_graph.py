# 베를린 dataset 오디오 파일 1개 불러오기
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

path = '/home/suser/dataset/audio_folder/berlin/wav/03a01Fa.wav'
y, sr = librosa.load(path)
f0,voiced_flag, voiced_probs = librosa.pyin(y,
                                            sr = 16000,
                                            frame_length = 2048,
                                            hop_length = 512,
                                            fmin=librosa.note_to_hz('C0'), 
                                            fmax=librosa.note_to_hz('C7'))
f0 = np.nan_to_num(f0,copy=True)

print('f0 : ', f0)

times = librosa.times_like(f0)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
fig, ax = plt.subplots(figsize=(12,3))
img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
ax.set(title='fundamental frequency estimation')
fig.colorbar(img, ax=ax, format="%+2.f dB")
ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
ax.legend(loc='upper right')
plt.show()
plt.savefig("/home/suser/minsun/f0_graph/file name : [{}].png".format(path.split("/")[7]))

###############################################################################

# kesdy18 폴더 전체 불러오기

path = '/home/suser/dataset/audio_folder/KESDy18'

def f0_graph(data):
    f0,voiced_flag, voiced_probs = librosa.pyin(y,
                                                sr = 16000,
                                                frame_length = 2048,
                                                hop_length = 512,
                                                fmin=librosa.note_to_hz('C0'), 
                                                fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0,copy=True)
    
    times = librosa.times_like(f0,sr = sr)
    # spectrogram 생성
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(12,3))
    img = librosa.display.specshow(D,sr =sr, x_axis='time', y_axis='log', ax=ax, fmax=8192)
    ax.set(title='fundamental frequency estimation')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')
    plt.show()
    plt.savefig("/home/suser/minsun/f0_graph/file name : [{}].png".format(path.split("/")[9]))
    print('done')
    return f0

st = librosa.util.find_files(path, ext=['wav'])
st = np.asarray(st)


for st_file in st:
    if st_file.split("/")[7] == 'st':
        y, sr = librosa.load(st_file, sr = 16000)
        file_name = st_file.split("/")[9]
        print('파일 이름 : ',file_name)
        f0,voiced_flag, voiced_probs = librosa.pyin(y,
                                                sr = 16000,
                                                frame_length = 2048,
                                                hop_length = 512,
                                                fmin=librosa.note_to_hz('C0'), 
                                                fmax=librosa.note_to_hz('C7'))
        f0 = np.nan_to_num(f0,copy=True)

        times = librosa.times_like(f0,sr = sr)
        # spectrogram 생성
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig, ax = plt.subplots(figsize=(12,3))
        img = librosa.display.specshow(D,sr =sr, x_axis='time', y_axis='log', ax=ax, fmax=8192)
        ax.set(title='fundamental frequency estimation')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
        ax.legend(loc='upper right')
        plt.show()
        plt.savefig("/home/suser/minsun/f0_graph/file name : [{}].png".format(file_name))
        print('이미지 저장 완료')

