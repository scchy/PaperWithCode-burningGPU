# WenZhou2Putong Proj

# Some Preblems
1. How to transfer audio to tensor?
    - [audio_io_tutorial](https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html)
2. How to transfer bytes to audio?
    - [creating-wav-file-from-bytes](https://stackoverflow.com/questions/52369925/creating-wav-file-from-bytes)
    - method1: bytes -> bytearray -> np.array -> wavfile: can not work well
    - method2: byte_io -> wavfile.write bytes -> byte_io.read() -> io.BytesIO -> torchaudio.load
        - some problem: some head in the bytes.
```python
def bytes2pt(b_in, use_tmp_file=False, tmp_file='./tmp.wav'):
    # method-1
    b_arr = bytearray(b_in)
    np_arr = np.array(b_arr, dtype=np.int16)
    if use_tmp_file:
        wavfile.write(tmp_file, 16000, np_arr)
        waveform, sample_rate = torchaudio.load(tmp_file)
        return waveform, sample_rate 

    # method-2
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    wavfile.write(byte_io, 16000, np_arr)
    # output_wav = byte_io.read()
    # waveform, sample_rate = torchaudio.load(io.BytesIO(output_wav))
    waveform, sample_rate = torchaudio.load(byte_io)
    return waveform, sample_rate 

waveform1, sample_rate = bytes2pt(df.iloc[0]['audio']['bytes'], True)
waveform2, sample_rate = bytes2pt(df.iloc[0]['audio']['bytes'], False)
b_in_head = b'RIFF$\xc7\x00\x00WAVEfmt \x10\'
```
    - 

# 0. Thinking
## using pretrain
- data clear
- stastic of word 

## data Augment
- label Aug: speech -> text ( trans(text) -> back(text))



# 1. Putong2Putong
## 1.1 data Prepare
### torchaudio data


### collect real data
- Data: [HuggingFace: genshin-voice-v3.5-mandarin](https://huggingface.co/datasets/hanamizuki-ai/genshin-voice-v3.5-mandarin/tree/main)
- Label: [Github: GenshinVoice](https://github.com/w4123/GenshinVoice.git)

