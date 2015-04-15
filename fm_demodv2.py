from __future__ import division
from __future__ import print_function
from rtlsdr import *
from scipy import signal
from Lab6.ssd import downsample
from Lab6.lab6 import discrim
from Queue import Queue
import threading
from scikits.audiolab import play
from numpy import max as npmax
import time



fc = 95.7e6
rs = 2.4e6
gain = 40
N1 = 10
N2 = 5
sdr = RtlSdr()
sdr.DEFAULT_ASYNC_BUF_NUMBER = 16
sdr.rs = rs
sdr.fc = fc
sdr.gain = gain



class SoundPlayer(threading.Thread):
    soundCache = Queue()
    running = True
    max = 0

    def run(self):
        while self.running:
            if not self.soundCache.empty():
                samples = self.soundCache.get()
                if not self.max:
                    self.max = npmax(abs(samples), axis=0)
                samples /= self.max
                try:
                    play(samples, fs=rs/N1/N2)
                except:
                    print("error")
            else:
                time.sleep(0.001)


class Demodulator(threading.Thread):

    rawData = Queue()
    running = True
    filterB1 = signal.firwin(12,2 * 100e3/rs)
    filterB2 = signal.firwin(12, 2 * 16e3/rs/10)

    def __init__(self, soundplayer):
        super(Demodulator, self).__init__()
        self.soundplayer = soundplayer

    def decode(self, samples):
        x = samples
        yb1 = signal.convolve(x, self.filterB1)
        yn1 = downsample(yb1, N1)
        zdis = discrim(yn1)
        zb2 = signal.convolve(zdis, self.filterB2)
        zn2 = downsample(zb2, N2)

        #self.soundplayer.soundCache.put(zn2)


    def run(self):
        while self.running:
            if not self.rawData.empty():
                print(self.rawData.qsize())
                self.decode(self.rawData.get())
            else:
                time.sleep(0.001)
                #print("demodulator idle")



sound = SoundPlayer()
demodulator = Demodulator(sound)

def test_callback(samples, rtlsdr_obj):
    demodulator.rawData.put(samples)


def main():

    sound.start()
    demodulator.start()
    sdr.read_samples_async(test_callback, 256*1024)  ### this is blocking!
    sdr.close()
    sound.running = False
    demodulator.running = False
if __name__ == '__main__':
    main()