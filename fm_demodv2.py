from __future__ import division
from __future__ import print_function
from rtlsdr import *
from scipy import signal
from Lab6.ssd import downsample
from Lab6.lab6 import discrim
from Queue import Queue
import threading
from numpy import cos

import time
from pysoundcard import Stream
from math import *
import pylab as mpl


# TODO implement deemphasis filter, and implement stereo!

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




def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


class SoundPlayer(threading.Thread):
    soundCache = Queue()
    running = True
    max = 0
    block_length = 2048
    s = Stream(sample_rate=rs/N1/N2, block_length=block_length)
    s.start()


    def run(self):
        s = self.s
        while self.running:
            if not self.soundCache.empty():
                samples = self.soundCache.get()
                try:
                    for n in chunks(samples, self.block_length):
                        s.write(n)
                except:
                    print("error")
            else:
                time.sleep(0.001)
        s.stop()

#@limit_calls(1)
def plotpsd(zdis):
    mpl.clf()
    mpl.psd(zdis, NFFT=1024, Fc=0, Fs=48e3)
    mpl.pause(0.0001)
    mpl.show(block=False)

class Demodulator(threading.Thread):

    rawData = Queue()
    running = True
    butterB1 = signal.butter(4, 2 * 100e3/rs, 'low')
    butterB2 = signal.butter(4, 2 * 16e3/(rs/N1), 'low')
    f3 = 1/(2*pi*75e-6)
    a1 = exp(-2*pi*f3/(rs/N1/N2))

    pll = cos()

    def __init__(self, soundplayer):
        super(Demodulator, self).__init__()
        self.soundplayer = soundplayer

    def decode(self, samples):
        x = samples
        #yb1 = signal.convolve(x, self.filterB1)
        yb1 = signal.lfilter(self.butterB1[0],self.butterB1[1],x)
        yn1 = downsample(yb1, N1)
        zdis = discrim(yn1)
        #zb2 = signal.convolve(zdis, self.filterB2)
        zb2 = signal.lfilter(self.butterB2[0],self.butterB2[1],zdis)
        zn2 = downsample(zb2, N2)
        a1 = self.a1
        deemp = signal.lfilter([1 -a1],[1,-a1],zn2)
        plotpsd(zn2)

        ##### second channel












        self.soundplayer.soundCache.put(deemp)







    def run(self):
        while self.running:
            if not self.rawData.empty():
                self.decode(self.rawData.get())
            else:
                time.sleep(0.001)



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