
from __future__ import division
from __future__ import print_function
from rtlsdr import *
import time
import pylab as mpl
from scipy import signal
from numpy import *
from numpy import max as npmax
from Lab6.ssd import downsample
from Lab6.lab6 import discrim, complex2wav, capture
from scikits.audiolab import play

sdr = RtlSdr()
sdr.DEFAULT_ASYNC_BUF_NUMBER = 16
sdr.rs = 2.4e6
sdr.fc = 99.1e6
sdr.gain = 40

filterB1 = signal.firwin(24,2 * 100e3/sdr.rs)
filterB2 = signal.firwin(24, 2 * 16e3/(sdr.rs/10))

dataset = []





@limit_calls(100)
def test_callback(samples, rtlsdr_obj):
    #print(samples)
    mpl.clf()
    #dataset.extend(samples)
    x = samples
    yb1 = signal.convolve(x, filterB1)
    #mpl.psd(yb1, NFFT=1024, Fc=0, Fs=48e3)
    yn1 = downsample(yb1, 10)
    #mpl.psd(yn1, NFFT=1024, Fc=0, Fs=sdr.rs/10)
    zdis = discrim(yn1)
    #mpl.psd(zdis, NFFT=1024, Fc=0, Fs=sdr.rs/10)

    zb2 = signal.convolve(zdis, filterB2)
    zn2 = downsample(zb2, 5)
    # mpl.psd(x)
    #mpl.psd(zn2, NFFT=1024, Fc=0, Fs=48e3)
    #mpl.plot(zdis)
    #mpl.pause(0.0001)
    #mpl.show(block=False)
    zn2 /= npmax(abs(zn2),axis=0)
    # complex2wav("teste.wav",48e3,zn2)
    play(zn2, fs=48e3)

#    y = downsample(x,10)




def main():


    mpl.figure()
    sdr.read_samples_async(test_callback, 256*1024)  ### this is blocking!
    sdr.close()

if __name__ == '__main__':
    main()