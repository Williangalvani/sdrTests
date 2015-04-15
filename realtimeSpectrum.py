
from __future__ import division
from __future__ import print_function
from rtlsdr import *
import time
import pylab as mpl
import scipy.signal as signal
from scipy import *

from numpy import *

def main():
    sdr = RtlSdr()
    print('Configuring SDR...')
    sdr.DEFAULT_ASYNC_BUF_NUMBER = 16
    sdr.rs = 2.5e6   ## sampling rate
    sdr.fc = 100e6   ## center frequency
    sdr.gain = 10

    print('  sample rate: %0.6f MHz' % (sdr.rs/1e6))
    print('  center frequency %0.6f MHz' % (sdr.fc/1e6))
    print('  gain: %d dB' % sdr.gain)

    print('Reading samples...')
    samples = sdr.read_samples(256*1024)
    print('  signal mean:', sum(samples)/len(samples))

    filter = signal.firwin(5, 2* array([99.5,100.5])/sdr.rs,pass_zero=False)



    mpl.figure()
    for i in range(100):
        print('Testing spectrum plotting...')
        mpl.clf()
        signal2 = convolve(sdr.read_samples(256*1024),filter)
        psd = mpl.psd(signal2, NFFT=1024, Fc=sdr.fc/1e6, Fs=sdr.rs/1e6)

        mpl.pause(0.001)
        #mpl.plot(sdr.read_samples(256*1024))
        mpl.show(block=False)

    print('Done\n')
    sdr.close()

if __name__ == '__main__':
    main()