
from __future__ import division
from __future__ import print_function
from rtlsdr import *
import time
import pylab as mpl


sdr = RtlSdr()


@limit_calls(100)
def test_callback(samples, rtlsdr_obj):
    mpl.clf()
    signal2 = samples
    mpl.psd(signal2, NFFT=1024, Fc=sdr.fc/1e6, Fs=sdr.rs/1e6)
    mpl.pause(0.0001)
    mpl.show(block=False)


def main():

    sdr.DEFAULT_ASYNC_BUF_NUMBER = 16
    sdr.rs = 2.4e6
    sdr.fc = 99.1e6
    sdr.gain = 10
    mpl.figure()
    sdr.read_samples_async(test_callback, 256*1024)  ### this is blocking!

    sdr.close()

if __name__ == '__main__':
    main()