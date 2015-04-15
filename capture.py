
from __future__ import division
from __future__ import print_function
from rtlsdr import *
import time
import pylab as mpl
from scipy import signal
from numpy import *



from Lab6.ssd import downsample
from Lab6.lab6 import discrim, complex2wav, capture

sdr = RtlSdr()
sdr.DEFAULT_ASYNC_BUF_NUMBER = 16
sdr.rs = 2.4e6
sdr.fc = 78.1e6
sdr.gain = 40

dataset = []


@limit_calls(50 )
def test_callback(samples, rtlsdr_obj):
    global dataset
    dataset.extend(samples)

sdr.read_samples_async(test_callback, 256*1024)  ### this is blocking!
sdr.close()
print(len(dataset))
complex2wav("capture.wav",2.4e6,dataset)