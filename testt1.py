#    This file is part of pyrlsdr.
#    Copyright (C) 2013 by Roger <https://github.com/roger-/pyrtlsdr>
#
#    pyrlsdr is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    pyrlsdr is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with pyrlsdr.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import division
from __future__ import print_function
from rtlsdr import *
import time


@limit_calls(10)
def test_callback(samples, rtlsdr_obj):
    print('  in callback')
    print('  signal mean:', sum(samples)/len(samples))

def main():
    sdr = RtlSdr()

    sdr.DEFAULT_ASYNC_BUF_NUMBER = 16
    print('Configuring SDR...')
    sdr.rs = 2.5e6
    sdr.fc = 100e6
    sdr.gain = 10

    print('  sample rate: %0.6f MHz' % (sdr.rs/1e6))
    print('  center frequency %0.6f MHz' % (sdr.fc/1e6))
    print('  gain: %d dB' % sdr.gain)

    print('Reading samples...')
    samples = sdr.read_samples(256*1024)
    print('  signal mean:', sum(samples)/len(samples))

    print('Testing callback...')
    sdr.read_samples_async(test_callback, 256*1024)


    try:
        import pylab as mpl

        print('Testing spectrum plotting...')
        mpl.figure()
        #mpl.psd(samples, NFFT=1024, Fc=sdr.fc/1e6, Fs=sdr.rs/1e6)
        mpl.plot(samples)
        mpl.show()
    except:
        # matplotlib not installed/working
        pass
    print('Done\n')
    sdr.close()

if __name__ == '__main__':
    main()