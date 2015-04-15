"""
Digital Communications Function Module

Mark Wickert November 2013 - March 2014
"""

from matplotlib import pylab
from matplotlib import mlab
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import erfc
from sys import exit

def farrow_resample(x, fs_old, fs_new):
    """
    ==========================================================================
     farrow_resample
    ==========================================================================
    y = farrow_resample(x,fs_old, fs_new)
    
    An cubic interpolator using a Farrow structure is used resample the
    input data at a new sampling rate that may be an irrational multiple of
    the input sampling rate.
    
     x = Input signal vector needing resampling
     y = Output signal vector
    
    The filter coefficients used here and a more comprehensive listing can be 
    found in H. Meyr, M. Moeneclaey, & S. Fechtel, "Digital Communication 
    Receivers," Wiley, 1998, Chapter 9, pp. 521-523.
    
    Another good paper on variable interpolators is: L. Erup, F. Gardner, &
    R. Harris, "Interpolation in Digital Modems--Part II: Implementation
    and Performance," IEEE Comm. Trans., June 1993, pp. 998-1008.
    
    A founding paper on the subject of interpolators is: C. W. Farrow, "A
    Continuously variable Digital Delay Element," Proceedings of the IEEE
    Intern. Symp. on Circuits Syst., pp. 2641-2645, June 1988.
    
    Mark Wickert April 2003, recoded to Python November 2013
    """
    
    #Cubic interpolator over 4 samples.
    #The base point receives a two sample delay.
    v3 = signal.lfilter([1/6., -1/2., 1/2., -1/6.],1,x)
    v2 = signal.lfilter([0, 1/2., -1, 1/2.],1,x)
    v1 = signal.lfilter([-1/6., 1, -1/2., -1/3.],1,x)
    v0 = signal.lfilter([0, 0, 1],1,x)
    
    Ts_old = 1/float(fs_old)
    Ts_new = 1/float(fs_new)
    
    T_end = Ts_old*(len(x)-3)
    t_old = np.arange(0,T_end+Ts_old,Ts_old)
    t_new = np.arange(0,T_end+Ts_old,Ts_new)
    if x.dtype == np.dtype('complex128'):
        y = np.zeros(len(t_new)) + 1j*np.zeros(len(t_new))
    else:
        y = np.zeros(len(t_new))

    for n in range(len(t_new)):
        n_old = np.floor(n*Ts_new/Ts_old)
        mu = (n*Ts_new - n_old*Ts_old)/Ts_old;
        #Combine outputs
        #y(n) = ((v3(n_old+2)*mu + v2(n_old+2))*mu + v1(n_old+2))*mu + v0(n_old+2);
        y[n] = ((v3[n_old+1]*mu + v2[n_old+1])*mu 
                + v1[n_old+1])*mu + v0[n_old+1]
    return y

def upsample(x,L):
    """
    Upsample by factor L
    
    Insert L - 1 zero samples in between each input sample.
    
    Parameters
    ----------
    x : ndarray of input signal values
    L : upsample factor
    
    Returns
    -------
    y : ndarray of the output signal values
    
    Examples
    --------
    >>> y = upsample(x,3)
    """
    N_input = len(x)
    y = np.hstack((x.reshape(N_input,1),np.zeros((N_input,L-1))))
    y = y.flatten()
    return y
    
def downsample(x,M,p=0):
    """
    Downsample by factor M

    Keep every Mth sample of the input. The phase of the input samples
    kept can be selected.

    Parameters
    ----------
    x : ndarray of input signal values
    M : upsample factor
    p : phase of decimated value, 0 (default), 1, ..., M-1

    Returns
    -------
    y : ndarray of the output signal values

    Examples
    --------
    >>> y = downsample(x,3)
    >>> y = downsample(x,3,1)
    """
    x = x[0:np.floor(len(x)/M)*M]
    x = x.reshape((len(x)/M,M))
    y = x[:,p]
    return y
    
def eye_plot(x,L,S=0):
    """
    Eye pattern plot of a baseband digital communications waveform.

    The signal must be real, but can be multivalued in terms of the underlying
    modulation scheme. Used for BPSK eye plots in the Case Study article.

    Parameters
    ----------
    x : ndarray of the real input data vector/array
    L : display length in samples (usually two symbols)
    S : start index

    Returns
    -------
    Nothing : A plot window opens containing the eye plot
    
    Notes
    -----
    Increase S to eliminate filter transients.
    
    Examples
    --------
    >>> # 1000 bits at 10 samples per bit with 'rc' shaping
    >>> x,b, data = NRZ_bits(1000,10,'rc')
    >>> eye_plot(x,20,60)
    """
    plt.figure(figsize=(6,4))
    idx = np.arange(0,L+1)
    plt.plot(idx,x[S:S+L+1],'b')
    k_max = int((len(x) - S)/L)-1
    for k in range(1,k_max):
         plt.plot(idx,x[S+k*L:S+L+1+k*L],'b')
    plt.grid()
    plt.xlabel('Time Index - n')
    plt.ylabel('Amplitude')
    plt.title('Eye Plot')
    return 0

def scatter(x,Ns,start):
    """
    Sample a baseband digital communications waveform at the symbol spacing.

    Parameters
    ----------
    x : ndarray of the input digital comm signal
    Ns : number of samples per symbol (bit)
    start : the array index to start the sampling

    Returns
    -------
    xI : ndarray of the real part of x following sampling
    xQ : ndarray of the imaginary part of x following sampling

    Notes
    -----
    Normally the signal is complex, so the scatter plot contains 
    clusters at point  in the complex plane. For a binary signal 
    such as BPSK, the point centers are nominally +/-1 on the real
    axis. Start is used to eliminate transients from the FIR
    pulse shaping filters from appearing in the scatter plot.

    Examples
    --------
    >>> x,b, data = NRZ_bits(1000,10,'rc')
    >>> # add some noise so points are now scattered about +/-1
    >>>  y = cpx_AWGN(x,20,10)
    >>>  yI,yQ = scatter(y,10,60)
    >>> plot(yI,yQ,'.')
    >>> axis('equal')
    """
    xI = np.real(x[start::Ns])
    xQ = np.imag(x[start::Ns])
    return xI, xQ

def strips(x,Nx,fig_size=(6,4)):
    """
    Plots the contents of real ndarray x as a vertical stacking of
    strips, each of length Nx. The default figure size is (6,4) inches.
    The yaxis tick labels are the starting index of each strip. The red
    dashed lines correspond to zero amplitude in each strip.

    strips(x,Nx,my_figsize=(6,4))

    Mark Wickert April 2014
    """
    plt.figure(figsize=fig_size)
    #ax = fig.add_subplot(111)
    N = len(x)
    Mx = int(np.ceil(N/float(Nx)))
    x_max = np.max(np.abs(x))
    for kk in range(Mx):
        plt.plot(np.array([0,Nx]),-kk*Nx*np.array([1,1]),'r-.')
        plt.plot(x[kk*Nx:(kk+1)*Nx]/x_max*0.4*Nx-kk*Nx,'b')
    plt.axis([0,Nx,-Nx*(Mx-0.5),Nx*0.5])
    plt.yticks(np.arange(0,-Nx*Mx,-Nx),np.arange(0,Nx*Mx,Nx))
    plt.xlabel('Index')
    plt.ylabel('Strip Amplitude and Starting Index')
    return 0

def bit_errors(z,data,start,Ns):
    """
    A simple bit error counting function.
    
    In its present form this function counts bit errors between
    hard decision BPSK bits in +/-1 form and compares them with
    0/1 binary data that was transmitted. Timing between the Tx
    and Rx data is the responsibility of the user. An enhanced 
    version of this function, which features automatic synching
    will be created in the future.

    Parameters
    ----------
    z : ndarray of hard decision BPSK data prior to symbol spaced sampling
    data : ndarray of reference bits in 1/0 format
    start : timing reference for the received
    Ns : the number of samples per symbol

    Returns
    -------
    Pe_hat : the estimated probability of a bit error

    Notes
    -----
    The Tx and Rx data streams are exclusive-or'd and the then the bit errors
    are summed, and finally divided by the number of bits observed to form an
    estimate of the bit error probability. This function needs to be 
    enhanced to be more useful.

    Examples
    --------
    >>> from scipy import signal
    >>> x,b, data = NRZ_bits(1000,10)
    >>> # set Eb/N0 to 8 dB
    >>>  y = cpx_AWGN(x,8,10)
    >>> # matched filter the signal
    >>> z = signal.lfilter(b,1,y)
    >>> # make bit decisions at 10 and Ns multiples thereafter
    >>> Pe_hat = bit_errors(z,data,10,10)
    """
    Pe_hat = np.sum(data[0:len(z[start::Ns])]^np.int64((np.sign(np.real(z[start::Ns]))+1)/2))/float(len(z[start::Ns]))
    return Pe_hat

def cpx_AWGN(x,EsN0,Ns):
    """
    Apply white Gaussian noise to a digital communications signal.

    This function represents a complex baseband white Gaussian noise
    digital communications channel. The input signal array may be real
    or complex.

    Parameters
    ----------
    x : ndarray noise free complex baseband input signal.
    EsNO : set the channel Es/N0 (Eb/N0 for binary) level in dB
    Ns : number of samples per symbol (bit)

    Returns
    -------
    y : ndarray x with additive noise added.

    Notes
    -----
    Set the channel energy per symbol-to-noise power spectral 
    density ratio (Es/N0) in dB.

    Examples
    --------
    >>> x,b, data = NRZ_bits(1000,10)
    >>> # set Eb/N0 = 10 dB
    >>> y = cpx_AWGN(x,10,10)
    """
    w = np.sqrt(Ns*np.var(x)*10**(-EsN0/10.)/2.)
    w *= (np.random.randn(len(x)) + 1j*np.random.randn(len(x)))                         
    return x+w
    
def CIC(M,K):
    """
    b = CIC(M,K)
    A functional form implementation of a cascade of integrator comb (CIC) 
    filters. Commonly used in multirate signal processing digital
    down-converters and digital up-converters. A true CIC filter requires no
    multiplies, only add and subtract operations. The functional form created
    here is a simple FIR requiring real coefficient multiplies via filter()
    ========================================================================
      M = Effective number of taps per section (typically the decimation
          factor).
      K = The number of CIC sections cascaded (larger K gives the filter a
          wider image rejection bandwidth.
      b = FIR filter coefficients for a simple direct form implementation
          using the filter() function.
    ========================================================================
    Mark Wickert July 2013
    """
    
    if K == 1:
        b = np.ones(M)
    else:
        h = np.ones(M)
        b = h
        for i in range(1,K):
            b = signal.convolve(b,h) # cascade by convolving impulse responses
    
    # Make filter have unity gain at DC
    return b/np.sum(b)

def QPSK_rx(fc,N_symb,Rs,EsN0=100,fs=125,lfsr_len=10,pulse='src'):
    """
    This function generates
    """
    Ns = int(np.round(fs/Rs))
    print 'Ns = ', Ns
    print 'Rs = ', fs/float(Ns)
    print 'EsN0 = ', EsN0, 'dB'
    print 'pulse = ', pulse
    x, b, data = QPSK_bb(N_symb,Ns,lfsr_len,pulse)
    # Add AWGN to x
    x = cpx_AWGN(x,EsN0,Ns)
    n = np.arange(len(x))
    xc = x*np.exp(1j*2*np.pi*fc/float(fs)*n)
    return xc, b, data

def QPSK_tx(fc,N_symb,Rs,fs=125,lfsr_len=10,pulse='src'):
    """

    """
    Ns = int(np.round(fs/Rs))
    print 'Ns = ', Ns
    print 'Rs = ', fs/float(Ns)
    print 'pulse = ', pulse
    x, b, data = QPSK_bb(N_symb,Ns,lfsr_len,pulse)
    n = np.arange(len(x))
    xc = x*np.exp(1j*2*np.pi*fc/float(fs)*n)
    return xc, b, data 

def QPSK_bb(N_symb,Ns,lfsr_len=5,pulse='src',alpha=0.25,M=6):
    """
    
    """
    if lfsr_len > 0:  # LFSR data
        data = PN_gen(2*N_symb,lfsr_len)
        dataI = data[0::2]
        dataQ = data[1::2]
        xI, b = NRZ_bits2(dataI,Ns,pulse,alpha,M)
        xQ, b = NRZ_bits2(dataQ,Ns,pulse,alpha,M)
    else:             # Random data
        data = np.zeros(2*N_symb)
        xI, b, data[0::2] = NRZ_bits(N_symb,Ns,pulse,alpha,M)
        xQ, b, data[1::2] = NRZ_bits(N_symb,Ns,pulse,alpha,M)        
    #print 'P_I: ',np.var(xI), 'P_Q: ',np.var(xQ)
    x = (xI + 1j*xQ)/np.sqrt(2.)
    return x, b, data

def QPSK_BEP(tx_data,rx_data,Ncorr = 1024,Ntransient = 0):
    """
    Count bit errors between a transmitted and received QPSK signal.
    Time delay between streams is detected as well as ambiquity resolution
    due to carrier phase lock offsets of k*pi/4, k=0,1,2,3.
    The ndarray sdata is Tx +/-1 symbols as complex numbers I + j*Q.
    The ndarray data is Rx +/-1 symbols as complex numbers I + j*Q.
    Note: Ncorr needs to be even
    """
    
    #Remove Ntransient symbols
    tx_data = tx_data[Ntransient:]
    rx_data = rx_data[Ntransient:]
    #Correlate the first Ncorr symbols at four possible phase rotations
    R0 = np.fft.ifft(np.fft.fft(rx_data,Ncorr)*
                     np.conj(np.fft.fft(tx_data,Ncorr)))
    R1 = np.fft.ifft(np.fft.fft(1j*rx_data,Ncorr)*
                     np.conj(np.fft.fft(tx_data,Ncorr)))
    R2 = np.fft.ifft(np.fft.fft(-1*rx_data,Ncorr)*
                     np.conj(np.fft.fft(tx_data,Ncorr)))
    R3 = np.fft.ifft(np.fft.fft(-1j*rx_data,Ncorr)*
                     np.conj(np.fft.fft(tx_data,Ncorr)))
    #Place the zero lag value in the center of the array
    R0 = np.fft.fftshift(R0)
    R1 = np.fft.fftshift(R1)
    R2 = np.fft.fftshift(R2)
    R3 = np.fft.fftshift(R3)
    R0max = np.max(R0.real)
    R1max = np.max(R1.real)
    R2max = np.max(R2.real)
    R3max = np.max(R3.real)
    R = np.array([R0max,R1max,R2max,R3max])
    Rmax = np.max(R)
    kphase_max = np.where(R == Rmax)[0]
    kmax = kphase_max[0]
    #Correlation lag value is zero at the center of the array
    if kmax == 0:
        lagmax = np.where(R0.real == Rmax)[0] - Ncorr/2
    elif kmax == 1:
        lagmax = np.where(R1.real == Rmax)[0] - Ncorr/2
    elif kmax == 2: 
        lagmax = np.where(R2.real == Rmax)[0] - Ncorr/2
    elif kmax == 3:
        lagmax = np.where(R3.real == Rmax)[0] - Ncorr/2
    taumax = lagmax[0]
    print('kmax =  %d, taumax = %d' % (kmax, taumax))
    #return R0,R1,R2,R3
    #Count bit and symbol errors over the entire input ndarrays
    #Begin by making tx and rx length equal and apply phase rotation to rx
    if taumax < 0:
        tx_data = tx_data[-taumax:]
        tx_data = tx_data[:min(len(tx_data),len(rx_data))]
        rx_data = 1j**kmax*rx_data[:len(tx_data)]
    else:
        rx_data = 1j**kmax*rx_data[taumax:]
        rx_data = rx_data[:min(len(tx_data),len(rx_data))]
        tx_data = tx_data[:len(rx_data)]
    #Convert to 0's and 1's
    S_count = len(tx_data)
    tx_I = np.int16((tx_data.real + 1)/2)
    tx_Q = np.int16((tx_data.imag + 1)/2)
    rx_I = np.int16((rx_data.real + 1)/2)
    rx_Q = np.int16((rx_data.imag + 1)/2)
    I_errors = tx_I ^ rx_I
    Q_errors = tx_Q ^ rx_Q
    #A symbol errors occurs when I or Q or both are in error
    S_errors = I_errors | Q_errors
    #return 0
    return S_count,np.sum(I_errors),np.sum(Q_errors),np.sum(S_errors)

def BPSK_BEP(tx_data,rx_data,Ncorr = 1024,Ntransient = 0):
    """
    Count bit errors between a transmitted and received BPSK signal.
    Time delay between streams is detected as well as ambiquity resolution
    due to carrier phase lock offsets of k*pi, k=0,1.
    The ndarray tx_data is Tx +/-1 symbols as real numbers I.
    The ndarray rx_data is Rx +/-1 symbols as real numbers I.
    Note: Ncorr needs to be even
    """
    
    #Remove Ntransient symbols
    tx_data = tx_data[Ntransient:]
    rx_data = rx_data[Ntransient:]
    #Correlate the first Ncorr symbols at four possible phase rotations
    R0 = np.fft.ifft(np.fft.fft(rx_data,Ncorr)*
                     np.conj(np.fft.fft(tx_data,Ncorr)))
    R1 = np.fft.ifft(np.fft.fft(-1*rx_data,Ncorr)*
                     np.conj(np.fft.fft(tx_data,Ncorr)))
    #Place the zero lag value in the center of the array
    R0 = np.fft.fftshift(R0)
    R1 = np.fft.fftshift(R1)
    R0max = np.max(R0.real)
    R1max = np.max(R1.real)
    R = np.array([R0max,R1max])
    Rmax = np.max(R)
    kphase_max = np.where(R == Rmax)[0]
    kmax = kphase_max[0]
    #Correlation lag value is zero at the center of the array
    if kmax == 0:
        lagmax = np.where(R0.real == Rmax)[0] - Ncorr/2
    elif kmax == 1:
        lagmax = np.where(R1.real == Rmax)[0] - Ncorr/2
    taumax = lagmax[0]
    print('kmax =  %d, taumax = %d' % (kmax, taumax))
    #return R0,R1,R2,R3
    #Count bit and symbol errors over the entire input ndarrays
    #Begin by making tx and rx length equal and apply phase rotation to rx
    if taumax < 0:
        tx_data = tx_data[-taumax:]
        tx_data = tx_data[:min(len(tx_data),len(rx_data))]
        rx_data = (-1)**kmax*rx_data[:len(tx_data)]
    else:
        rx_data = (-1)**kmax*rx_data[taumax:]
        rx_data = rx_data[:min(len(tx_data),len(rx_data))]
        tx_data = tx_data[:len(rx_data)]
    #Convert to 0's and 1's
    S_count = len(tx_data)
    tx_I = np.int16((tx_data.real + 1)/2)
    rx_I = np.int16((rx_data.real + 1)/2)
    I_errors = tx_I ^ rx_I
    #Symbol errors and bit errors are equivalent
    S_errors = I_errors
    #return tx_data, rx_data
    return S_count,np.sum(S_errors)
        
def BPSK_tx(N_bits,Ns,ach_fc=2.0,ach_lvl_dB=-100,pulse='rect',alpha = 0.25,M=6):
    """
    Generates biphase shift keyed (BPSK) transmitter with adjacent channel interference.

    Generates three BPSK signals with rectangular or square root raised cosine (SRC) 
    pulse shaping of duration N_bits and Ns samples per bit. The desired signal is
    centered on f = 0, which the adjacent channel signals to the left and right
    are also generated at dB level relative to the desired signal. Used in the 
    digital communications Case Study supplement.

    Parameters
    ----------
    N_bits : the number of bits to simulate
    Ns : the number of samples per bit
    ach_fc : the frequency offset of the adjacent channel signals (default 2.0)
    ach_lvl_dB : the level of the adjacent channel signals in dB (default -100)
    pulse :the pulse shape 'rect' or 'src'
    alpha : square root raised cosine pulse shape factor (default = 0.25)
    M : square root raised cosine pulse truncation factor (default = 6)

    Returns
    -------
    x : ndarray of the composite signal x0 + ach_lvl*(x1p + x1m)
    b : the transmit pulse shape 
    data0 : the data bits used to form the desired signal; used for error checking

    Notes
    -----

    Examples
    --------
    >>> x,b,data0 = BPSK_tx(1000,10,'src')
    """
    x0,b,data0 = NRZ_bits(N_bits,Ns,pulse,alpha,M)
    x1p,b,data1p = NRZ_bits(N_bits,Ns,pulse,alpha,M)
    x1m,b,data1m = NRZ_bits(N_bits,Ns,pulse,alpha,M)
    n = np.arange(len(x0))
    x1p = x1p*np.exp(1j*2*np.pi*ach_fc/float(Ns)*n)
    x1m = x1m*np.exp(-1j*2*np.pi*ach_fc/float(Ns)*n)
    ach_lvl = 10**(ach_lvl_dB/20.)
    return x0 + ach_lvl*(x1p + x1m), b, data0
    
def rc_imp(Ns,alpha,M=6):
    """
    A truncated raised cosine pulse used in digital communications.

    The pulse shaping factor 0< alpha < 1 is required as well as the 
    truncation factor M which sets the pulse duration to be 2*M*Tsymbol.

    Parameters
    ----------
    Ns : number of samples per symbol
    alpha : excess bandwidth factor on (0, 1), e.g., 0.35
    M : equals RC one-sided symbol truncation factor

    Returns
    -------
    b : ndarray containing the pulse shape

    Notes
    -----
    The pulse shape b is typically used as the FIR filter coefficients
    when forming a pulse shaped digital communications waveform.

    Examples
    --------
    >>> # ten samples per symbol and alpha = 0.35
    >>> b = rc_imp(10,0.35)
    >>> n = arange(-10*6,10*6+1)
    >>> stem(n,b)
    """
    # Design the filter
    n = np.arange(-M*Ns,M*Ns+1)
    b = np.zeros(len(n));
    a = alpha;
    Ns *= 1.0
    for i in range(len(n)):
        if (1 - 4*(a*n[i]/Ns)**2) == 0:
            b[i] = np.pi/4*np.sinc(1/(2.*a))
        else:
            b[i] = np.sinc(n[i]/Ns)*np.cos(np.pi*a*n[i]/Ns)/(1 - 4*(a*n[i]/Ns)**2)
    return b

def sqrt_rc_imp(Ns,alpha,M=6):
    """
    A truncated square root raised cosine pulse used in digital communications.

    The pulse shaping factor 0< alpha < 1 is required as well as the 
    truncation factor M which sets the pulse duration to be 2*M*Tsymbol.
     

    Parameters
    ----------
    Ns : number of samples per symbol
    alpha : excess bandwidth factor on (0, 1), e.g., 0.35
    M : equals RC one-sided symbol truncation factor

    Returns
    -------
    b : ndarray containing the pulse shape

    Notes
    -----
    The pulse shape b is typically used as the FIR filter coefficients
    when forming a pulse shaped digital communications waveform. When 
    square root raised cosine (SRC) pulse is used generate Tx signals and 
    at the receiver used as a matched filter (receiver FIR filter), the 
    received signal is now raised cosine shaped, this having zero 
    intersymbol interference and the optimum removal of additive white 
    noise if present at the receiver input.

    Examples
    --------
    >>> # ten samples per symbol and alpha = 0.35
    >>> b = sqrt_rc_imp(10,0.35)
    >>> n = arange(-10*6,10*6+1)
    >>> stem(n,b)
    """
    # Design the filter
    n = np.arange(-M*Ns,M*Ns+1)
    b = np.zeros(len(n))
    Ns *= 1.0
    a = alpha
    for i in range(len(n)):
       if abs(1 - 16*a**2*(n[i]/Ns)**2) <= np.finfo(np.float).eps/2:
           b[i] = 1/2.*((1+a)*np.sin((1+a)*np.pi/(4.*a))-(1-a)*np.cos((1-a)*np.pi/(4.*a))+(4*a)/np.pi*np.sin((1-a)*np.pi/(4.*a)))
       else:
           b[i] = 4*a/(np.pi*(1 - 16*a**2*(n[i]/Ns)**2))
           b[i] = b[i]*(np.cos((1+a)*np.pi*n[i]/Ns) + np.sinc((1-a)*n[i]/Ns)*(1-a)*np.pi/(4.*a))
    return b    

def NRZ_bits(N_bits,Ns,pulse='rect',alpha = 0.25,M=6):
    """
    Generate non-return-to-zero (NRZ) data bits with pulse shaping.

    A baseband digital data signal using +/-1 amplitude signal values
    and including pulse shaping.

    Parameters
    ----------
    N_bits : number of NRZ +/-1 data bits to produce
    Ns : the number of samples per bit,
    pulse_type : 'rect' , 'rc', 'src' (default 'rect')
    alpha : excess bandwidth factor(default 0.25)
    M : single sided pulse duration (default = 6) 

    Returns
    -------
    x : ndarray of the NRZ signal values
    b : ndarray of the pulse shape
    data : ndarray of the underlying data bits

    Notes
    -----
    Pulse shapes include 'rect' (rectangular), 'rc' (raised cosine), 
    'src' (root raised cosine). The actual pulse length is 2*M+1 samples.
    This function is used by BPSK_tx in the Case Study article.

    Examples
    --------
    >>> x,b,data = NRZ_bits(100,10)
    >>> t = arange(len(x))
    >>> plot(t,x)
    """
    data = np.random.randint(0,2,N_bits) 
    x = np.hstack((2*data.reshape(N_bits,1)-1,np.zeros((N_bits,Ns-1))))
    x =x.flatten()
    if pulse.lower() == 'rect':
        b = np.ones(Ns)
    elif pulse.lower() == 'rc':
        b = rc_imp(Ns,alpha,M)
    elif pulse.lower() == 'src':
        b = sqrt_rc_imp(Ns,alpha,M)
    else:
        print 'pulse type must be rec, rc, or src'
    x = signal.lfilter(b,1,x)
    return x,b/float(Ns),data

def NRZ_bits2(data,Ns,pulse='rect',alpha = 0.25,M=6):
    """
    Generate non-return-to-zero (NRZ) data bits with pulse shaping with user data

    A baseband digital data signal using +/-1 amplitude signal values
    and including pulse shaping. The data sequence is user supplied.

    Parameters
    ----------
    data : ndarray of the data bits as 0/1 values
    Ns : the number of samples per bit,
    pulse_type : 'rect' , 'rc', 'src' (default 'rect')
    alpha : excess bandwidth factor(default 0.25)
    M : single sided pulse duration (default = 6) 

    Returns
    -------
    x : ndarray of the NRZ signal values
    b : ndarray of the pulse shape

    Notes
    -----
    Pulse shapes include 'rect' (rectangular), 'rc' (raised cosine), 
    'src' (root raised cosine). The actual pulse length is 2*M+1 samples.

    Examples
    --------
    >>> x,b = NRZ_bits2([m_seq(5),10)
    >>> t = arange(len(x))
    >>> plot(t,x)
    """
    N_bits = len(data)
    x = np.hstack((2*data.reshape(N_bits,1)-1,np.zeros((N_bits,Ns-1))))
    x = x.flatten()
    if pulse.lower() == 'rect':
        b = np.ones(Ns)
    elif pulse.lower() == 'rc':
        b = rc_imp(Ns,alpha,M)
    elif pulse.lower() == 'src':
        b = sqrt_rc_imp(Ns,alpha,M)
    else:
        print 'pulse type must be rec, rc, or src'
    x = signal.lfilter(b,1,x)
    return x,b/float(Ns)
    
def PN_gen(N_bits,m=5):
    """
    Maximal length sequence signal generator.

    Generates a sequence 0/1 bits of N_bit duration. The bits themselves
    are obtained from an m-sequence of length m. Available m-sequence
    (PN generators) include m = 2,3,...,12, & 16.
            
    Parameters
    ----------
    N_bits : the number of bits to generate
    m : the number of shift registers. 2,3, .., 12, & 16

    Returns
    -------
    PN : ndarray of the generator output over N_bits

    Notes
    -----
    The sequence is periodic having period 2**m - 1 (2^m - 1).

    Examples
    --------
    >>> # A 15 bit period signal nover 50 bits
    >>> PN = PN_gen(50,4)
    """
    c = m_seq(m)
    Q = len(c)
    max_periods = int(np.ceil(N_bits/float(Q)))
    PN = np.zeros(max_periods*Q)
    for k in range(max_periods):
        PN[k*Q:(k+1)*Q] = c
    PN = np.resize(PN, (1,N_bits))
    return PN.flatten()

def m_seq(m):
    """
    Generate an m-sequence ndarray using an all-ones initialization.

    Available m-sequence (PN generators) include m = 2,3,...,12, & 16.

    Parameters
    ----------
    m : the number of shift registers. 2,3, .., 12, & 16

    Returns
    -------
    c : ndarray of one period of the m-sequence

    Notes
    -----
    The sequence period is 2**m - 1 (2^m - 1).    

    Examples
    --------
    >>> c = m_seq(5)
    """
    # Load shift register with all ones to start
    sr = np.ones(m)
    # M-squence length is:
    Q = 2**m - 1
    c = np.zeros(Q)

    if m == 2:
        taps = np.array([1, 1, 1])
    elif m == 3:
        taps = np.array([1, 0, 1, 1])
    elif m == 4:
        taps = np.array([1, 0, 0, 1, 1])
    elif m == 5:
        taps = np.array([1, 0, 0, 1, 0, 1])
    elif m == 6:
        taps = np.array([1, 0, 0, 0, 0, 1, 1])
    elif m == 7:
        taps = np.array([1, 0, 0, 0, 1, 0, 0, 1])
    elif m == 8:
        taps = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1])
    elif m == 9:
        taps = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    elif m == 10:
        taps = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
    elif m == 11:
        taps = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1])
    elif m == 12:
        taps = np.array([1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1])
    elif m == 16:
        taps = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1])
    else:
        print 'Invalid length specified'
    for n in range(Q):
        tap_xor = 0
        c[n] = sr[-1]
        for k in range(1,m):
            if taps[k] == 1:
                tap_xor = np.bitwise_xor(tap_xor,np.bitwise_xor(int(sr[-1]),int(sr[m-1-k])))
        sr[1:] = sr[:-1]
        sr[0] = tap_xor
    return c
    
def my_psd(x,NFFT=2**10,Fs=1):
    """
    A local version of NumPy's PSD function that returns the plot arrays.

    A mlab.psd wrapper function that returns two ndarrays;
    makes no attempt to auto plot anything.

    Parameters
    ----------
    x : ndarray input signal
    NFFT : a power of two, e.g., 2**10 = 1024
    Fs : the sampling rate in Hz

    Returns
    -------
    Px : ndarray of the power spectrum estimate
    f : ndarray of frequency values
    
    Notes
    -----
    This function makes it easier to overlay spectrum plots because
    you have better control over the axis scaling than when using psd()
    in the autoscale mode.
    
    Examples
    --------
    >>> x,b, data = NRZ_bits(10000,10)
    >>> Px,f = my_psd(x,2**10,10)
    >>> plot(f, 10*log10(Px))

    """
    Px,f = pylab.mlab.psd(x,NFFT,Fs)
    return Px.flatten(), f
    
def time_delay(x,D,N=2):
    """
    A time varying time delay which takes advantage of the Farrow structure
    for cubic interpolation:

    y = time_delay(x,D,N = 2)

    Note that D is an array of the same length as the input signal x. This
    allows you to make the delay a function of time. If you want a constant 
    delay just use D*zeros(len(x)). The minimum delay allowable is one sample
    or D = 1.0. This is due to the causal system nature of the Farrow 
    structure.

    A founding paper on the subject of interpolators is: C. W. Farrow, "A
    Continuously variable Digital Delay Element," Proceedings of the IEEE
    Intern. Symp. on Circuits Syst., pp. 2641-2645, June 1988.

    Mark Wickert, February 2014
    """

    if type(D) == float or type(D) == int:
        #Make sure D stays with in the tapped delay line bounds
        if np.fix(D) < 1:
            print('D has integer part less than one')
            exit(1)
        if np.fix(D) > N-2:
            print('D has integer part greater than N - 2')
            exit(1)
        # Filter 4-tap input with four Farrow FIR filters
        # Since the time delay is a constant, the LTI filter
        # function from scipy.signal is convenient.
        mu = 1 - (D-np.fix(D))
        Nd = np.fix(D)
        x = np.hstack((x[Nd-1:], np.zeros(Nd-1)))
        v3 = signal.lfilter([1/6., -1/2., 1/2., -1/6.],1,x)
        v2 = signal.lfilter([0, 1/2., -1, 1/2.],1,x)
        v1 = signal.lfilter([-1/6., 1, -1/2., -1/3.],1,x)
        v0 = signal.lfilter([0, 0, 1],1,x)
        y = ((v3*mu + v2)*mu + v1)*mu + v0
    else:
        #Make sure D stays with in the tapped delay line bounds
        if np.fix(np.min(D)) < 1:
            print('D has integer part less than one')
            exit(1)
        if np.fix(max(D)) > N-2:
            print('D has integer part greater than N - 2')
            exit(1)
        y = np.zeros(len(x))
        X = np.zeros(N+1)
        # Farrow filter tap weights
        W3 = np.array([[1./6, -1./2, 1./2, -1./6]])
        W2 = np.array([[0, 1./2, -1., 1./2]])
        W1 = np.array([[-1./6, 1., -1./2, -1./3]])
        W0 = np.array([[0, 0, 1., 0]])
        for k in range(len(x)):
            Nd = np.fix(D[k])
            mu = 1 - (D[k]-np.fix(D[k]))
            # Form a row vector of signal samples, present and past values
            X = np.hstack((np.array(x[k]), X[:-1]))
            # Filter 4-tap input with four Farrow FIR filters
            # Here numpy dot(A,B) performs the matrix multiply
            # since the filter has time-varying coefficients
            v3 = np.dot(W3,np.array(X[Nd-1:Nd+3]).T)
            v2 = np.dot(W2,np.array(X[Nd-1:Nd+3]).T)
            v1 = np.dot(W1,np.array(X[Nd-1:Nd+3]).T)
            v0 = np.dot(W0,np.array(X[Nd-1:Nd+3]).T)
            #Combine sub-filter outputs using mu = 1 - d
            y[k] = ((v3[0]*mu + v2[0])*mu + v1[0])*mu + v0[0]
    return y

def xcorr(x1,x2,Nlags):
    """
    r12, k = xcorr(x1,x2,Nlags), r12 and k are ndarray's
    Compute the energy normalized cross correlation between the sequences
    x1 and x2. If x1 = x2 the cross correlation is the autocorrelation.
    The number of lags sets how many lags to return centered about zero
    """
    K = 2*(len(x1)/2)
    X1 = fft.fft(x1[:K])
    X2 = fft.fft(x2[:K])
    E1 = sum(abs(x1[:K])**2)
    E2 = sum(abs(x2[:K])**2)
    r12 = np.fft.ifft(X1*np.conj(X2))/np.sqrt(E1*E2)
    k = np.arange(K) - K/2
    r12 = np.fft.fftshift(r12)
    idx = mlab.find(abs(k) <= Nlags)
    return r12[idx], k[idx]

def Q_fctn(x):
    """
    Gaussian Q-function
    """
    return 1./2*erfc(x/np.sqrt(2.))

if __name__ == '__main__':
    w1 = np.random.randn(100) + 1j*np.random.randn(100)
    #rw1w1 = xcorr(w1,w1,10)
    x1 = farrow_resample(w1, 1.0, 1.1)
    
    print(x1)
    