from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from pprint import pprint
from copy import deepcopy
from cv2 import filter2D


def show_magnitude_spectrum(img, label) :
    magspec = 10*np.log(np.abs(img))

    # clip and clamp values 
    for i in range(magspec.shape[0]) :
        for j in range(magspec.shape[1]) :
            if(magspec[i,j] < 0) :
                magspec[i,j] = 0
            if(magspec[i,j] > 255) :
                magspec[i,j] = 255
    plt.imshow(magspec,cmap='gray')
    plt.title(label)
    plt.xlabel("Frequency")
    plt.ylabel("Energy")
    plt.show()


if __name__ == "__main__":

    """  Q1  Solution  """
    img = Image.open("UdayaanNath_2017119_Noise-lines.jpg")
    npimg = np.asarray(img)

    # showing magnitude spectrum of noisy image
    fftimg = np.fft.fft2(npimg)
    fftimg  = np.fft.fftshift(fftimg)
    show_magnitude_spectrum(fftimg,"Q1: Magnitude spectrum of Noisy Image")

    # this is the img (after zero padding) in frequency domain
    fftimg = np.fft.fft2(npimg,s=(2*npimg.shape[0],2*npimg.shape[1]))
    fftimg  = np.fft.fftshift(fftimg)

    # estimating the filter
    H = np.zeros((2*npimg.shape[0],2*npimg.shape[1]),dtype=complex)
    for i in range(H.shape[0]) :
        if(i>=511 and i<=512) :
            continue
        H[i][514] = 1
        H[i][513] = 1
        H[i][512] = 1
        H[i][511] = 1
        H[i][510] = 1
        H[i][509] = 1
        
    # show magnitude spectrum of the obtained noise
    noise = np.multiply(H,fftimg)
    show_magnitude_spectrum(noise,"Q1: magnitude spectrum of the obtained noise")

    # H is the filter in frequency domain
    H = 1-H 

    # show magnitude spectrum of the filter
    L = deepcopy(H)
    L.real = L.real*255
    show_magnitude_spectrum(L,"Q1: magnitude spectrum of the filter")
    
    # do the Hadamard product of the filter and fftimg
    prod = np.multiply(H,fftimg)

    # calucate inverse dft and crop the output
    ifftimg = np.fft.ifft2(prod)
    g = abs(ifftimg.real)
    g = g[0:npimg.shape[0],0:npimg.shape[1]]

    # show the filtered image after dft
    plt.imshow(g,cmap="gray")
    plt.title("Q1: De-noised Image via DFT")
    plt.show()

    # tranforming the filter in frequency domain (H) to filter in spatial domain (f)
    L = deepcopy(H)
    L = np.fft.fftshift(L)
    f = np.fft.ifft2(L)
    f = (f.real)
    
    #convolution of noisy image with filter in spatial domain (f)
    denoised_img = filter2D(npimg,-1,f)

    # show de-noised image after convolution
    denoised_img = np.rot90(denoised_img)
    denoised_img = np.rot90(denoised_img)
    plt.imshow(denoised_img,cmap="gray")
    plt.title("Q1: De-noised Image via conv")
    plt.show()

    # show magnitude spectrum of the filtered image
    fftg = np.fft.fft2(g)
    fftg = np.fft.fftshift(fftg)
    show_magnitude_spectrum(fftg,"Q1: magnitude spectrum of the filtered image")


    """  Q2  Solution  """
    img = Image.open("UdayaanNath_2017119_Barbara.bmp")
    img = np.asarray(img)
    h = [[0,1,0],[1,-4,1],[0,1,0]]
    h = np.asarray(h)

    # computing the filter in the fourier domain
    H = np.fft.fft2(h,s=(h.shape[0]+img.shape[0], h.shape[1]+img.shape[1]))
    H = np.fft.fftshift(H)

    # show magnitude spectrum of filter in fourier domain
    show_magnitude_spectrum(H,"Q2: magnitude spectrum of filter in fourier domain")

    # computing the image in the fourier domain
    I = np.fft.fft2(img,s=(h.shape[0]+img.shape[0], h.shape[1]+img.shape[1]))
    I = np.fft.fftshift(I)

    # do the Hadamard product
    prod = np.multiply(H,I)

    # calucate inverse dft and crop the output
    ifftimg = np.fft.ifft2(prod)
    g = abs(ifftimg.real)
    g = g[0:npimg.shape[0],0:npimg.shape[1]]

    # show the filtered image after dft
    plt.imshow(g,cmap="gray")
    plt.title("Q2: Output of Laplacian filtering in the fourier domain")
    plt.show()

    """  Q3  Solution  """
    img = Image.open("UdayaanNath_2017119_Barbara.bmp")
    img = np.asarray(img)
    D0 = 20

    # creating Gaussian LPF in fourier domain
    H = np.zeros((2*img.shape[0],2*img.shape[1]),dtype=complex)
    for k in range(H.shape[0]) :
        for l in range(H.shape[1]):
            H[k,l] = (-1)*( (k-H.shape[0]/2)*(k-H.shape[0]/2) + (l-H.shape[1]/2)*(l-H.shape[1]/2) )/(2*D0*D0)
    H = np.exp(H)

    # show magnitude spectrum of the filter
    L = deepcopy(H)
    x = np.max(L)
    L.real = L.real*(255/x.real)
    show_magnitude_spectrum(L,"Q3: magnitude spectrum of Gaussian LPF")

    # computing the image in the fourier domain
    I = np.fft.fft2(img,s=H.shape)
    I = np.fft.fftshift(I)

    # do the Hadamard product
    prod = np.multiply(H,I)

    # calucate inverse dft and crop the output
    ifftimg = np.fft.ifft2(prod)
    g = abs(ifftimg.real)
    g = g[0:img.shape[0],0:img.shape[1]]

    # show the filtered image after dft
    plt.imshow(g,cmap="gray")
    plt.title("Q3: Output of Fourier Domain Gaussian LPF filtering")
    plt.show()

    # show magnitude spectrum of filtered image
    fftg = np.fft.fft2(g)
    fftg = np.fft.fftshift(fftg)
    show_magnitude_spectrum(fftg,"Q3: Magnitude spectrum of filtered image")