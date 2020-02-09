
from scipy import ndimage
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Harris_Corner_Detector:
    def __init__(self, sigma=3, corner_thresh=0.1, k=0.04):
        """ Harris Corner Dector. Initialize object and then call get_corners to obtain corners, or disp_corners to
            to obtain overlay.
        Dependencies:
            ndimage.convolve
            numpy
        Args:
            sigma: (float) std deviation for convolution with gaussian window
            cor_thresh: (float) threshold for cornerness. Range: [0,1]
            k: (float) Hyperparameter in cornerness equation, default setting is 0.04 (R = Det(M) - K*trace(M))
        """
        self.sigma = sigma
        self.k = k
        self.filter = self.gaus(self.sigma)
        self.corner_thresh = corner_thresh

    def convolve(self, img, filter):
        """ Convolution operation
        Args:
            img: (np.array) input image, Shape [rows, cols]
        Returns:
            convolved image of same size
        """

        return ndimage.convolve(img,  filter, mode='constant')

    def gaus(self, sigma):
        """ Creates a gaussian kernel for convolution
        Args:
            sigma: (float) std deviation of gaussian
        Returns:
            gFilter: (np.array) filer, Shape: [5,5]
        """
        kSize = 5
        gFilter = np.zeros((kSize, kSize))
        gausFunc = lambda u,v, sigma : (1/(2*np.pi*(sigma**2))) * np.exp( - ( (u**2) + (v**2) )/ (2 * (sigma**2)) )

        centerPoint = kSize//2

        for i in range(kSize):
            for j in range(kSize):

                u = i - centerPoint
                v = j - centerPoint

                gFilter[i, j] = gausFunc(u,v,sigma)


        return gFilter

    def grayscale(self, img):
        """ RGB to Grayscale conversion
        Args:
            img: (np array) input image, Shape [3, Rows, Cols]
        Returns:
            img_gray: (np array) input image, Shape [Rows, Cols]
        """

        img_gray = img

        # Omit if already grayscale
        if(len(img.shape) > 2):
            img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img_gray

    def gradxy(self, img):
        """ Gradient computation through convolution
        Args:
            img: (np array) Grayscale input image, Shape [Rows, Cols]
        Returns:
            Ix: (np array) x gradients of image, Shape: [Rows, Cols]
            Iy: (np array) y gradients of image, Shape: [Rows, Cols]
        """
        fx = np.array([[1,-1]])
        fy = np.array([[1],[-1]])

        # compute df/dx
        Ix = self.convolve(img, fx)

        # compute df/dy
        Iy = self.convolve(img, fy)

        return Ix, Iy

    def get_corners(self, img):
        """ Computation of Harris Corners
        Args:
            img: (np array) input image, Shape: [3, Rows, Cols]
        Returns:
            img_corners: (np array), Shape [3, Rows, Cols]
        """

        # Process RGB image
        img = img.astype(np.float32)
        img_gray = self.grayscale(img)
        img_gray = self.convolve(img_gray, self.gaus(3)) # smooth before gradients
        rows, cols = img_gray.shape

        # Compute required Gradients and Gradient polynomials
        Ix, Iy = self.gradxy(img_gray)
        Ix2 = np.square(Ix)
        Iy2 = np.square(Iy)
        Ixy = np.multiply(Ix, Iy)

        # Window function summation
        Sum_Ix2 = self.convolve(Ix2, self.filter)
        Sum_Iy2 = self.convolve(Iy2, self.filter)
        Sum_Ixy = self.convolve(Ixy,  self.filter)

        # Cornerness Scores
        # R = np.multiply(Ix2 , Iy2) - self.k* (np.square(Ix2 + Iy2))

        detM = np.multiply(Sum_Ix2 , Sum_Iy2) - np.multiply(Sum_Ixy, Sum_Ixy)
        traceM = np.square(Sum_Ix2 + Sum_Iy2)
        R = detM - self.k*traceM


        # normalizing, so thresholding is in terms of a ratio
        norm_R = R - R.min()    # [rows, cols]
        norm_R = R / R.max()
        norm_R[norm_R < self.corner_thresh] = 0

        return norm_R

    def disp_corners(self, img, rad=3, col=(0,0,255), thk=1):
        """ Computes corners and plots vizible circles around them, overlaid on the original image.
        Args:
            img: (np array) input image, Shape: [3, Rows, Cols]
        Returns:
            img: (np array) image with overlaid corners, Shape [3, Rows, Cols]
        """

        # Process RGB image
        img = img.astype(np.float32)
        img_gray = self.grayscale(img)

        # Obtain corners
        corners = self.get_corners(img)

        # plot circles
        rows, cols = corners.nonzero()
        coords = zip(cols, rows)
        for c in coords:
            img = cv2.circle(img, c, radius=rad, color=col, thickness=thk, lineType=8, shift=0)

        return img
