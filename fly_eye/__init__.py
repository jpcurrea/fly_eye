from bird_call import Recording
from rolling_window import rolling_window
import cv2
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.filters import gaussian_filter
from scipy import spatial, ndimage, stats, signal, interpolate
from numpy.linalg import eig, inv, norm
from skimage.feature import peak_local_max
from skimage.draw import ellipse as Ellipse
import skimage
from matplotlib import mlab
from matplotlib import pyplot as plt
from matplotlib import colors
import seaborn as sbn
import subprocess
import numpy as np
import math
import PIL
import os
print("test")


def rgb_2_gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def print_progress(part, whole):
    import sys
    prop = float(part)/float(whole)
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ("="*int(20*prop), 100*prop))
    sys.stdout.flush()


def interpolate_max(arr, heights=(0., 1., 2.)):
    """Given 3 points or arrays, and their corresponding height
    coordinates, calculate the maximum of the Lagrange polynomial
    interpolation of those points. Useful for fast calculations.
    """
    if len(arr) != 3:
        return None
    y1, y2, y3 = arr
    x1, x2, x3 = heights

    x1 = float(x1)
    x2 = float(x2)
    x3 = float(x3)

    num = -(y1*(x2 - x3)*(-x2 - x3)
            + y2*(x1 - x3)*(x1 + x3)
            + y3*(x1 - x2)*(-x1 - x2))
    den = 2. * (y1*(x2 - x3)
                - y2*(x1 - x3)
                + y3*(x2 - x3))

    non_zero_den = np.array(den != 0, dtype=bool)
    zero_den = np.array(den == 0, dtype=bool)
    # print zero_den
    max_heights = np.zeros(num.shape, dtype=float)
    old_err_state = np.seterr(divide='raise')
    ignore_states = np.seterr(**old_err_state)
    max_heights = np.copy(num)
    max_heights[non_zero_den] = max_heights[non_zero_den]/den[non_zero_den]
    max_heights[zero_den] = 0

    # print np.isnan(max_heights).sum()
    # The maximum of the interpolation may lie outside the given height
    # values. If so, ouput the highest value from the data.
    i = np.logical_or(
        max_heights > max(heights), max_heights < min(heights))
    max_heights[i] = np.argmax(arr, axis=0)[i]
    return max_heights


def cartesian_to_spherical(xs, ys, zs, center=[0, 0, 0]):
    pts = np.array([xs, ys, zs])
    center = np.array(center)
    pts -= np.repeat(center, pts.shape[1]).reshape(pts.shape)
    xs, ys, zs = pts
    radial_dists = norm(pts, axis=0)
    inclination = np.arccos(zs / radial_dists)
    azimuth = np.arctan2(ys, xs)
    return inclination, azimuth, radial_dists


def spherical_to_cartesian(inclination, azimuth, radial_dists, center=[0, 0, 0]):
    pts = np.array([inclination, azimuth, radial_dists])
    cx, cy, cz = center
    xs = radial_dists * np.sin(inclination) * np.cos(azimuth)
    ys = radial_dists * np.sin(inclination) * np.sin(azimuth)
    zs = radial_dists * np.cos(inclination)
    xs += cx
    ys += cy
    zs += cz
    return xs, ys, zs


def sphereFit(spX, spY, spZ):
    #   Assemble the f matrix
    f = np.zeros((len(spX), 1))
    f[:, 0] = (spX**2) + (spY**2) + (spZ**2)
    A = np.zeros((len(spX), 4))
    A[:, 0] = spX*2
    A[:, 1] = spY*2
    A[:, 2] = spZ*2
    A[:, 3] = 1
    C, residules, rank, sigval = np.linalg.lstsq(A, f, rcond=None)
    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)
    return radius, np.squeeze(C[:-1])


def rotate(arr, theta, axis=0):
    if axis == 0:
        rot_matrix = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]])
    elif axis == 1:
        rot_matrix = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 2:
        rot_matrix = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]])
    nx, ny, nz = np.dot(arr, rot_matrix).T
    nx = np.squeeze(nx)
    ny = np.squeeze(ny)
    nz = np.squeeze(nz)
    return np.array([nx, ny, nz])


def inside_angle(A, B, C):
    CA = A - C
    CB = B - C
    angles = []
    for x in range(len(A)):
        angles += [np.dot(CA[x], CB[x]) / (norm(CA[x]) * norm(CB[x]))]
    angles = np.array(angles)
    return np.degrees(np.arccos(angles))


def fit_ellipse(x, y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    return a


def ellipse_center(a):
    b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0 = (c*d-b*f)/num
    y0 = (a*f-b*d)/num
    return np.array([x0, y0])


def set_radius_span(a1, a2, R=1, n=100):
    vals = np.linspace(a1, a2, n)
    vals.sort()
    res = np.zeros((n*2+3, 2))
    res[[0, -1], 0] = vals.max()
    res[1:n+1, 0] = vals[::-1]
    res[n+1, 0] = vals.min()
    res[n+2:2*n+2, 0] = vals
    res[0, 1] = R
    res[n+1:, 1] = R
    return res


def local_maxima(img, p=5, window=10, min_diff=None,
                 disp=True):
    max_windows = ndimage.filters.maximum_filter(img, window)
    maxima = (img == max_windows)
    min_windows = ndimage.filters.minimum_filter(img, window)
    diffs = max_windows - min_windows
    if disp:
        sbn.distplot(diffs.flatten())
        plt.axvline(np.percentile(diffs.flatten(), p))
    if min_diff is None:
        cutoff = np.percentile(diffs.flatten(), p)
        delta = diffs > cutoff
    else:
        delta = diffs > min_diff
    # return max_windows - min_windows

    maxima[delta == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(int(x_center))
        y_center = (dy.start + dy.stop - 1)/2
        y.append(int(y_center))

    vals = img[y, x]
    ind = np.argsort(vals)

    x = np.array(x)
    y = np.array(y)

    return x[ind][::-1], y[ind][::-1]


def fundamental_maxima(img, disp=True, p=5, window=5):
    # img = img - img.mean()
    l, w = img.shape[:2]
    # fft = np.fft.fft2(img)
    # fshift = np.fft.fftshift(fft)
    fshift = np.copy(img)
    fshift[:, :int(np.round(w/2))] = 0
    # fshift[l/2, w/2] = 0
    # if disp:
    #     subplot(131)
    x, y = local_maxima(abs(fshift), p=p, disp=False, window=window)

    if disp:
        plt.subplot(131)
        fs = np.copy(abs(fshift))
        fs[l/2, w/2] = 0
        plt.imshow(abs(fs))
        plt.plot(x, y, 'bo')

    points = np.array([x, y])
    num = points.shape[1]
    points = points[:, :min(num, 5)]

    if len(points.T) > 3:
        points = points.T
        tree = spatial.KDTree(points)
        distance, index = tree.query((l/2, w/2), k=4)

        i = distance > 1
        distance = distance[i]
        index = index[i]

        # return distance, index

        inds = points[index].T
        key_freqs = fshift[inds[1], inds[0]]
        new_fshift = np.zeros(fshift.shape, dtype=complex)
        new_fshift[inds[1], inds[0]] = key_freqs  # /abs(key_freqs)
        new_fft = np.fft.ifftshift(new_fshift)
        new_img = np.fft.ifft2(new_fft).real
        if disp:
            plt.plot(inds[0], inds[1], 'r.')
            plt.subplot(132)
            plt.imshow(new_img)
        x, y = local_maxima(-new_img, p=5, window=5, disp=False)
        if disp:
            plt.plot(x, y, 'ro')

    else:
        x = []
        y = []
    return x, y


class Layer():

    """A multi-purpose class used for processing images.
    """

    def __init__(self, filename, bw=False):
        self.filename = filename
        self.bw = bw
        if isinstance(self.filename, str):
            self.image = None
        elif isinstance(self.filename, np.ndarray):
            self.image = self.filename
            self.load_image()
        else:
            self.image = None
        self.sob = None

    def load_image(self):
        """Loads image if not done yet. This is so that we can loop through
        layers without redundantly loading the image.
        """
        if isinstance(self.filename, str):
            self.image = np.asarray(PIL.Image.open(self.filename))
        elif isinstance(self.filename, np.ndarray):
            self.image = np.asarray(self.filename)
        if self.image.ndim < 3:
            self.bw = True
        if self.image.ndim < 2:
            self.image = None
            print("file {} is not an appropriate format.".format(
                self.filename))
        if self.image.ndim == 3:
            if self.image.shape[-1] == 1:
                self.image = np.squeeze(self.image)
            elif self.image.shape[-1] > 3:
                self.image = self.image[..., :-1]
        if (self.image[..., 0] == self.image.mean(-1)).mean() == 1:
            self.image = self.image[..., 0]
            self.bw = True
        return self.image

    def focus(self, smooth=0):
        """Measures the relative focus of each pixel using the Sobel 
        operator.
        """
        if self.image is None:
            self.load_image()
        # image = self.load_image()
        # print self.image
        if not self.bw:
            gray = rgb_2_gray(self.image)
        else:
            gray = self.image
        sx = ndimage.filters.sobel(gray, axis=0, mode='constant')
        sy = ndimage.filters.sobel(gray, axis=1, mode='constant')
        sob = np.hypot(sx, sy)
        self.image = None
        self.sob = sob
        if smooth > 0:
            sob = ndimage.filters.gaussian_filter(sob, sigma=smooth)
        return sob

    def generate_mask(self, thresh=50, b_ground=None):
        """Generates a masking image by thresholding the image. If a
        background is provided, this will also subtract the background
        before generating the mask.
        """
        img = self.load_image()
        thresh = np.zeros(img.shape, "uint8")
        if b_ground is not None:
            img = img - b_ground
        thresh[img > 25] = 255
        mask = ndimage.morphology.binary_dilation(thresh).astype("uint8")
        self.mask = 255*mask

    def select_color(self, range=None, hue_only=False):
        if self.image is None:
            self.load_image()
        self.cs = ColorSelector(self.image, bw=self.bw, hue_only=hue_only)
        self.cs.start_up()
        # while self.cs.displaying():
        #     pass


class Eye(Layer):

    """A class specifically used for processing images of fly eyes. Maybe
    could be modified for other eyes (square lattice, for instance)
    """

    def __init__(self, filename, bw=False, pixel_size=1):
        Layer.__init__(self, filename, bw)
        self.eye_contour = None
        self.ellipse = None
        self.ommatidia = None
        self.pixel_size = pixel_size
        self.mask = None

    def get_eye_outline(self, mask=None, hue_only=False):
        if mask is None and self.mask is None:
            self.select_color(hue_only=hue_only)
            self.mask = self.cs.mask
        elif mask is not None:
            self.mask = mask
        if self.mask is not None:
            conts, h = cv2.findContours(
                self.mask,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_NONE)
            self.conts = conts
            cont = max(conts, key=cv2.contourArea)
            self.cont = cont
            self.eye_contour = cont.reshape((cont.shape[0], cont.shape[-1]))
            mask = np.zeros(self.mask.shape, int)
            mask[self.eye_contour[:, 1], self.eye_contour[:, 0]] = 1
            vert1 = np.cumsum(mask, axis=0)
            vert2 = np.cumsum(mask[::-1], axis=0)[::-1]
            self.eye_mask = (vert1 * vert2) > 0

    def get_eye_sizes(self, disp=False, mask=None, hue_only=False):
        if self.eye_contour is None:
            self.get_eye_outline(mask=mask, hue_only=hue_only)
        self.ellipse = cv2.fitEllipse(self.eye_contour)
        self.eye_length = self.pixel_size*max(self.ellipse[1])
        self.eye_width = self.pixel_size*min(self.ellipse[1])
        self.eye_area = self.pixel_size**2*cv2.contourArea(self.eye_contour)
        if disp:
            plt.imshow(self.image)
            plt.plot(self.eye_contour[:, 0], self.eye_contour[:, 1])
            plt.show()

    def crop_eye(self, padding=1.05, mask=None, hue_only=False):
        if self.image is None:
            self.load_image()
        if self.ellipse is None:
            self.get_eye_sizes(mask=mask, hue_only=hue_only)
        (x, y), (w, h), ang = self.ellipse
        self.angle = ang
        w = padding*w
        h = padding*h
        self.rr, self.cc = Ellipse(x, y, w/2., h/2.,
                                   shape=self.image.shape[:2][::-1],
                                   rotation=np.deg2rad(ang))
        out = np.copy(self.image)
        # out = np.zeros(self.image.shape, dtype='uint8')
        # out[self.cc, self.rr] = self.image[self.cc, self.rr]
        # self.eye = out
        self.eye = Eye(out[min(self.cc):max(self.cc),
                           min(self.rr):max(self.rr)])
        self.eye.mask = self.mask[min(self.cc):max(
            self.cc), min(self.rr):max(self.rr)]
        return self.eye

    def get_ommatidia(self, overlap=5, window_length=5, sigma=3, mask=None,
                      white_peak=True):
        if self.image is None:
            self.load_image()
        if self.bw is False:
            # eye_sats = colors.rgb_to_hsv(self.image.astype('uint8'))[:, :, -1]
            eye_sats = rgb_2_gray(self.image.astype('uint8'))
        else:
            eye_sats = self.image.astype('uint8')
        if self.eye_contour is None:
            self.get_eye_sizes(disp=False, mask=mask)
        # eye_sats[self.eye_mask == False] = eye_sats[self.eye_mask].mean()

        eye_fft = np.fft.fft2(eye_sats)
        eye_fft_shifted = np.fft.fftshift(eye_fft)

        xinds, yinds = np.meshgrid(
            range(eye_fft.shape[1]), range(eye_fft.shape[0]))
        ycenter, xcenter = np.array(eye_fft.shape)/2

        xdiffs, ydiffs = xinds - xcenter, yinds - ycenter
        dists_2d = np.sqrt(xdiffs**2 + ydiffs**2)
        self.dists_2d = dists_2d
        self.angs_2d = np.arctan2(yinds - ycenter, xinds - xcenter)
        i = self.angs_2d < 0
        self.angs_2d[i] = self.angs_2d[i] + np.pi

        # measure 2d power spectrum as a function of radial distance from center
        # using rolling maxima function to find the bounds of the fundamental spatial frequency
        peaks = []
        window_size = 3
        for dist in np.arange(int(dists_2d.max()) - window_size):
            i = np.logical_and(
                dists_2d.flatten() >= dist, dists_2d.flatten() < dist + window_size)
            peaks += [abs(eye_fft_shifted.flatten()[i]).mean()]

        # use peaks to find the local maxima and minima
        peaks = np.array(peaks)
        self.peaks = peaks
        fs = np.linspace(1, len(peaks) + 1, len(peaks))

        # optimum = peak_local_max(fs*peaks, num_peaks=10, min_distance=10)  # second highest maximum
        # optimum = np.squeeze(
        #     peak_local_max(fs*peaks, num_peaks=10, min_distance=10))  # second highest maximum
        optimum = np.squeeze(
            peak_local_max(fs * peaks, num_peaks=1, exclude_border=True))
        optimum = optimum.min()

        # lower_bound = peak_local_max(peaks.max() - peaks[:optimum],
        #                              num_peaks=1)
        # minima = peak_local_max(peaks.max() - peaks[optimum:],
        #                         min_distance=10)
        # upper_bound = 2 * optimum - lower_bound
        upper_bound = 1.5 * optimum
        self.upper_bound = upper_bound

        # std = (upper_bound - lower_bound)/4  #
        # std = .1 * optimum
        # weights = gaussian(dists_2d, optimum, std)
        # in_range = np.logical_and(
        #     dists_2d > optimum - 2*std,
        #     dists_2d < optimum + 2*std)
        in_range = dists_2d < upper_bound
        weights = np.ones(dists_2d.shape)
        weights[in_range == False] = 0
        self.weights = weights

        # using the gaussian weights, invert back to the filtered image
        selection_shifted = np.zeros(eye_fft.shape, dtype=complex)
        selection_shifted = eye_fft_shifted*weights
        selection_fft = np.fft.ifftshift(selection_shifted)
        selection = np.fft.ifft2(selection_fft)

        self.eye_fft_shifted = eye_fft_shifted
        self.selection_fft = selection_fft
        self.filtered_eye = selection.real
        self.centers = np.zeros(selection.shape)
        self.centers[self.mask] = self.filtered_eye[self.mask]

        # filtered_eye = selection.real
        # centers = np.zeros(selection.shape)
        # centers[eye.eye_mask] = filtered_eye[eye.eye_mask]

        # use optimization function for find min_distance that minimizes
        # the variance distances between centers and their nearest neighbors
        d = int(np.round(max(self.image.shape) / 150.))
        if white_peak:
            ys, xs = peak_local_max(self.filtered_eye, min_distance=d).T
        else:
            ys, xs = peak_local_max(
                self.filtered_eye.max() - self.filtered_eye,
                min_distance=d).T

        in_eye = self.mask[ys, xs] == 1
        ys, xs = ys[in_eye], xs[in_eye]

        self.ommatidia = np.array([self.pixel_size*xs, self.pixel_size*ys])

    def get_ommatidial_diameter(self, k_neighbors=7, radius=100,
                                mask=None, window_length=5,
                                white_peak=True):
        if self.ommatidia is None:
            self.get_ommatidia(mask=mask, window_length=window_length,
                               white_peak=white_peak)
        self.tree = spatial.KDTree(self.ommatidia.T)
        dists, inds = self.tree.query(self.ommatidia.T, k=k_neighbors+1)
        dists = dists[:, 1:]
        meds = np.repeat(np.median(dists, axis=1),
                         k_neighbors).reshape(dists.shape)
        too_small = dists < .2*meds
        dists[too_small] = np.nan
        mins = np.repeat(np.nanmin(dists, axis=1),
                         k_neighbors).reshape(dists.shape)
        magn = dists / mins
        too_large = magn > 1.75
        dists[too_large] = np.nan
        self.ommatidial_dists = np.nanmean(dists, axis=1)

        (x, y), (w, h), ang = self.ellipse
        x, y, radius = self.pixel_size * x, self.pixel_size * y, self.pixel_size * radius
        near_center = self.tree.query_ball_point([x, y], r=radius)

        self.ommatidial_diameter = self.ommatidial_dists[near_center].mean()
        self.ommatidial_diameter_SD = self.ommatidial_dists[near_center].std()


class Stack():
    """ A class for combining multiple images into one by taking those
    with the highest focus value determined by the sobel operator.
    """

    def __init__(self, dirname="./", f_type=".jpg", bw=False, eye=True):
        self.dirname = dirname
        fns = os.listdir(self.dirname)
        fns = [os.path.join(dirname, f) for f in fns]
        fns = [f for f in fns if "focus" not in f.lower()]
        fns = sorted(fns)
        fns = [f for f in fns if f.endswith(f_type)]
        fns = [f for f in fns if os.path.split(f)[-1].startswith(".") is False]
        self.layers = []
        self.eyes = eye
        for f in fns:
            if self.eyes:
                layer = Eye(f, bw)
                self.layers.append(layer)
            else:
                layer = Layer(f, bw)
                self.layers.append(layer)

        self.stack = None
        self.bw = bw

    def get_average(self, samples=50):
        """Grab unmoving 'background' of the stack by averaging over
        a sample of layers. The default is 50 samples.
        """
        first = self.layers[0].load_image()
        res = np.zeros(first.shape, dtype=float)
        intervals = len(self.layers)/samples
        for l in self.layers[::int(intervals)]:
            img = l.load_image().astype(float)
            res += img
            l.image = None
        return samples**-1*res

    def focus_stack(self, smooth=0, interpolate=True, use_all=False,
                    layer_smooth=0):
        """The main method which compares the layers of the stack and
        forms the focus stack using the sobel operator, which is generated
        by the layer objects.
        """
        if len(self.layers) == 0:
            print("no images were properly imported")
        else:
            if use_all:
                self.images = []
                self.focuses = []
                for layer in self.layers:
                    self.images += [layer.load_image()]
                    self.focuses += [layer.focus(smooth=layer_smooth)]
                self.focuses = np.array(self.focuses)
                self.images = np.array(self.images)
                if interpolate:
                    print("this is not available yet")
                else:
                    top_focus = np.argmax(self.focuses, axis=0)
                    self.stack = np.zeros(self.images.shape[1:],
                                          dtype='uint8')
                    for val in set(top_focus.flatten()):
                        coords = top_focus == val
                        self.stack[coords] = self.images[val][coords]
            else:
                first = self.layers[0].load_image()
                if first.ndim == 3:
                    l, w, d = first.shape
                    images = np.zeros((3, l, w, d), first.dtype)
                elif first.ndim == 2:
                    l, w = first.shape
                    images = np.zeros((3, l, w), first.dtype)
                focuses = np.zeros((3, l, w), dtype=float)
                heights = focuses[0].astype(int)
                images[0] = first
                previous = self.layers[0].focus()
                focuses[0] = previous
                better = np.greater(focuses[0], focuses[1])
                x = 1
                for l in self.layers[1:]:
                    img = l.load_image()
                    foc = l.focus(smooth=layer_smooth)
                    focuses[2, better] = foc[better]
                    images[2, better] = img[better]
                    better = np.greater(foc, focuses[1])
                    focuses[1, better] = foc[better]
                    images[1, better] = img[better]
                    heights[better] = x
                    focuses[0, better] = previous[better]
                    previous = foc
                    x += 1
                    print_progress(x, len(self.layers))
                self.focuses = focuses
                self.images = images
                h = interpolate_max(focuses)
                self.heights = (heights-1) + h
                if interpolate:
                    down = np.floor(h)
                    up = np.ceil(h)
                    up[up == 0] = 1
                    if self.bw:
                        down = down.flatten().reshape(first.shape)
                        up = up.flatten().reshape(first.shape)
                        h = h.flatten().reshape(first.shape)
                    else:
                        down = down.flatten().repeat(3).reshape(first.shape)
                        up = up.flatten().repeat(3).reshape(first.shape)
                        h = h.flatten().repeat(3).reshape(first.shape)
                    down_img = np.zeros(first.shape)
                    up_img = np.zeros(first.shape)
                    for x in range(3):
                        down_img[np.where(down == x)] = images[x][
                            np.where(down == x)]
                        up_img[np.where(up == x)] = images[x][
                            np.where(up == x)]
                    stack = (up - h)*down_img + (h - down)*up_img
                    stack[np.where(h == 0)] = images[0][np.where(h == 0)]
                    stack[np.where(h == 1)] = images[1][np.where(h == 1)]
                    stack[np.where(h == 2)] = images[2][np.where(h == 2)]
                    self.stack = stack
                    if smooth > 0:
                        self.smooth(smooth)
                else:
                    self.stack = self.images[1]
            print("done")

    def smooth(self, sigma):
        """A 2d smoothing filter for the heights array"""
        self.heights = self.heights.astype("float32")
        self.heights = np.fft.ifft2(
            ndimage.fourier_gaussian(
                np.fft.fft2(self.heights),
                sigma=sigma)).real


class EyeStack(Stack):
    """A special stack for handling a focus stack of fly eye images.
    """

    def __init__(self, dirname, f_type=".jpg", bw=False,
                 pixel_size=1, depth_size=1):
        Stack.__init__(self, dirname, f_type, bw)
        self.eye = None
        self.pixel_size = pixel_size
        self.depth_size = depth_size

    def get_eye_stack(self, smooth=0, interpolate=True, use_all=False,
                      layer_smooth=0, padding=1.1):
        """Generate focus stack of images and then crop out the eye.
        """
        if self.eye is None:
            self.focus_stack(smooth, interpolate, use_all, layer_smooth)
            self.eye = Eye(self.stack.astype('uint8'))
        self.eye.crop_eye(padding)
        self.heights = self.heights[min(self.eye.cc):max(self.eye.cc),
                                    min(self.eye.rr):max(self.eye.rr)]
        self.stack = self.stack[min(self.eye.cc):max(self.eye.cc),
                                min(self.eye.rr):max(self.eye.rr)]
        self.mask = self.eye.cs.mask[min(self.eye.cc):max(self.eye.cc),
                                     min(self.eye.rr):max(self.eye.rr)]
        self.eye.eye_contour[:, 0] -= min(self.eye.rr)
        self.eye.eye_contour[:, 1] -= min(self.eye.cc)
        # self.eye = Eye(self.eye.eye)

    def get_3d_data(self, averaging_range=5, white_peak=True):
        if self.stack is None:
            self.get_eye_stack()
        height, width = self.heights.shape
        xvals = np.linspace(0, (width - 1), width)
        yvals = np.linspace(0, (height - 1), height)
        xs, ys = np.meshgrid(xvals, yvals)

        self.eye_3d = np.array(
            [self.pixel_size * xs.flatten(),
             self.pixel_size * ys.flatten(),
             self.depth_size * self.heights.flatten()])

        self.eye_3d_masked = np.array(
            [self.pixel_size * xs[self.mask == 1].flatten(),
             self.pixel_size * ys[self.mask == 1].flatten(),
             self.depth_size * self.heights[self.mask == 1].flatten()])

        # store the rgb color channels associated with the 3d data
        r, g, b = self.stack.transpose((2, 0, 1))
        self.eye_3d_colors = np.array(
            [r.flatten(), g.flatten(), b.flatten()])

        xs, ys, zs = self.eye_3d_masked
        self.radius, self.center = sphereFit(xs, ys, zs)
        self.eye_3d -= np.repeat(
            self.center, self.eye_3d.shape[1]).reshape(self.eye_3d.shape)

        # rotate points until center of mass is only on the x axis
        # this centers the eye in polar coordinates, minimizing distortion due to the projection
        # also, rotate the eye contour by the same rotation
        # 1. find center of mass
        com = self.eye_3d.mean(1)
        # xs, ys = self.eye.eye_contour.T
        # zs = self.heights[xs, ys].
        # outline = np.array([self.pixel_size * xs.flatten(),
        #                     self.pixel_size * ys.flatten(),
        #                     self.depth_size * zs.flatten()])
        # 2. rotate com along x (com[0]) axis until z (com[2]) = 0
        ang1 = np.arctan2(com[2], com[1])
        com1 = rotate(com, ang1, axis=0)
        rot1 = rotate(self.eye_3d.T, ang1, axis=0)
        # outline1 = rotate(outline.T, ang1, axis=0)
        # 3. rotate com along z (com[2]) axis until y (com[1])= 0
        ang2 = np.arctan2(com1[1], com1[0])
        com2 = rotate(com1, ang2, axis=2)
        rot2 = rotate(rot1.T, ang2, axis=2)
        # outline2 = rotate(outline1.T, ang2, axis=0)
        # 4. convert to spherical coordinates now that they are centered
        xs, ys, zs = rot2
        self.inclination, self.azimuth, self.radii = cartesian_to_spherical(
            xs, ys, zs)
        self.polar = np.array([self.inclination, self.azimuth, self.radii])
        # outline_inclination, outline_azimuth, outline_radii = cartesian_to_spherical(
        #     outline2[0], outline[1], outline[2])
        # self.polar_eye_outline = np.array([outline_inclination, outline_azimuth, outline_radii])

        # using polar coordinates, 'flatten' the image of the eye
        # 1. create a grid for sampling the polar data
        # use a resultion that is high enough for the smallest range
        # resolution of the original image
        incl_range = self.inclination.max() - self.inclination.min()
        azim_range = self.azimuth.max() - self.azimuth.min()
        resolution = min(incl_range, azim_range) / \
            np.sqrt(width**2 + height**2)
        self.polar_grid_resolution = resolution
        incl_new = np.linspace(
            self.inclination.min(),
            self.inclination.max(),
            incl_range / resolution)
        azim_new = np.linspace(
            self.azimuth.min(),
            self.azimuth.max(),
            azim_range / resolution)
        incl_new, azim_new = np.meshgrid(incl_new, azim_new)
        # 2. sample data and interpolate into a grid image
        pred = np.array([self.inclination, self.azimuth])

        # get grid form of data using seperate color channels
        # use linear interpolation for most, and nearest interpolation for far away points
        r_grid_nearest = interpolate.griddata(
            pred.T, self.eye_3d_colors[0], (incl_new, azim_new), method='nearest')
        # r_grid = interpolate.griddata(
        #     pred.T, self.eye_3d_colors[0], (incl_new, azim_new), method='linear')
        # nans = np.isnan(r_grid)
        # r_grid[nans] = r_grid_nearest[nans]

        g_grid_nearest = interpolate.griddata(
            pred.T, self.eye_3d_colors[1], (incl_new, azim_new), method='nearest')
        # g_grid = interpolate.griddata(
        #     pred.T, self.eye_3d_colors[1], (incl_new, azim_new), method='linear')
        # nans = np.isnan(g_grid)
        # g_grid[nans] = g_grid_nearest[nans]

        b_grid_nearest = interpolate.griddata(
            pred.T, self.eye_3d_colors[2], (incl_new, azim_new), method='nearest')
        # b_grid = interpolate.griddata(
        #     pred.T, self.eye_3d_colors[2], (incl_new, azim_new), method='linear')
        # nans = np.isnan(b_grid)
        # b_grid[nans] = b_grid_nearest[nans]

        self.flat_eye = np.array(
            [r_grid_nearest, g_grid_nearest, b_grid_nearest]).transpose((1, 2, 0))
        self.flat_eye = Eye(
            self.flat_eye, pixel_size=self.polar_grid_resolution)

        # in polar coordinates, distances correspond to angles in cartesian space
        self.flat_eye.get_ommatidial_diameter(white_peak=white_peak)
        # interommatidial_ange in degrees
        self.interommatidial_angle = self.flat_eye.ommatidial_diameter * 180. / np.pi
        # ommatidial diameter in mm
        self.ommatidial_diameter = 2 * self.radius * \
            np.sin(self.flat_eye.ommatidial_diameter)

        # vertical field of view using the major axis of the flat eye, in degrees
        self.fov_vertical = self.flat_eye.eye_length * 180. / np.pi
        # horizontal field of view using the minor axis of the flat eye, in degrees
        self.fov_horizontal = self.flat_eye.eye_width * 180. / np.pi
        # field of view approximating the area as an ellipse, in steradians
        self.fov = np.pi * (self.flat_eye.eye_width / 2) * \
            (self.flat_eye.eye_length / 2)


class Video(Stack):
    """Takes a stack of images, or a video that is converted to a stack of images,
    and uses common functions to track motion or color."""

    def __init__(self, filename, fps=30, f_type=".jpg"):
        self.vid_formats = [
            '.mov',
            '.mp4',
            '.mpg',
            '.avi']
        self.filename = filename
        self.f_type = f_type
        self.track = None
        self.colors = None
        if ((os.path.isfile(self.filename) and
             self.filename.lower()[-4:] in self.vid_formats)):
            # if file is a video, use ffmpeg to generate a jpeg stack
            self.dirname = self.filename[:-4]
            self.fps = subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate", "-of",
                "default=noprint_wrappers=1:nokey=1",
                self.filename])
            self.fps = int(str(fps))
            # self.fps = int(self.fps.split("/"))
            # self.fps = float(self.fps[0])/float(self.fps[1])
            if os.path.isdir(self.dirname) is False:
                os.mkdir(self.dirname)
            try:
                if len(os.listdir(self.dirname)) == 0:
                    failed = subprocess.check_output(
                        ["ffmpeg", "-i", self.filename,
                         "-vf", "scale=720:-1",
                         "./{}/frame%05d{}".format(self.dirname, self.f_type)])
            except subprocess.CalledProcessError:
                print("failed to parse video into {}\
                stack!".format(self.f_type))
            try:
                audio_fname = "{}.wav".format(self.filename[:-4])
                failed = subprocess.check_output(
                    ["ffmpeg", "-i", self.filename, "-f", "wav",
                     "-ar", "44100",
                     "-ab", "128",
                     "-vn", audio_fname])
                self.audio = Recording(audio_fname, trim=False)
            except subprocess.CalledProcessError:
                print("failed to get audio from video!")

        elif os.path.isdir(self.filename):
            self.dirname = self.filename
            self.fps = fps
        Stack.__init__(self, self.dirname, f_type=self.f_type)

    def select_color_range(self, samples=5):
        color_range = []
        intervals = len(self.layers)/samples
        for l in self.layers[::intervals]:
            l.select_color()
            color_range += [l.cs.colors]
            l.image = None
        color_range = np.array(color_range)
        self.colors = np.array([color_range.min((0, 1)),
                                color_range.max((0, 1))])

    def track_foreground(self, diff_threshold=None, frames_avg=50,
                         smooth_std=3):
        """Simple motion tracking using an average of the whole video as the
        background and the absolut difference between each frame and the
        background as the foreground.
        """
        avg = self.get_average(frames_avg)
        self.track = []
        self.diffs = []
        for ind, layer in enumerate(self.layers):
            diff = abs(layer.load_image() - avg)
            diff = colors.rgb_to_hsv(diff)[..., 2]
            layer.image = None
            diff = gaussian_filter(diff, smooth_std)
            layer.diff = diff
            if diff_threshold is None:
                xs, ys = local_maxima(diff, disp=False, p=95)
                if len(xs) > 0:
                    self.track += [(xs[0], ys[0])]
                else:
                    self.track += [(np.nan, np.nan)]
            else:
                xs, ys = local_maxima(diff, disp=False,
                                      min_diff=diff_threshold)
                if len(xs) > 0:
                    self.track += [(xs, ys)]
                else:
                    self.track += [(np.nan, np.nan)]
            # self.diffs += [diff]
            # self.track += [(np.argmax(diff.mean(0)),
            #                 np.argmax(diff.mean(1)))]
            print_progress(ind, len(self.layers))

    def color_key(self, samples=5, display=True):
        """Grab unmoving 'background' of the stack by averaging over
        a sample of layers. The default is 50 samples.
        """
        if self.colors is None:
            self.select_color_range(samples=samples)
        if self.track is None:
            print("tracking color range")
            self.track = []
            progress = 0
            for l in self.layers:
                img = l.load_image()
                hsv = colors.rgb_to_hsv(img)
                low_hue = self.colors[:, 0].min()
                hi_hue = self.colors[:, 0].max()
                if low_hue < 0:
                    hues = np.logical_or(
                        hsv[:, :, 0] > 1 + low_hue,
                        hsv[:, :, 0] < hi_hue)
                else:
                    hues = np.logical_and(
                        hsv[:, :, 0] > low_hue,
                        hsv[:, :, 0] < hi_hue)
                sats = np.logical_and(
                    hsv[:, :, 1] > self.colors[:, 1].min(),
                    hsv[:, :, 1] < self.colors[:, 1].max())
                vals = np.logical_and(
                    hsv[:, :, 2] > self.colors[:, 2].min(),
                    hsv[:, :, 2] < self.colors[:, 2].max())
                mask = np.logical_and(hues, sats, vals)
                track = center_of_mass(mask)
                self.track += [(track[1], track[0])]
                # l.image = None
                progress += 1
                print_progress(progress, len(self.layers))
        if display:
            # plt.ion()
            first = True
            for l, (x, y) in zip(self.layers, self.track):
                # l.load_image()
                if first:
                    self.image_fig = plt.imshow(l.image)
                    dot = plt.plot(x, y, 'o')
                    plt.show()
                else:
                    self.image_fig.set_data(l.image)
                    dot[0].set_data(x, y)
                    plt.draw()
                    plt.pause(.001)
                l.image = None
                if first:
                    first = False


class ColorSelector():

    """Provides a GUI for selecting color ranges based on a selected sample of
    hues, saturation, and values. Makes a simple assumption about the
    distribution of those colors, namely that the colors of interest lie within
    2 standard deviations of the selection.
    """

    def __init__(self, frame, bw=False, hue_only=False):
        from matplotlib import gridspec
        self.bw = bw
        if isinstance(frame, str):
            frame = ndimage.imread(frame)
        self.frame = frame
        self.colors = np.array([[0, 0, 0], [1, 1, 255]])
        self.fig = plt.figure(figsize=(8, 8), num="Color Selector")
        self.hue_only = hue_only
        gs = gridspec.GridSpec(6, 4)

        self.pic = self.fig.add_subplot(gs[:3, 1:])
        plt.title("Original")
        plt.imshow(self.frame.astype('uint8'))
        self.key = self.fig.add_subplot(gs[3:, 1:])
        plt.title("Color Keyed")
        self.img = self.key.imshow(self.frame.astype('uint8'))
        self.hues = self.fig.add_subplot(gs[0:2, 0], polar=True)
        plt.title("Hues")
        self.sats = self.fig.add_subplot(gs[2:4, 0])
        plt.title("Saturations")
        self.vals = self.fig.add_subplot(gs[4:, 0])
        plt.title("Values")
        self.fig.tight_layout()

        self.hsv = colors.rgb_to_hsv(self.frame)

        self.hue_dist = list(np.histogram(
            self.hsv[:, :, 0], 255, range=(0, 1), density=True))
        self.hue_dist[0] = np.append(self.hue_dist[0], self.hue_dist[0][0])
        self.sat_dist = np.histogram(
            self.hsv[:, :, 1], 255, range=(0, 1), density=True)
        self.val_dist = np.histogram(
            self.hsv[:, :, 2], 255, range=(0, 255), density=True)

        self.h_line, = self.hues.plot(
            2*np.pi*self.hue_dist[1], self.hue_dist[0], "b")
        self.hues.set_rlim(
            rmin=-.75*self.hue_dist[0].max(),
            rmax=1.1*self.hue_dist[0].max())
        self.s_line, = self.sats.plot(
            self.sat_dist[1][1:], self.sat_dist[0], "r")
        self.v_line, = self.vals.plot(
            self.val_dist[1][1:], self.val_dist[0], "g")

        self.sats.set_xlim(0, 1)
        self.vals.set_xlim(0, 255)
        # self.huespan = self.hues.fill_between(
        #     linspace(2*pi*self.colors[0][0], 2*pi*self.colors[1][0]),
        #     0, max(self.hue_dist[0]),
        #     color="blue", alpha=.3)
        self.satspan = self.sats.axvspan(
            self.colors[0][1], self.colors[1][1], color="red", alpha=.3)
        self.valspan = self.vals.axvspan(
            self.colors[0][2], self.colors[1][2], color="green", alpha=.3)

    def select_color(self, dilate=5):
        hsv = colors.rgb_to_hsv(self.frame.copy())
        self.hue_low = self.colors[:, 0].min()
        self.hue_hi = self.colors[:, 0].max()
        # hue_low, hue_hi = np.percentile(self.hsv[:, 0], [.5, 99.5])
        # self.sats_low, self.sats_hi = np.percentile(self.hsv[:, 1], [2.5, 97.5])
        self.sats_low, self.sats_hi = self.colors[:, 1].min(
        ), self.colors[:, 1].max()
        # self.vals_low, self.vals_hi = np.percentile(self.hsv[:, 2], [2.5, 97.5])
        self.vals_low, self.vals_hi = self.colors[:, 2].min(
        ), self.colors[:, 2].max()
        if self.hue_low < 0:    # if range overlaps 0, use or logic
            self.hue_low = 1 + self.hue_low
            comp_func = np.logical_or
        else:
            comp_func = np.logical_and
        hues = comp_func(
            hsv[:, :, 0] > self.hue_low,
            hsv[:, :, 0] < self.hue_hi)
        sats = np.logical_and(
            hsv[:, :, 1] > self.sats_low,
            hsv[:, :, 1] < self.sats_hi)
        vals = np.logical_and(
            hsv[:, :, 2] > self.colors[:, 2].min(),
            hsv[:, :, 2] < self.colors[:, 2].max())
        hues = ndimage.morphology.binary_dilation(
            hues,
            iterations=dilate).astype("uint8")
        sats = ndimage.morphology.binary_dilation(
            sats,
            iterations=dilate).astype("uint8")
        vals = ndimage.morphology.binary_dilation(
            vals,
            iterations=dilate).astype("uint8")
        if self.bw:
            self.mask = vals
        else:
            if self.hue_only:
                self.mask = hues
            else:
                self.mask = np.logical_and(hues, sats, vals)
        # cv2.inRange(hsv, np.array(colors[0]), np.array(colors[1]))
        # mask = cv2.erode(mask, None, iterations=2)
        self.mask = ndimage.morphology.binary_dilation(
            self.mask,
            iterations=dilate).astype("uint8")
        keyed = self.frame.copy()
        keyed[self.mask == 0] = [0, 0, 0]
        return keyed

    def onselect(self, eclick, erelease):
        'eclick and erelease are matplotlib events at press and release'
        self.select = self.frame[
            int(eclick.ydata):int(erelease.ydata),
            int(eclick.xdata):int(erelease.xdata)]
        if self.select.shape[0] != 0 and self.select.shape[1] != 0:
            self.hsv = colors.rgb_to_hsv(self.select)
            # hsv = hsv.reshape((-1, 3)).transpose()
            m = self.hsv[:, :, 1:].reshape((-1, 2)).mean(0)
            sd = self.hsv[:, :, 1:].reshape((-1, 2)).std(0)
            h_mean = stats.circmean(self.hsv[:, :, 0].flatten(), 0, 1)
            h_std = stats.circstd(self.hsv[:, :, 0].flatten(), 0, 1)
            m = np.append(h_mean, m)
            sd = np.append(h_std, sd)
            self.colors = np.array([m-3*sd, m+3*sd])
            self.keyed = self.select_color()
            self.img.set_array(self.keyed.astype('uint8'))
            self.hue_dist = list(np.histogram(
                self.hsv[:, :, 0], 255,
                range=(0, 1), density=True))
            self.hue_dist[0] = np.append(
                self.hue_dist[0], self.hue_dist[0][0])
            self.sat_dist = np.histogram(
                self.hsv[:, : 1], 255,
                range=(0, 1), density=True)
            self.val_dist = np.histogram(
                self.hsv[:, :, 2], 255,
                range=(0, 255), density=True)
            self.h_line.set_ydata(self.hue_dist[0])
            self.s_line.set_ydata(self.sat_dist[0])
            self.v_line.set_ydata(self.val_dist[0])
            self.hues.set_rlim(
                -.75*max(self.hue_dist[0]), 1.1*max(self.hue_dist[0]))
            self.sats.set_ylim(0, 1.1*max(self.sat_dist[0]))
            self.vals.set_ylim(0, 1.1*max(self.val_dist[0]))
            # self.huespan.set_xy(
            #     self.setAxvspan(self.colors[0][0], self.colors[1][0]))
            self.satspan.set_xy(
                self.setAxvspan(self.colors[0][1], self.colors[1][1]))
            self.valspan.set_xy(
                self.setAxvspan(self.colors[0][2], self.colors[1][2]))
            self.fig.canvas.draw()

    def toggle_selector(event, self):
        print(' Key pressed.')
        if event.key in ['Q', 'q'] and self.RS.active:
            plt.close()
        if event.key in ['A', 'a'] and not self.RS.active:
            print(' RectangleSelector activated.')
            self.RS.set_active(True)

    def setAxvspan(self, x1, x2):
        return np.array([
            [x1, 0.],
            [x1, 1.],
            [x2, 1.],
            [x2, 0.],
            [x1, 0.]])

    def set_radius_span(self, a1, a2, R=1, n=100):
        vals = np.linspace(a1, a2, n)
        vals.sort()
        res = np.zeros(n*2+3)
        res[[0, -1]] = vals.max()
        res[1:n+1] = vals[::-1]
        res[n+1] = vals.min()
        res[n+2:2*n+2] = vals
        return res

    def displaying(self):
        return plt.fignum_exists(self.fig.number)

    def start_up(self):
        from matplotlib.widgets import RectangleSelector

        self.RS = RectangleSelector(
            self.pic, self.onselect, drawtype="box")
        plt.connect('key_press_event', self.toggle_selector)
        plt.show()
