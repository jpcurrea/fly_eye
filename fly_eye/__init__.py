import os
import PIL
from Queue import Queue
import numpy as np
import subprocess

from matplotlib import colors
from matplotlib import pyplot as plt
from skimage.draw import ellipse as Ellipse

from numpy.linalg import eig, inv, norm
from scipy import spatial, ndimage, stats
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass

import cv2


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def printProgress(part, whole):
    import sys
    prop = float(part)/float(whole)
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ("="*int(20*prop), 100*prop))
    sys.stdout.flush()


def interpMax(arr, heights=(0.,1.,2.)):
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

    num =   -(y1*(x2 - x3)*(-x2 - x3)
            + y2*(x1 - x3)*(x1 + x3)
            + y3*(x1 - x2)*(-x1 - x2))
    den = 2. * (y1*(x2 - x3)
             -  y2*(x1 - x3)
             +  y3*(x2 - x3))

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


def insideAngle(A, B, C):
    CA = A - C
    CB = B - C
    angles = []
    for x in range(len(A)):
        angles += [np.dot(CA[x], CB[x]) / (norm(CA[x]) * norm(CB[x]))]
    angles = np.array(angles)
    return np.degrees(np.arccos(angles))


def fitEllipse(x, y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def setRadSpan(a1, a2, R=1, n = 100):
    vals = np.linspace(a1, a2, n)
    vals.sort()
    res = np.zeros((n*2+3, 2))
    res[[0, -1], 0] = vals.max()
    res[1:n+1, 0] = vals[::-1]
    res[n+1, 0] = vals.min()
    res[n+2:2*n+2, 0] = vals
    res[0,1] = R
    res[n+1:, 1] = R
    return res


class Layer():

    """A multi-purpose class used for images that is efficient for
    processing video frames.
    """

    def __init__(self, filename, bw=False):
        self.filename = filename
        self.bw = bw
        if isinstance(self.filename, str):
            self.image = None
        elif isinstance(self.filename, np.ndarray):
            self.image = self.filename
        else:
            self.image = None
        self.sob = None
        self.eye_contour = None
        self.ellipse = None

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
            gray = rgb2gray(self.image)
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

    def generateMask(self, thresh=50, b_ground=None):
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

    def crop_eye(self):
        if self.image is None:
            self.load_image()
        if self.ellipse is None:
            self.getEyeSizes()
        (x, y), (w, h), ang = self.ellipse
        self.angle = ang
        w = 1.05*w
        h = 1.05*h
        self.rr, self.cc = Ellipse(x, y, w/2., h/2.,
                                   shape=self.image.shape[:2],
                                   rotation=np.deg2rad(ang))
        out = np.copy(self.image)
        # out = np.zeros(self.image.shape, dtype='uint8')
        # out[self.cc, self.rr] = self.image[self.cc, self.rr]
        # self.eye = out
        self.eye = out[min(self.cc):max(self.cc), min(self.rr):max(self.rr)]
        return self.eye

    def selectColor(self, range=None):
        if self.image is None:
            self.load_image()
        self.cs = colorSelector(self.image)
        self.cs.startUp()
        while self.cs.displaying():
            pass

    def getEyeOutline(self):
        self.selectColor()
        if self.cs.mask is not None:
            im, conts, h = cv2.findContours(
                self.cs.mask,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
            cont = max(conts, key=cv2.contourArea)
            self.eye_contour = cont.reshape((cont.shape[0], cont.shape[-1]))

    def getEyeSizes(self, disp=True):
        if self.eye_contour is None:
            self.getEyeOutline()
        self.ellipse = cv2.fitEllipse(self.eye_contour)
        self.eye_length = max(self.ellipse[1])
        self.eye_width = min(self.ellipse[1])
        self.eye_area = cv2.contourArea(self.eye_contour)
        if disp:
            plt.imshow(self.image)
            plt.plot(self.eye_contour[:, 0], self.eye_contour[:, 1])
            plt.show()

    def get_ommatidial_diameter(self, disp=True):
        self.crop_eye()
        val = colors.rgb_to_hsv(self.eye)[:, :, 2]
        val = rgb2gray(self.eye)
        self.blurred = gaussian_filter(val, sigma=1)
        self.startBlobDetector(minArea=5, maxArea=400, maxThreshold=256,
                               minThreshold=20)
        self.ommatidia = self.detector.detect(self.blurred.astype('uint8'))
        points = np.array([k.pt for k in self.ommatidia])
        self.tree = spatial.KDTree(points)
        dists, inds = self.tree.query(points, k=7)
        dists = dists.T[1:].flatten()
        nan = np.isnan(dists)
        not_nan = nan is False
        not_inf = dists != np.inf
        not_0 = dists > 0
        self.ommatidial_dists = dists[not_nan & not_inf & not_0]
        if disp is True:
            plt.imshow(self.eye.astype('uint8'))
            for p in points:
                plt.plot(p[0], p[1], 'b.')
            # plt.show()
        return self.ommatidial_dists

    def startBlobDetector(
            self, filterByArea=True, minArea=14400, maxArea=422500,
            filterByConvexity=True, minConvexity=.8,
            filterByInertia=False, minInertiaRatio=.05,
            maxInertiaRatio=.25, filterByCircularity=False,
            maxThreshold=256, minThreshold=50):
        """Set up the Blob detector with default params for detecting the
        ticks from our first video.
        """
        self.params = cv2.SimpleBlobDetector_Params()
        self.minThreshold = minThreshold
        self.maxThreshold = maxThreshold
        self.params.filterByArea = filterByArea
        self.params.minArea = minArea
        self.params.maxArea = maxArea
        self.params.filterByConvexity = filterByConvexity
        self.params.minConvexity = minConvexity
        self.params.filterByInertia = filterByInertia
        self.params.minInertiaRatio = minInertiaRatio
        self.params.maxInertiaRatio = maxInertiaRatio
        self.params.filterByCircularity = filterByCircularity
        self.detector = cv2.SimpleBlobDetector_create(self.params)


class Stack():
    """ A class for combining multiple images into one by taking those
    with the highest focus value determined by the sobel operator.
    """
    def __init__(self, dirname="./", f_type=".jpg", bw=False):
        self.dirname = dirname
        fns = os.listdir(self.dirname)
        fns = [os.path.join(dirname, f) for f in fns]
        fns = [f for f in fns if "focus" not in f.lower()]
        fns = sorted(fns)
        fns = [f for f in fns if f.endswith(f_type)]
        fns = [f for f in fns if os.path.split(f)[-1].startswith(".") is False]
        self.layers = []
        for f in fns:
            layer = Layer(f, bw)
            self.layers.append(layer)
            # layer.load_image()
            # if layer.image is not None:
            #     self.layers += [layer]
            #     if len(self.layers) > 1000:
            #         layer.image = None
            # else:
            #     fns.remove(f)
            #     print("File {} is not appropriate format for import".format(f))

        self.images = Queue(maxsize=len(self.layers))
        self.focuses = Queue(maxsize=len(self.layers))
        self.stack = None
        self.bw = bw

    def getEyeMeasurements(self, disp=True):
        self.subjects = []
        self.ommatidial_lengths = []
        self.eye_lengths = []
        self.eye_widths = []
        for layer in self.layers:
            self.subjects += [
                os.path.split(layer.filename)[-1].split(".")[0]]
            layer.getEyeSizes(disp=disp)
            self.ommatidial_lengths += [
                np.median(layer.get_ommatidial_diameter(disp=disp))]

    def getAverage(self, samples=50):
        """Grab unmoving 'background' of the stack by averaging over
        a sample of layers. The default is 50 samples.
        """
        first = self.layers[0].load_image()
        res = np.zeros(first.shape, dtype=float)
        intervals = len(self.layers)/samples
        for l in self.layers[::intervals]:
            img = l.load_image().astype(float)
            res += img
            l.image = None
        return samples**-1*res

    def focusStack(self, smooth=0, interpolate=True, use_all=False,
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
                    printProgress(x, len(self.layers))
                self.focuses = focuses
                self.images = images
                h = interpMax(focuses)
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
            print "done"

    def smooth(self, sigma):
        """A 2d smoothing filter for the heights array"""
        self.heights = self.heights.astype("float32")
        self.heights = np.fft.ifft2(
            ndimage.fourier_gaussian(
                np.fft.fft2(self.heights),
                sigma=sigma)).real


class dialStack(Stack):
    """A special stack class for footage of the dial wheel on the microscope.
    """
    def __init__(self, dirname, f_type, bw=True):
        self.dirname = dirname
        self.f_type = f_type
        Stack.__init__(self, self.dirname, f_type=self.f_type)

    def getMotion(self):
        """Measure the angle of rotation for the tick marks in this dial
        stack.
        """
        self.getAverage()
        self.startBlobDetector()
        self.getCenter()
        self.angles = []
        self.layers[0].generateMask(b_ground = self.average)
        previous = 255 - self.layers[0].mask
        prev_kp = self.detector.detect(previous)
        self.layers[0].keypoints = prev_kp
        x = 1
        whole = len(self.layers)
        for l in self.layers[1:]:
            l.generateMask(b_ground=self.average)
            current = 255 - l.mask
            l.image = None
            curr_kp = self.detector.detect(current)
            l.keypoints = curr_kp
            self.angles += [self.angleRotated(prev_kp, curr_kp, self.center)]
            prev_kp = curr_kp
            printProgress(x, whole)
            x += 1
        self.angles = np.array(self.angles)
        printProgress(x, whole)
        #todo! consider how to get absolute angle rotation by tracking
        #individual ticks throughout the video. Jamie says that this'll
        #contribute less noise
        # return self.angles

    def getCenter(self, samples=5):
        """Find the center of the ellipse based on a sampling of the dial
        ticks
        """
        comb = []
        if self.average is None:
            self.getAverage()
        if self.detector is None:
            self.startBlobDetector()
        intervals = len(self.layers)/samples
        for l in self.layers[::intervals]:
            l.generateMask(b_ground=self.average)
            kp = self.detector.detect(l.mask)
            comb += [k.pt for k in kp]
        comb = np.array(comb)
        comb = comb.T
        self.center = ellipse_center(fitEllipse(comb[0], comb[1]))

    def angleRotated(self, prev_kp, curr_kp, center=None):
        """ Given two sets of points, grab the average angle change given
        that they are due to one general rotation.
        """
        if center is None and self.center is not None:
            if self.center is None:
                self.getCenter()
            center = self.center
        prev_pos = np.array([k.pt for k in prev_kp])
        curr_pos = np.array([k.pt for k in curr_kp])
        tree = spatial.KDTree(prev_pos)
        distance, index = tree.query(curr_pos, distance_upper_bound=3)
        curr_pos = curr_pos[distance != np.inf]
        index = index[distance != np.inf]
        distance = distance[distance != np.inf]
        prev_pos = prev_pos[index]
        angle = insideAngle(prev_pos, curr_pos, center)
        angle = np.mean(angle)
        return angle

    def startBlobDetector(
            self, filterByArea=True, minArea=200, maxArea=1000,
            filterByConvexity=True, minConvexity=.8,
            filterByInertia=False, minInertiaRatio=.05,
            maxInertiaRatio=.25, filterByCircularity=False,
            maxThreshold=256, minThreshold=50):
        """Set up the Blob detector with default params for detecting the
        ticks from our first video.
        """
        self.params = cv2.SimpleBlobDetector_Params()
        self.minThreshold = minThreshold
        self.maxThreshold = maxThreshold
        self.params.filterByArea = filterByArea
        self.params.minArea = minArea
        self.params.maxArea = maxArea
        self.params.filterByConvexity = filterByConvexity
        self.params.minConvexity = minConvexity
        self.params.filterByInertia = filterByInertia
        self.params.minInertiaRatio = minInertiaRatio
        self.params.maxInertiaRatio = maxInertiaRatio
        self.params.filterByCircularity = filterByCircularity
        self.detector = cv2.SimpleBlobDetector_create(self.params)


class Video(Stack):
    import bird_call as aud
    """Takes a stack of images and uses common functions to track motion or
    color."""
    def __init__(self, filename, fps=30, f_type=".jpg"):
        self.vid_formats = [
            '.mov',
            '.mp4',
            '.avi']
        self.filename = filename
        self.f_type = f_type
        self.track = None
        if ((os.path.isfile(self.filename) and
             self.filename.lower()[-4:] in self.vid_formats)):
            # if file is a video, use ffmpeg to generate a jpeg stack
            self.dirname = self.filename[:-4]
            self.fps = subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate", "-of",
                "default=noprint_wrappers=1:nokey=1",
                self.filename])
            self.fps = self.fps.split("/")
            self.fps = float(self.fps[0])/float(self.fps[1])
            if os.path.isdir(self.dirname) is False:
                os.mkdir(self.dirname)
            try:
                failed = subprocess.check_output(
                    ["ffmpeg", "-i", self.filename,
                     "-vf", "scale=360:-1",
                     "./{}/frame%05d{}".format(self.dirname, self.f_type)])
            except subprocess.CalledProcessError:
                print("failed to parse video into {}\
                stack!".format(self.f_type))
                return False
            try:
                audio_fname = "{}.wav".format(self.filename[:-4])
                failed = subprocess.check_output(
                    ["ffmpeg", "-i", self.filename, "-f", "wav",
                     "-ar", "44100",
                     "-ab", "128",
                     "-vn", audio_fname])
                self.audio = self.aud.Recording(audio_fname, trim=False)
            except subprocess.CalledProcessError:
                print("failed to get audio from video!")
                return False

        elif os.path.isdir(self.filename):
            self.dirname = self.filename
            self.fps = fps
        Stack.__init__(self, self.dirname, f_type=self.f_type)
        self.colors = None

    def select_color_range(self, samples=5):
        color_range = []
        intervals = len(self.layers)/samples
        for l in self.layers[::intervals]:
            l.selectColor()
            color_range += [l.cs.colors]
            l.image = None
        color_range = np.array(color_range)
        self.colors = np.array([color_range.min((0, 1)),
                                color_range.max((0, 1))])

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
                printProgress(progress, len(self.layers))
        if display:
            plt.ion()
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


class colorSelector():
    def __init__(self, frame):
        from matplotlib import gridspec
        if isinstance(frame, str):
            frame = ndimage.imread(frame)
        self.frame = frame
        self.colors = np.array([[0, 0, 0], [1, 1, 255]])
        self.fig = plt.figure(figsize=(8, 8), num="Color Selector")
        gs = gridspec.GridSpec(6, 4)

        self.pic = self.fig.add_subplot(gs[:3, 1:])
        plt.title("Original")
        plt.imshow(self.frame)
        self.key = self.fig.add_subplot(gs[3:, 1:])
        plt.title("Color Keyed")
        self.img = self.key.imshow(self.frame)
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

    def selectColor(self, dilate=5):
        hsv = colors.rgb_to_hsv(self.frame.copy())
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
            self.keyed = self.selectColor()
            self.img.set_array(self.keyed)
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

    def setRadSpan(self, a1, a2, R=1, n=100):
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

    def startUp(self):
        from matplotlib.widgets import RectangleSelector

        self.RS = RectangleSelector(
            self.pic, self.onselect, drawtype="box")
        plt.connect('key_press_event', self.toggle_selector)
        plt.show()
