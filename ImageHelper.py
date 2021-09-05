import cmath
import collections
import math
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageFilter
from scipy.fft import rfft
from scipy.ndimage.filters import gaussian_filter


class ImageHelper:
    def __init__(self):
        """
        Copyright Dayne Bergman, 2020
        """
        self.threshold = 90  # or 80
        self.hscale = 1
        self.lineheight = 27

    def container(self, *args):
        selection = self.get_selection()
        # self.gui.img = Image.eval(self.gui.img, lambda z: 255-z) # invert image
        selection = selection.convert(mode="L").filter(ImageFilter.UnsharpMask(radius=1))  # unsharp mask
        selection = Image.eval(selection, lambda z: 255 * (z <= self.threshold))  # apply threshold
        # print(str(g1.histogram()).replace(",", ";"))
        # (r, g, b) = self.voronoi(img=self.gui.scaled())  # prescale for efficiency
        # (r, g, b) = self.voronoi(img=selection)
        # self.voronoi_store(r, g, b, setting="squared")
        self.gui.img = selection
        self.gui.display_img()
        # os.chdir("C:\\Users\Dayne\Desktop\Sparbook Database\Segmentation Attempts")
        # self.gui.img.save(fp="attempt 4.1.png")
        # self.gui.img.show()

    def old_segment(self, wavelength, phase_shift, img, nlines):
        """
        :param wavelength: vertical height of cycles (in pixels)
        :param phase_shift: vertical offset of lines (in cycles)
        :param img: PIL image object with mode=L
        :param nlines: number of lines of text in img & length of returned array
        :return: list of PIL images with standard height corresponding to individual lines of text
        """
        text_lines = math.ceil(img.size[1] / wavelength + phase_shift - math.floor(phase_shift))
        print("text_lines: {}".format(text_lines))
        assert text_lines - 2 <= nlines <= text_lines
        lines = []
        l0 = math.floor(phase_shift) - phase_shift  # always <= 0
        ln = 0
        k = self.lineheight / wavelength
        img = img.convert(mode="F")
        while ln < text_lines:  # l0 + ln < img.size[1] / wavelength:
            lines.append(img.transform(
                (math.floor(k * img.size[0]), self.lineheight),
                Image.AFFINE,
                data=(1 / k, 0, 0, 0, 1 / k, wavelength * (l0 + ln)),
                resample=Image.BICUBIC,
                fillcolor=np.median(np.asarray(img)))
            )
            ln += 1
        fl0 = img.size[1] / wavelength + phase_shift - math.ceil(img.size[1] / wavelength + phase_shift)
        if text_lines - 2 == nlines or (text_lines - 1 == nlines and l0 <= fl0):
            del lines[0]
        if text_lines - 2 == nlines or (text_lines - 1 == nlines and l0 > fl0):
            del lines[-1]
        assert nlines == len(lines)
        return lines

    def standardize(self, wavelength, phase_shift, img):
        """
        :param wavelength: vertical height of cycles (in pixels)
        :param phase_shift: vertical offset of lines (in cycles). Optionally a list of length img.width
        :param img: PIL image object with mode=L or F
        :return: PIL image where every self.lineheight pixels is a single line of text
        """
        print("--standardize method--")
        k = self.lineheight / wavelength
        if isinstance(phase_shift, collections.Sequence):
            assert (len(phase_shift) == img.width), 'Length of phase array does not match image width.'
            text_lines = math.ceil(img.height / wavelength + max(phase_shift) - math.floor(min(phase_shift)))
            print("text_lines: {}".format(text_lines))
            img = img.convert(mode="F")
            img2 = Image.new("F", (img.width, self.lineheight * text_lines))
            for c in range(img.width):
                img2.paste(
                    img.transform(
                        (1, self.lineheight * text_lines),
                        Image.AFFINE,
                        data=(1, 0, c, 0, 1 / k, -wavelength * (phase_shift[c] - math.floor(min(phase_shift)))),
                        resample=Image.BICUBIC,
                        fillcolor=np.median(np.array(img))
                    ),
                    (c, 0)
                )
            return img2.resize((round(k * img2.width), img2.height), resample=Image.BICUBIC)
        else:
            text_lines = math.ceil(img.height / wavelength + phase_shift - math.floor(phase_shift))
            print("text_lines: {}".format(text_lines))
            img = img.convert(mode="F")
            return img.transform(
                (round(k * img.width), self.lineheight * text_lines),
                Image.AFFINE,
                data=(1 / k, 0, 0, 0, 1 / k, -wavelength * abs(phase_shift)),
                resample=Image.BICUBIC,
                fillcolor=np.median(np.array(img))
            )

    def fourier(self, img, nlines, **kwargs):
        """
        :param img: PIL Image object to be segmented
        :param nlines: number of lines of text in the input image
        :param kwargs:
            scale: horizontal scale factor in voronoi processing (default self.hscale)
            rotate: whether the image should be rotated before segmentation (default False)
            nonlinear: whether the returned phase shift should be specified per-column or uniformly (default False/uniform)
        :return: wavelength, phase_shift, img
        """
        print("--fourier method--")
        print("input img size: {}".format(img.size))
        scale = kwargs.get('scale', self.hscale) ** 2
        (r, g, b, img) = self.voronoi(img=img, crop=True, scale=scale)
        print("voronoi output shape: {}".format(g.shape))
        e = np.sqrt(np.add(scale * np.square(r), np.square(b)))
        mythreshold = e.shape[0] / (3.0 * nlines)
        cumsum = 0
        count = 0
        eget = e.item
        for x in range(e.shape[0]):
            for y in range(e.shape[1]):
                curval = eget((x, y))
                if curval < mythreshold:
                    cumsum += curval
                    count += 1
        mymean = cumsum / count
        eset = e.itemset
        for x in range(e.shape[0]):
            for y in range(e.shape[1]):
                curval = eget((x, y))
                if curval < mythreshold:
                    eset((x, y), curval - mymean)
                else:
                    eset((x, y), 0)
        columns = np.hsplit(e, e.shape[1])
        fts = []
        # sampling length for sufficient frequency resolution
        n = max(e.shape[0], e.shape[0] * (e.shape[0] + nlines) / (nlines ** 2.0))
        # sampling length must be an even integer and is most efficient for powers of 2
        n = 2 ** math.ceil(math.log(n, 2))
        print("n: {}".format(n))
        for column in columns:
            fts.append(rfft(column, n=n, axis=0))
        ft = np.concatenate(fts, axis=1)
        print("ft.shape: {}".format(ft.shape))

        if kwargs.get('rotate', True) or kwargs.get('nonlinear', False):
            fta = np.abs(ft)
            ftam = np.sum(fta, axis=1)
            maxfreq = np.argmax(ftam)
            my_min = math.ceil((nlines - 1.0) * n / e.shape[0])
            my_max = math.ceil((nlines + 1.0) * n / e.shape[0])
            if not (my_min < maxfreq <= my_max):
                messagebox.showinfo('', f'Spatial frequency ({maxfreq}) outside of expected range '
                                        f'({math.ceil((nlines - 1.0) * n / e.shape[0])}, '
                                        f'{math.ceil((nlines + 1.0) * n / e.shape[0])}]')
            print("maxfreq: {}".format(maxfreq))
            wavelength = 1.0 * n / maxfreq
            print("wavelength: {}".format(wavelength))
            reals = gaussian_filter(ft[maxfreq, :].real, sigma=wavelength * 0.65)
            imags = gaussian_filter(ft[maxfreq, :].imag, sigma=wavelength * 0.65)
            realget = reals.item
            imagget = imags.item
            phaseshifts = [math.atan2(imagget(0), realget(0)) / (2 * math.pi)]
            for h in range(1, reals.shape[0]):
                newphaseshift = math.atan2(imagget(h), realget(h)) / (2 * math.pi)
                phaseshifts.append(
                    newphaseshift - round((newphaseshift - phaseshifts[-1]))
                )
            if kwargs.get('rotate', True):
                linreg = np.linalg.lstsq(
                    np.transpose(np.vstack((np.ones(len(phaseshifts)), np.arange(len(phaseshifts))))),
                    phaseshifts,
                    rcond=None
                )[0]
                print("slope: {}".format(linreg[1] * wavelength))
                angle = -math.atan2(linreg[1] * n, maxfreq)  # * 180 / math.pi
                print("angle: {}".format(angle))
                assert (abs(angle) <= 0.8), 'Rotation angle too large.'  # approx math.pi / 4
                print("pythagorean wavelength: {}".format(1 / math.sqrt(wavelength ** -2 + linreg[1] ** 2)))
                # rescale image while rotating to preserve detail
                k = math.ceil(self.lineheight * math.sqrt(wavelength ** -2 + linreg[1] ** 2))
                print("scale factor: {}".format(k))
                rotshape = (
                    math.ceil(k * (img.size[0] * math.cos(angle) + abs(img.size[1] * math.sin(angle)))),
                    math.ceil(k * (abs(img.size[0] * math.sin(angle)) + img.size[1] * math.cos(angle)))
                )
                img = img.convert(mode="F")
                rotimg = img.transform(rotshape, Image.AFFINE, data=(
                    math.cos(angle) / k,
                    -math.sin(angle) / k,
                    (-math.cos(angle) * rotshape[0] / 2 + math.sin(angle) * rotshape[1] / 2) / k + img.size[0] / 2,
                    math.sin(angle) / k,
                    math.cos(angle) / k,
                    (-math.sin(angle) * rotshape[0] / 2 - math.cos(angle) * rotshape[1] / 2) / k + img.size[1] / 2,
                ), resample=Image.BICUBIC, fillcolor=np.median(np.asarray(img)))
                # theoretically fillcolor is not necessary
                # rotimg = img.rotate(angle=angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
                return self.fourier(
                    rotimg,
                    nlines,
                    scale=kwargs.get('scale', self.hscale),
                    rotate=False,
                    nonlinear=kwargs.get('nonlinear', False)
                )  # reprocess the rotated image
            else:  # kwargs.get('nonlinear', False) must be True
                # phase_shifts = np.arctan2(imags, reals) / (2 * math.pi)
                return wavelength, phaseshifts, img
        else:
            ftm = np.sum(ft, axis=1)
            ftma = np.abs(ftm)
            maxfreq = np.argmax(ftma)
            if not math.ceil((nlines - 1.0) * n / e.shape[0]) < maxfreq <= math.ceil((nlines + 1.0) * n / e.shape[0]):
                print("maxfreq not in expected range ({}, {}]".format(
                    math.ceil((nlines - 1.0) * n / e.shape[0]),
                    math.ceil((nlines + 1.0) * n / e.shape[0])
                ))
            print("maxfreq: {}".format(maxfreq))
            wavelength = 1.0 * n / maxfreq
            print("wavelength: {}".format(wavelength))
            phase_shift = cmath.phase(ftm[maxfreq]) / (2 * math.pi)
            print("phase_shift: {}".format(phase_shift))
            return wavelength, phase_shift, img

    def voronoi(self, img, *args, **kwargs):
        """
        TODO: reimplement with a directional vector for increased efficiency
        :param img: PIL Image object to be processed
        :param args:
        :param kwargs:
            crop: whether or not to crop the image to contain all detected text pixels (default False)
            hscale: horizontal scale factor in voronoi processing (default self.hscale)
            progress: whethor or not to print progress to console
        :return:
            r: np array of horizontal distance to nearest text pixel
            g: np array of text pixels
            b: np array of vertical distance to nearest text pixel
            img: cropped input image with mode=L
        """
        print("--voronoi method--")
        img = img.convert(mode="L")
        g = np.asarray(img.filter(ImageFilter.UnsharpMask(radius=1)))
        f1 = lambda z: 255 * (z <= self.threshold)
        g = f1(g)

        r = np.full(shape=g.shape, fill_value=max(g.shape[0], g.shape[1]))
        b = np.full(shape=g.shape, fill_value=max(g.shape[0], g.shape[1]))
        queue = []

        rset = r.itemset  # avoids attribute look-up at each loop iteration
        bset = b.itemset
        inbox = (max(g.shape[0], g.shape[1]), max(g.shape[0], g.shape[1]), -1, -1)
        for p in np.transpose(np.nonzero(g)):
            pos = tuple(p)
            rset(pos, 0)
            bset(pos, 0)
            queue.append(pos)
            inbox = (min(inbox[0], pos[0]), min(inbox[1], pos[1]), max(inbox[2], pos[0]), max(inbox[3], pos[1]))
        if kwargs.get('crop', False):
            print("inbox: {}".format(inbox))
            r = r[inbox[0]:(inbox[2] + 1), inbox[1]:(inbox[3] + 1)]
            g = g[inbox[0]:(inbox[2] + 1), inbox[1]:(inbox[3] + 1)]
            b = b[inbox[0]:(inbox[2] + 1), inbox[1]:(inbox[3] + 1)]
            img = img.crop((inbox[1], inbox[0], inbox[3]+1, inbox[2]+1))
            queue2 = []
            for pos in queue:
                queue2.append((pos[0] - inbox[0], pos[1] - inbox[1]))
            queue = queue2

        hscale = kwargs.get('hscale', self.hscale) ** 2
        interval = round(g.shape[0] * g.shape[1] / 10.0)
        progress = 0
        rset = r.itemset  # avoids attribute look-up at each loop iteration
        bset = b.itemset
        rget = r.item
        bget = b.item
        # could be reimplemented with a directional vector for increased efficiency
        for pos in queue:
            for (x, y) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                new_pos = (pos[0] + x, pos[1] + y)
                if 0 <= new_pos[0] < g.shape[0] and 0 <= new_pos[1] < g.shape[1]:
                    new_eval = hscale * (rget(pos) + x) ** 2 + (bget(pos) + y) ** 2
                    old_eval = hscale * rget(new_pos) ** 2 + bget(new_pos) ** 2
                    if new_eval < old_eval:
                        rset(new_pos, rget(pos) + x)
                        bset(new_pos, bget(pos) + y)
                        queue.append(new_pos)
                        progress = progress + 1
                        if kwargs.get('progress', False) and progress % interval == 0:
                            print("{} ({:.0f}%)".format(progress, 10 * progress / interval))
        print("queue length = {} ({:.0f}%)".format(len(queue), 10 * len(queue) / interval))
        return r, g, b, img

    def voronoi_store(self, r, g, b, **kwargs):
        """
        Deprecated method (as of 0.4) to store the output of voronoi in self.gui.img and display it
        :param r, g, b: output of voronoi
        :param kwargs:
            setting:
                default:
                squared:
                mono:
                nonmaxsup1:
                nonmaxsup1.1:
                nonmaxsup1.2:
                nonmaxsup1.3:
                nonmaxsup4:
                nonmaxsup4.1:
                nonmaxsup2:
                nonmaxsup3: deprecated
            scale: horizontal scale factor for voronoi processing (default self.hscale)
        :return: None
        """
        rset = r.itemset  # avoids attribute look-up at each loop iteration
        bset = b.itemset
        rget = r.item
        bget = b.item
        setting = kwargs.get('setting', None)
        scale = kwargs.get('scale', self.hscale) ** 2
        max_eval = 0
        for x in range(g.shape[0]):
            for y in range(g.shape[1]):
                max_eval = max(max_eval, scale * rget((x, y)) ** 2 + bget((x, y)) ** 2)
        print("max_eval = {}".format(max_eval))
        if setting == "squared":
            for x in range(g.shape[0]):
                for y in range(g.shape[1]):
                    rset((x, y), 255 * (scale * rget((x, y)) ** 2 / max_eval))
                    bset((x, y), 255 * (bget((x, y)) ** 2 / max_eval))
        elif setting == "mono":
            for x in range(g.shape[0]):
                for y in range(g.shape[1]):
                    rset((x, y), 255 * math.sqrt((scale * rget((x, y)) ** 2 + bget((x, y)) ** 2) / max_eval))
                    bset((x, y), rget((x, y)))
        elif setting == "nonmaxsup1":
            c = np.zeros_like(r)
            cset = c.itemset
            for x in range(g.shape[0]):
                for y in range(g.shape[1]):
                    pos = (x, y)
                    if abs(bget(pos)) > abs(rget(pos)):
                        if 0 <= y + np.sign(bget(pos)) < g.shape[1] and abs(bget(pos)) > abs(
                                bget((x, y + np.sign(bget(pos))))):
                            cset(pos, 255)
                    elif 0 <= x + np.sign(rget(pos)) < g.shape[0] and abs(rget(pos)) > abs(
                            rget((x + np.sign(rget(pos)), y))):
                        cset(pos, 255)
            r = c
            b = c
        elif setting == "nonmaxsup1.1":
            c = np.zeros_like(r)
            cset = c.itemset
            for x in range(g.shape[0]):
                for y in range(g.shape[1]):
                    pos = (x, y)
                    if abs(bget(pos)) > abs(rget(pos)):
                        if 0 <= y + np.sign(bget(pos)) < g.shape[1] and scale * rget(pos) ** 2 + bget(
                                pos) ** 2 > scale * rget((x, y + np.sign(bget(pos)))) ** 2 + bget(
                            (x, y + np.sign(bget(pos)))) ** 2:
                            cset(pos, 255)
                    elif 0 <= x + np.sign(rget(pos)) < g.shape[0] and scale * rget(pos) ** 2 + bget(pos) > scale * rget(
                            (x + np.sign(rget(pos)), y)) ** 2 + bget((x + np.sign(rget(pos)), y)):
                        cset(pos, 255)
            r = c
            b = c
        elif setting == "nonmaxsup1.2":  # best so far
            c = np.zeros_like(r)
            cset = c.itemset
            for x in range(g.shape[0]):
                for y in range(g.shape[1]):
                    pos = (x, y)
                    if 0 <= y + np.sign(bget(pos)) < g.shape[1] and abs(bget(pos)) > abs(
                            bget((x, y + np.sign(bget(pos))))):
                        cset(pos, 255)
                    if 0 <= x + np.sign(rget(pos)) < g.shape[0] and abs(rget(pos)) > abs(
                            rget((x + np.sign(rget(pos)), y))):
                        cset(pos, 255)
            r = c
            b = c
        elif setting == "nonmaxsup1.3":
            c = np.zeros_like(r)
            cset = c.itemset
            for x in range(g.shape[0]):
                for y in range(g.shape[1]):
                    pos = (x, y)
                    if 0 <= y + np.sign(bget(pos)) < g.shape[1] and scale * rget(pos) ** 2 + bget(
                            pos) ** 2 > scale * rget((x, y + np.sign(bget(pos)))) ** 2 + bget(
                        (x, y + np.sign(bget(pos)))) ** 2:
                        cset(pos, 255)
                    if 0 <= x + np.sign(rget(pos)) < g.shape[0] and scale * rget(pos) ** 2 + bget(pos) > scale * rget(
                            (x + np.sign(rget(pos)), y)) ** 2 + bget((x + np.sign(rget(pos)), y)):
                        cset(pos, 255)
            r = c
            b = c
        elif setting == "nonmaxsup4":
            k = np.add(scale * np.square(r), np.square(b))
            c = np.zeros_like(r)
            cset = c.itemset
            kget = k.item
            for x in range(g.shape[0]):
                for y in range(g.shape[1]):
                    val = kget((x, y))
                    d = 3
                    if 0 <= y - 1 and y + 1 < g.shape[1]:
                        d = min(d, np.sign(kget((x, y - 1)) - val) + np.sign(kget((x, y + 1)) - val))
                    if 0 <= x - 1 and x + 1 < g.shape[0]:
                        d = min(d, np.sign(kget((x - 1, y)) - val) + np.sign(kget((x + 1, y)) - val))
                    if d < 0:
                        cset((x, y), 255)
            r = c
            b = c
        elif setting == "nonmaxsup4.1":
            k = np.add(scale * np.square(r), np.square(b))
            k = gaussian_filter(k, sigma=1, output="float", mode='nearest')
            c = np.zeros_like(r)
            cset = c.itemset
            kget = k.item
            for x in range(g.shape[0]):
                for y in range(g.shape[1]):
                    val = kget((x, y))
                    d = 3
                    if 0 <= y - 1 and y + 1 < g.shape[1]:
                        d = min(d, np.sign(kget((x, y - 1)) - val) + np.sign(kget((x, y + 1)) - val))
                    if 0 <= x - 1 and x + 1 < g.shape[0]:
                        d = min(d, np.sign(kget((x - 1, y)) - val) + np.sign(kget((x + 1, y)) - val))
                    if d < 0:
                        cset((x, y), 255)
            r = c
            b = c
        elif setting == "nonmaxsup2":
            c = np.zeros_like(r)
            cset = c.itemset
            for x in range(g.shape[0]):
                for y in range(g.shape[1]):
                    pos = (x, y)
                    x2 = scale * rget(pos) ** 2
                    y2 = bget(pos) ** 2
                    if x2 + y2 > 0:
                        if min(x2, y2) / max(x2, y2) > 3 - math.sqrt(8):  # vector is approximately diagonal
                            newpos = (x + np.sign(rget(pos)), y + np.sign(bget(pos)))
                        elif y2 > x2:
                            newpos = (x, y + np.sign(bget(pos)))
                        else:
                            newpos = (x + np.sign(rget(pos)), y)
                        if not (0 <= newpos[0] < g.shape[0] and 0 <= newpos[1] < g.shape[1]) or x2 + y2 > scale * rget(
                                newpos) ** 2 + bget(newpos) ** 2:
                            cset(pos, 255)
            r = c
            b = c
        else:
            for x in range(g.shape[0]):
                for y in range(g.shape[1]):
                    rset((x, y), 255 * math.sqrt(scale * rget((x, y)) ** 2 / max_eval))
                    bset((x, y), 255 * math.sqrt(bget((x, y)) ** 2 / max_eval))
        self.gui.img = Image.merge(
            mode="RGB",
            bands=(
                Image.fromarray(r.astype("uint8"), mode="L"),
                Image.fromarray(g.astype("uint8"), mode="L"),
                Image.fromarray(b.astype("uint8"), mode="L")
            )
        )


if __name__ == '__main__':
    instance = ImageHelper()
    # C:\Users\Dayne\Desktop\Sparbook Database\Smudge Sample\3.jpg
    # C:\Users\Dayne\Desktop\Sparbook Database\Smudge Sample\25.jpg
    # smudge: 1 4 12 15 16 19
    # skew: 3 21 22 23 24 25
