import json
import math
import ntpath
import numpy as np
import os
import statistics
import traceback
from PIL import ImageTk, Image
from PIL.ExifTags import TAGS
from datetime import datetime
from secrets import token_hex
from tkinter import *
from tkinter import ttk, filedialog, messagebox

from ImageHelper import ImageHelper
from TranscriptionWindow import TranscriptionWindow


class Sparbook:
    def __init__(self, **kwargs):
        """
        Copyright Dayne Bergman, 2020
        TODO: implement classifier,
            reduce memory usage w/ PIL & numpy functions,
            make resistant to multiple transcription windows,
            do not error when file save is cancelled,
            decrease imports,
            open with functionality,
            save transcript as,
            remove print statements and deprecated methods pre-release (generate crash logs?)
        :param kwargs:
            filepath: the initial image to be displayed
        """
        self.version = [0, 7]  # 0.7
        self.version_name = 'Beta'
        self.languages = ['No Language', 'Custom', 'English']
        self.imghelper = ImageHelper()
        self.msglist = ("Open an image to proceed.",
                        "Click and drag to make a selection.",
                        "An integer number of lines greater than one must be specified.",  # never used
                        )
        self.cols = 3
        self.x = 0
        self.y = 0
        self.has_selection = False
        self.selection = []
        self.selectioncoords = []
        self.transcript = []

        self.root = Tk()

        def report_callback_exception(*args):
            messagebox.showerror('Exception', traceback.format_exception(*args))

        self.root.report_callback_exception = report_callback_exception
        try:
            with open('sparbook.json') as f:
                self.static_settings = json.load(f)
        except OSError:
            messagebox.showinfo('', 'There was an error while loading and settings may have been lost.')
            self.static_settings = {}
            self.update_static_settings()
        assert (
                0 <= self.static_settings['version'][0] <= self.version[0]
        ), "Internal JSON data has an incompatible version number"
        if self.version[0] > self.static_settings['version'][0] or self.version[1] > self.static_settings['version'][1]:
            messagebox.showinfo('', 'JSON was updated.')
            self.update_static_settings()
        print(self.static_settings)
        print(f"Lang Entries: "
              f"{sum(self.static_settings[self.languages[1]])}, {sum(self.static_settings[self.languages[2]])}")

        self.root.title(f"Sparbook {self.version[0]}.{self.version[1]} {self.version_name}")
        for i in range(self.cols): self.root.columnconfigure(i, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.msg = StringVar()
        self.msg.set(self.msglist[0])
        ttk.Label(self.root, textvariable=self.msg).grid(column=0, row=0, columnspan=self.cols, sticky=(N,))

        self.imagelabel = ttk.Label(self.root)
        self.imagelabel.grid(column=0, row=1, columnspan=self.cols, sticky=())
        self.imagelabel.bind('<Configure>', self.resize_canvas)
        self.canvas = Canvas(self.root)
        self.canvas.grid(column=0, row=1, columnspan=self.cols, sticky=())
        self.canvasimg = self.canvas.create_image(0, 0, image=None, anchor=NW)
        self.canvas.bind('<Button-1>', self.left_down)
        self.canvas.bind('<Motion>', self.motion)
        self.canvas.bind('<ButtonRelease-1>', self.left_up)
        self.canvas.bind('<Button-3>', self.clear_selection)

        txt = StringVar()
        txt.set("Number of lines in selection:")
        ttk.Label(self.root, textvariable=txt).grid(
            column=0,
            row=2,
            columnspan=1,
            sticky=(E, W)
        )

        self.textentry = StringVar()
        self.text_entry = ttk.Entry(self.root, width=50, textvariable=self.textentry)
        self.text_entry.grid(column=1, row=2, columnspan=1, sticky=(E, W))
        self.text_entry.focus()

        self.mainbutton = ttk.Button(self.root, text="Transcribe", command=self.transcribe)
        self.mainbutton.grid(column=2, row=2, columnspan=self.cols, sticky=(W,))
        self.root.bind('<Return>', self.transcribe)

        for child in self.root.winfo_children(): child.grid_configure(padx=5, pady=5)
        menubar = Menu(self.root)

        settingsmenu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settingsmenu)
        self.settings = {'rotate fourier': BooleanVar(),
                         'jello': BooleanVar(),
                         'line-by-line': BooleanVar(),
                         'continuous training': BooleanVar(),
                         'record samples': BooleanVar(),
                         'preferred language': IntVar()
                         }
        for key in self.settings.keys():
            self.settings[key].set(self.static_settings[key])
        settingsmenu.add_checkbutton(label="Correct For Rotation", variable=self.settings['rotate fourier'])
        settingsmenu.add_checkbutton(label="Jello Segmentation", variable=self.settings['jello'])
        settingsmenu.add_checkbutton(label="Write Line-By-Line", variable=self.settings['line-by-line'])
        settingsmenu.add_checkbutton(label="Continuously Train Model", variable=self.settings['continuous training'])
        settingsmenu.add_checkbutton(label="Record Transcription Samples", variable=self.settings['record samples'])
        langmenu = Menu(settingsmenu, tearoff=0)
        settingsmenu.add_cascade(label='Preferred Language', menu=langmenu)
        for i in range(len(self.languages)):
            langmenu.add_radiobutton(
                label=self.languages[i],
                value=i,
                variable=self.settings['preferred language'],
            )

        filemenu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Open Image", command=self.openfile)
        filemenu.add_command(label="Save Transcript", command=self.save_transcript)
        # filemenu.add_command(label="Save Transcript As", command=temp_func)
        filemenu.add_command(label="Clear Transcript", command=self.clear_transcript)

        self.root.config(menu=menubar)

        self.openfile(kwargs.get('filepath', None))
        if not hasattr(self, 'parameters'):  # expensive, and may have been created upon updating json
            self.parameters = np.asarray(self.static_settings['parameters'])  # multithread if possible
        self.root.mainloop()

    def openfile(self, filepath=None):
        """
        :param filepath: optional path of the file to open
        :return: None
        """
        self.root.update_idletasks()  # otherwise text_entry cannot be interacted with
        if filepath is None:
            filepath = filedialog.askopenfilename(parent=self.root)
        self.timestamp = datetime.now()
        head, tail = ntpath.split(filepath)
        leaf = tail or ntpath.basename(head)
        if 'capture' in leaf:
            self.sampleid = self.timestamp.strftime("%Y%m%d%H%M%S")
        else:
            self.sampleid = leaf
        self.samplenum = 0
        self.img = Image.open(filepath)
        self.msg.set(self.msglist[1])
        self.display_img()

        exifdata = self.img.getexif()
        for tag_id in exifdata:
            tag = TAGS.get(tag_id, tag_id)
            data = exifdata.get(tag_id)
            if isinstance(data, bytes):
                data = data.decode()
            print(f"{tag:25}: {data}")

    def get_selection(self):
        """
        :return: smallest PIL Image containing the current selection with mode=L
        """
        if len(self.selectioncoords) == 0:  # return the entire image
            return self.img.convert(mode="L")
        elif len(self.selectioncoords) == 1:  # return a selection of one box
            scale = self.scale_factor(intscale=True)
            return self.img.crop(box=tuple(scale * z for z in self.selectioncoords[0])).convert(mode="L")
        else:  # return a selection made of multiple boxes
            scale = self.scale_factor(intscale=True)
            superbox = (self.img.size[0], self.img.size[1], -1, -1)
            for box in self.selectioncoords:
                superbox = (
                    min(superbox[0], box[0]),
                    min(superbox[1], box[1]),
                    max(superbox[2], box[2]),
                    max(superbox[3], box[3])
                )
            imgsize = (int(scale * (superbox[2] - superbox[0] + 1)), int(scale * (superbox[3] - superbox[1] + 1)))
            actual = self.img.crop(
                box=(
                    scale * superbox[0],
                    scale * superbox[1],
                    scale * (superbox[2] + 1),
                    scale * (superbox[3] + 1)
                )).convert(mode="L")
            blank = Image.new(mode="L", size=imgsize, color=round(np.median(np.asarray(actual))))
            arraymask = np.zeros(shape=(imgsize[1], imgsize[0]), dtype='?')
            maskset = arraymask.itemset
            for box in self.selectioncoords:
                for y in range(int(scale * (box[0] - superbox[0])), int(scale * (box[2] - superbox[0] + 1))):
                    for x in range(int(scale * (box[1] - superbox[1])), int(scale * (box[3] - superbox[1] + 1))):
                        maskset((x, y), True)
            return Image.composite(image1=actual, image2=blank, mask=Image.fromarray(arraymask))

    def transcribe(self, event=None):
        """
        :return: None
        """
        selection = self.get_selection()
        nlines = int(self.textentry.get())
        assert (0 < nlines < selection.size[1] / 2.0), 'Number of lines is outside of accepted range.'
        print("nlines: {}".format(nlines))
        self.root.config(cursor="wait")
        self.root.update()
        (wavelength, phase_shift, img) = self.imghelper.fourier(
            selection,
            nlines,
            rotate=self.settings['rotate fourier'].get(),
            jello=self.settings['jello'].get()
        )
        img2 = self.imghelper.standardize(wavelength, phase_shift, img)
        self.root.config(cursor="")
        self.root.update()
        # img2.show()
        tw = TranscriptionWindow(self, img2, self.imghelper.lineheight)

    def display_img(self):
        """
        Display self.img without mutating it
        """
        self.imgobj = ImageTk.PhotoImage(self.scaled())  # store to prevent garbage collection
        self.imagelabel.configure(image=self.imgobj)
        self.canvas.itemconfig(self.canvasimg, image=self.imgobj)

    def scale_factor(self, **kwargs):
        """
        :param kwargs:
            img: PIL image to compare screen to (default self.img)
            modifier: added to scale factor before potential rounding (default zero)
            intscale: if displayed images should only be rescaled by integer factors (default True)
        :return: the factor by which img should be scaled before being displayed
        """
        img = kwargs.get('img', self.img)
        scale = max(1.0 * img.size[0] / self.root.winfo_screenwidth(),
                    1.0 * img.size[1] / self.root.winfo_screenheight()
                    )
        scale += kwargs.get('modifier', 0.0)
        if kwargs.get('intscale', True): scale = math.ceil(scale)
        return scale

    def scaled(self, **kwargs):
        """
        :param kwargs:
            img: PIL image to be scaled (default self.img)
            modifier: added to scale factor before potential rounding  (default zero)
            intscale: if displayed images should only be rescaled by integer factors (default True)
        :return: img scaled to fit on the current screen
        """
        scale = self.scale_factor(**kwargs)
        img = kwargs.get('img', self.img)
        if scale > 1:  # img is too big for the screen
            print("scale factor: {}".format(scale))
            return img.reduce(scale)
        else:
            return img

    def save_transcript(self):
        """
        :return: None
        """
        file = filedialog.asksaveasfile(mode='w', parent=self.root, initialfile=self.sampleid.split('.')[0])
        if self.settings['line-by-line'].get():
            for textlist in self.transcript:
                file.write('\n'.join(textlist))
                file.write('\n')
        else:
            for textlist in self.transcript:
                file.write('    ')
                for text in textlist:
                    if text[-1] == '-':
                        file.write(text[:-1])
                    else:
                        file.write(text + ' ')
                file.write('\n\n')
        file.write(f"\nTranscribed using {self.root.title()}")
        file.close()

    def update_static_settings(self):
        # reasonable limit is approximately 1,000,000 32-bit numbers
        if 'version' in self.static_settings and len(self.static_settings['version']) > 3:
            v1 = self.static_settings['version'][2] // 2
            v2 = self.static_settings['version'][3]
        else:
            v1 = 0
            v2 = 0
        if 'parameters' in self.static_settings:
            self.parameters = np.nanmean(
                np.array([np.array(self.static_settings['parameters']),
                          np.array([0.0, 7.0])]),
                axis=0
            )
        else:
            self.parameters = np.array([0.0, 7.0])
        new_settings = {'version': self.version + [v1, v2],
                        'dist id': self.static_settings.get('dist id', token_hex(16)),
                        'rotate fourier': self.static_settings.get('rotate fourier', False),
                        'jello': self.static_settings.get('jello', True),
                        'line-by-line': self.static_settings.get('line-by-line', False),
                        'continuous training': self.static_settings.get('continuous training', True),
                        'record samples': self.static_settings.get('record samples', False),
                        'preferred language': self.static_settings.get('preferred language', 2),
                        'stat1': [],
                        'stat2': [],
                        'parameters': self.parameters.tolist(),
                        'updates': self.static_settings.get('updates', [0, 0]),
                        }
        custom = self.static_settings.get('Custom', [0] * 9025)
        if len(custom) != 9025:
            custom = [0] * 9025
        new_settings['Custom'] = custom
        # repeat for every supported language
        english = self.static_settings.get('English', [0] * 9025)
        new_settings['English'] = [0] * 9025
        if len(english) == 9025:
            for e in range(9025):
                new_settings['English'][e] += english[e]
        with open('sparbook.json', 'w') as json_file:
            json.dump(new_settings, json_file)
        self.static_settings = new_settings

    def upon_transcription(self, img, lines):
        """
        :param img: PIL image object in mode L
        :param lines: List of strings representing transcribed lines
        :return: None
        """
        print(f"Lang Entries: "
              f"{sum(self.static_settings[self.languages[1]])}, {sum(self.static_settings[self.languages[2]])}")
        self.transcript.append(lines)
        self.static_settings['rotate fourier'] = self.settings['rotate fourier'].get()
        self.static_settings['jello'] = self.settings['jello'].get()
        self.static_settings['line-by-line'] = self.settings['line-by-line'].get()
        self.static_settings['continuous training'] = self.settings['continuous training'].get()
        self.static_settings['record samples'] = self.settings['record samples'].get()
        self.static_settings['preferred language'] = self.settings['preferred language'].get()
        if self.static_settings['record samples']:  # store the sample
            desc = ''
            for line in lines:
                desc += line + chr(31)
            if self.settings['rotate fourier'].get():
                desc += 'R'
            if self.settings['jello'].get():
                desc += 'J'
            if self.settings['line-by-line'].get():
                desc += 'L'
            if self.settings['continuous training'].get():
                desc += 'C'
            desc += f"{self.settings['preferred language'].get()}"
            exif = Image.Exif()
            exif.__setitem__(270, desc)  # ImageDescription
            exif.__setitem__(305, self.root.title())  # Software
            exif.__setitem__(315, self.static_settings['dist id'])  # Artist
            # exif.__setitem__(32781, hash(self.sampleid) + self.samplenum)           # ImageID
            exif.__setitem__(36868, self.timestamp.strftime("%Y:%m:%d %H:%M:%S"))  # DateTimeDigitized
            print(f"{os.getcwd()}\samples\{self.sampleid.split('.')[0]}{self.samplenum:2}.jpg")
            img.save(f"{os.getcwd()}\samples\{self.sampleid.split('.')[0]}{self.samplenum:2}.jpg", exif=exif)
            self.samplenum += 1
        if self.static_settings['continuous training']:  # update model
            for line in lines:
                self.static_settings['stat1'].append(img.width)
                self.static_settings['stat2'].append(len(line))
            for line in lines:
                if self.static_settings['preferred language'] != 0:
                    left = 0
                    for ch in line:
                        right = ord(ch) - 31
                        assert (0 < right < 96), 'Text contains unsupported characters.'
                        if right == 65:  # replaces grave accent with acute accent
                            right = 8
                        elif right > 65:
                            right += -1
                        l1 = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9, 10, 11, 12, 13, 14, 15, 16, 16, 16,
                             16, 16, 16, 16, 16, 16, 16, 17, 18, 19, 20, 19, 21, 22, 23, 24, 25, 26, 27,
                             28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                             47, 48, 49, 15, 49, 50, 51,  8, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                             34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 52, 53, 52, 54
                             ]  # optimize information storage
                        l2 = (40, 47, 60, 91, 123)  # left-characters
                        l3 = (41, 92, 62, 93, 125)  # right-characters
                        # 65 <= Capital characters <= 90
                        # 97 <= Lowercase Characters <= 122
                        self.static_settings[
                            self.languages[self.static_settings['preferred language']]
                        ][95 * right + left] += 1
                        left = right
                    self.static_settings[self.languages[self.static_settings['preferred language']]][left] += 1
                updates = np.asarray(self.static_settings['updates']) + np.multiply(
                    np.random.randint(-1, 2, self.parameters.shape),
                    np.cbrt(self.parameters)
                )  # calculate and store gradients
                self.static_settings['version'][3] += len(line) + 1
                if self.static_settings['version'][3] >= 128:
                    self.parameters += (updates / 128.0)
                    self.static_settings['parameters'] = self.parameters.tolist()
                    self.static_settings['updates'] = [0, 0]
                    self.static_settings['version'][3] = 0
                    self.static_settings['version'][2] += 1
                else:
                    self.static_settings['updates'] = updates.tolist()
        print(f"Lang Entries: "
              f"{sum(self.static_settings[self.languages[1]])}, {sum(self.static_settings[self.languages[2]])}")
        print(f"Img Width: {min(self.static_settings['stat1'])}, "
              f"{statistics.quantiles(self.static_settings['stat1'], n=4)}, {max(self.static_settings['stat1'])} : "
              f"{len(self.static_settings['stat1'])}, {statistics.mean(self.static_settings['stat1'])}, "
              f"{statistics.stdev(self.static_settings['stat1'])}")
        print(f"Text Width: {min(self.static_settings['stat2'])}, "
              f"{statistics.quantiles(self.static_settings['stat2'], n=4)}, {max(self.static_settings['stat2'])} : "
              f"{len(self.static_settings['stat2'])}, {statistics.mean(self.static_settings['stat2'])}, "
              f"{statistics.stdev(self.static_settings['stat2'])}")
        my_mean = 1.0 * sum(self.static_settings['stat1']) / sum(self.static_settings['stat2'])
        st_dev = 0.0
        for i in range(len(self.static_settings['stat1'])):
            st_dev += self.static_settings['stat2'][i] * (
                    self.static_settings['stat1'][i] / self.static_settings['stat2'][i] - my_mean
            ) ** 2
        print(f"Char Width: {my_mean}, {math.sqrt(st_dev / sum(self.static_settings['stat2']))}")

    def clear_transcript(self):
        self.transcript = []

    def resize_canvas(self, event):
        self.canvas.configure(width=event.width, height=event.height)

    def left_down(self, event):
        self.has_selection = True
        self.x = event.x
        self.y = event.y
        self.selection.append(self.canvas.create_rectangle(event.x, event.y, event.x, event.y, fill=''))

    def motion(self, event):
        if self.has_selection:
            self.canvas.coords(self.selection[-1], (self.x, self.y, event.x, event.y))

    def left_up(self, event):
        if self.has_selection:
            coords = self.canvas.coords(self.selection[-1])
            coords[0] = max(0, coords[0])
            coords[1] = max(0, coords[1])
            coords[2] = min(self.imgobj.width(), coords[2])
            coords[3] = min(self.imgobj.height(), coords[3])
            self.selectioncoords.append(coords)
        self.has_selection = False

    def clear_selection(self, event=None):
        for b in self.selection:
            self.canvas.delete(b)
        self.has_selection = False
        self.selection = []
        self.selectioncoords = []


if __name__ == '__main__':
    gui = Sparbook()  # filename='C:/Users/Dayne/Desktop/Sparbook Database/Smudge Sample/3.jpg')
