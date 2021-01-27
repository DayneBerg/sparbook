import math
import numpy as np
from PIL import Image, ImageTk
from tkinter import *
from tkinter import Toplevel, ttk


class TranscriptionWindow:
    def __init__(self, sparbook, img, lineheight):
        """
        Copyright Dayne Bergman, 2020
        TODO: classifier
        :param sparbook: Sparbook object which created this window
        :param img: the PIL image
        :param lineheight: in pixels
        """
        print("--Transcription Window--")
        # self.lines = []
        self.transcript = []
        self.entries = []
        self.toggles = []
        self.parent = Toplevel(sparbook.root)
        self.parent.columnconfigure(1, weight=1)
        self.rows = math.ceil(img.height / lineheight)
        for i in range(self.rows):
            # add cropped img to lines      ??
            self.transcript.append(StringVar())
            # self.transcript[i].set("Lorem Ipsum")  # transcribe from line
            self.entries.append(ttk.Entry(self.parent, width=60, textvariable=self.transcript[i]))
            self.entries[i].grid(column=1, row=i)
            self.toggles.append(IntVar())
            self.toggles[i].set(1)
            box = Checkbutton(self.parent, variable=self.toggles[i], command=self.update_states)
            box.grid(column=2, row=i)
        '''r = np.asarray(img).copy()
        rget = r.item
        rset = r.itemset
        for x1 in range(self.rows):
            r2 = gaussian_filter(r, sigma=(0, 1))
            r2get = r2.item
            vals = []
            for y in range(r.shape[1]):
                val = 1000000
                #val = rget(lineheight * x1, y)
                for x2 in range(lineheight):
                    #val += math.exp(rget(lineheight * x1 + x2, y))
                    #val += (255 - rget(lineheight * x1 + x2, y)) ** 2
                    #val = min(val, rget(lineheight * x1 + x2, y))
                    val = min(val, (1 + x1 / 4) * rget(lineheight * x1 + x2, y) - r2get(lineheight * x1 + x2, y))
                # val is smallest where there is text
                # vals.append((self.rows * 255 / 2) - val)
                vals.append(val)
            for y in range(r.shape[1]):
                print(vals[y])
                x2 = round((lineheight - 1) * ((vals[y] - min(vals)) / (max(vals) - min(vals))) ** 0.5)
                assert 0 <= x2 < lineheight
                print(x2)
                rset(lineheight*x1+x2, y, 0)'''
        g = np.asarray(img).copy()
        gset = g.itemset
        for x in range(lineheight - 1, img.height - 1, lineheight):
            for y in range(g.shape[1]):
                gset(x, y, 0)
        img2 = Image.merge(mode="RGB", bands=(
            Image.fromarray(g).convert('L'),
            img.convert('L'),
            Image.fromarray(g).convert('L')
        ))
        self.imgobj = ImageTk.PhotoImage(sparbook.scaled(img=img2, modifier=0.0))  # store to prevent garbage collection
        imagelabel = ttk.Label(self.parent, image=self.imgobj)
        imagelabel.grid(column=0, row=0, rowspan=self.rows, sticky=(E,))

        ttk.Label(self.parent, text="Instructions", anchor=CENTER).grid(column=0, row=self.rows, columnspan=3)
        frame = Frame(self.parent)
        frame.grid(column=0, row=self.rows + 1, columnspan=3, pady=4)

        def cancel(*args):
            self.parent.destroy()

        button1 = ttk.Button(frame, text="Cancel", command=cancel)
        button1.grid(column=0, row=0, columnspan=1, sticky=(E, W), padx=5)
        self.parent.bind('<Escape>', cancel)

        def save_transcription(write_file=False, *args):
            lines = []
            nlines = 0
            for e in range(self.rows):
                if self.toggles[e].get():
                    temp = self.transcript[e].get().split()
                    assert (len(temp) > 0), 'Each line must contain text'
                    lines.append(' '.join(temp))
                    nlines += 1
            sample = Image.new("F", (img.width, lineheight * nlines))
            for e in range(self.rows):
                if self.toggles[e].get():
                    sample.paste(
                        img.crop((0, lineheight * e, img.width, lineheight * (e + 1))),
                        (0, sample.height - lineheight * nlines)
                    )
                    nlines += -1
            if sample.height > 0:
                sparbook.upon_transcription(sample.convert('L'), lines)
                if write_file:
                    sparbook.save_transcript()
            self.parent.destroy()

        button2 = ttk.Button(frame, text="Save", command=save_transcription)
        button2.grid(column=1, row=0, columnspan=1, sticky=(E, W), padx=5)
        self.parent.bind('<Return>', lambda foo: save_transcription(write_file=False))

        def save_and_write(*args):
            save_transcription(write_file=True, *args)

        button3 = ttk.Button(frame, text="Save & Write", command=save_and_write)
        button3.grid(column=2, row=0, columnspan=1, sticky=(E, W), padx=5)

        self.parent.focus()

    def update_states(self):
        for i in range(self.rows):
            if self.toggles[i].get():
                self.entries[i].configure(state=NORMAL)
            else:
                self.entries[i].configure(state=DISABLED)
