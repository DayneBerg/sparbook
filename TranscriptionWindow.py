import math
from tkinter import Toplevel, ttk, StringVar, IntVar, Checkbutton, E, CENTER, Frame, NORMAL, DISABLED, W

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk


class TranscriptionWindow:
    def __init__(self, sparbook, img, lineheight, network):
        """
        Copyright Dayne Bergman, 2020
        :param sparbook: Sparbook object which created this window
        :param img: the PIL image
        :param lineheight: in pixels
        :param network: neural network for transcribing text
        """
        print("--Transcription Window--")
        assert (img.height % lineheight == 0), "sample's height was invalid"
        # pad edges with median color
        median_color = round(np.median(np.asarray(img)).item())
        image = Image.new(mode="L", size=(img.width, img.height + 4), color=median_color)
        image.paste(img, (0, 2))
        network.train(mode=False)
        self.transcript = []
        self.entries = []
        self.toggles = []
        self.parent = Toplevel(sparbook.root)
        self.parent.columnconfigure(1, weight=1)

        self.rows = math.ceil(img.height / lineheight)
        for i in range(self.rows):
            img_data = image.crop((0, i * lineheight, img.width, (i + 1) * lineheight + 4))
            # size = (W, 31)
            # img_data = img_data.resize((img.width, 16))
            img_data = torch.from_numpy(np.array(img_data, dtype=float)).permute(1, 0)

            # preprocessing
            img_data = img_data - median_color
            # result from dataloader should be N,W,27+4
            # median is zero, but norms of positive and negative values need to be equalized, respectively
            pos = torch.gt(img_data, 0.0)
            pos_norm = torch.sqrt(torch.sum(torch.square(torch.maximum(img_data, torch.zeros_like(img_data)))) / torch.sum(pos))
            neg = torch.lt(img_data, 0.0)
            neg_norm = torch.sqrt(torch.sum(torch.square(torch.minimum(img_data, torch.zeros_like(img_data)))) / torch.sum(neg))
            img_data = torch.where(pos, img_data / pos_norm, img_data / neg_norm)

            img_data = torch.unsqueeze(img_data, dim=0)
            output = torch.argmax(F.log_softmax(network(img_data), dim=2), dim=2)
            prediction = []
            for col in range(output.shape[1]):
                if len(prediction) == 0 or prediction[-1] != output[0][col]:
                    prediction.append(output[0][col])
            prediction = [e for e in prediction if e > 0]
            predicted_str = ''
            for e in prediction:
                char = e
                if char >= 65:  # grave accent (65) should have been replaced with acute accent (8)
                    char += 1
                predicted_str += chr(31 + char)

            self.transcript.append(StringVar())
            self.transcript[i].set(predicted_str)
            self.entries.append(ttk.Entry(self.parent, width=60, textvariable=self.transcript[i]))
            self.entries[i].grid(column=1, row=i)
            self.toggles.append(IntVar())
            self.toggles[i].set(1)
            box = Checkbutton(self.parent, variable=self.toggles[i], command=self.update_states)
            box.grid(column=2, row=i)

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
