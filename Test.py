import torchtext as torchtext
from torchtext.vocab import GloVe


class Test:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.has_selection = False


if __name__ == '__main__':
    # my_car = Car()
    # my_car.exec()
    # x = np.array([[0,       1, 2,   3,   4, 5, 6],
    # [-2.5, -1.5, 0, 1.5, 2.5, 3.5-2*math.pi, 4.5-2*math.pi]])

    batch_size = 2  # 2700

    print('0')
    # set up fields
    SRC = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True)
    TRG = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True)

    print('1')
    train, val, test = torchtext.datasets.WMT14.splits(
        exts=('.en', '.de'),
        fields=(SRC, TRG),
        root='C:/Users/Dayne/Documents/Data',
    )

    print('2')
    SRC.build_vocab(train, vectors=GloVe(name='6B', dim=300))

    print('3')
    train_i = torchtext.data.Iterator(train, batch_size=batch_size)

    print('4')

    temp = train_i.__iter__().__next__()
    print(type(temp))
    print(type(temp[0]))
    print(type(temp[0][0]))
    print(type(temp[0][0][0]))
    print(type(temp[0][0][0][0]))

    '''# get summary statistics of samples
    img_names = os.listdir(f"{os.getcwd()}\samples")
    nlines = []
    img_width = []
    num_chars = []
    char_width = []
    gamma = []
    for img_name in img_names:
        with Image.open(f"{os.getcwd()}\samples\{img_name}") as cur_img:
            exifdata = cur_img.getexif()
            data = exifdata.get(270).split(chr(31))[:-1]
            num_lines = len(data)
            nlines.append(num_lines)
            img_width.extend([cur_img.width] * num_lines)
            hist = cur_img.histogram()
            for i in range(1, len(hist)):
                hist[i] += hist[i-1]
            hist2 = []
            for i in range(len(hist)):
                try:
                    temp = math.log(hist[i]/hist[-1], (2 * i - 1)/(2 * len(hist)))
                    hist2.append(temp)
                except ValueError:
                    pass
            cur_gamma = statistics.mean(hist2)
            gamma.extend([cur_gamma] * num_lines)
            for text in data:
                num_chars.append(1+len(text))  # must add unit seperator back
                char_width.append(cur_img.width / len(text))
    print(f"Nlines: {min(nlines)}, "
          f"{statistics.quantiles(nlines, n=4)}, {max(nlines)} : "
          f"{len(nlines)}, {statistics.mean(nlines)}, "
          f"{statistics.stdev(nlines)}")
    print(f"Img Width: {min(img_width)}, "
          f"{statistics.quantiles(img_width, n=4)}, {max(img_width)} : "
          f"{len(img_width)}, {statistics.mean(img_width)}, "
          f"{statistics.stdev(img_width)}")
    print(f"Num Chars: {min(num_chars)}, "
          f"{statistics.quantiles(num_chars, n=4)}, {max(num_chars)} : "
          f"{len(num_chars)}, {statistics.mean(num_chars)}, "
          f"{statistics.stdev(num_chars)}")
    print(f"Char Width: {min(char_width)}, "
          f"{statistics.quantiles(char_width, n=4)}, {max(char_width)} : "
          f"{len(char_width)}, {statistics.mean(char_width)}, "
          f"{statistics.stdev(char_width)}")
    print(f"Gamma: {min(gamma)}, "
          f"{statistics.quantiles(gamma, n=4)}, {max(gamma)} : "
          f"{len(gamma)}, {statistics.mean(gamma)}, "
          f"{statistics.stdev(gamma)}")'''

    '''img = Image.open('C:/Users/Dayne/Pictures/Saved Pictures/IMG_0195.JPG').convert(mode="L")
    img.show()
    #ang = 0.3
    wavelength = img.size[0] / 8
    k = 32 / wavelength
    newsize = (math.floor(k * img.size[0]), 32)
    print(newsize)
    newimg = img.transform(newsize, Image.AFFINE, data=(
        1 / k,
        0,
        0,
        0,
        1 / k,
        -wavelength/2,# + img.size[1] / 2,
    ), resample=Image.BICUBIC, fillcolor=255)
    newimg.show()'''

    '''with open('sparbook.json') as f:
        static_settings = json.load(f)
    engl = np.array(static_settings['English']).reshape((95, 95))  # + np.ones((95, 95))
    #ef = np.hstack([a / np.sum(a, dtype='float64') for a in np.hsplit(engl, 95)])
    #print(np.round(0.013 * engl[:, 1]))
    #           2nd 1st
    # hsplit forwards, vsplit backwards
    # 0:unit 1:space 14:-
    # space- 13
    # -space 2
    right = ord('-')-31
    if right == 65:  # replaces grave accent with acute accent
        right = 8
    elif right > 65:
        right += -1
    #print(right)'''
    '''engl[ord('?')-31, ord('"')-31] = 0
    engl[ord('?')-31, ord('1')-31] = 1
    temp1 = np.hsplit(engl, 96)
    temp2 = []
    for e in range(96):
        if e != ord('`')-31:
            temp2.append(temp1[e])
    temp3 = np.vsplit(np.hstack(temp2), 96)
    temp4 = []
    for e in range(96):
        if e != ord('`')-31:
            temp4.append(temp3[e])'''
    # static_settings['Custom'] = [0] * 9025
    # static_settings['English'] = engl.flatten().tolist()  # np.vstack(temp4).flatten().tolist()
    '''with open('sparbook.json', 'w') as json_file:
        json.dump(static_settings, json_file)'''

    '''# filename = f"{os.getcwd()}\samples\Chicago World Fair 1 1.jpg"
    filename = filedialog.askopenfilename()
    img = Image.open(filename)
    img.show()
    exifdata = img.getexif()
    for tag_id in exifdata:
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        if isinstance(data, bytes):
            data = data.decode()
        print(f"{tag:25}: {data}")
    # exifdata.__setitem__(270, exifdata.get(270).replace('?', '1').replace(' -', '-').replace('- ', '-'))
    exifdata.__setitem__(270, exifdata.get(270).replace('accomodations', 'accommodations'))
    # 4
    # img.save(filename, exif=exifdata)'''
