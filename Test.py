from PIL import Image
from PIL.ExifTags import TAGS
from tkinter import *
from tkinter import filedialog


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

    # filename = f"{os.getcwd()}\samples\Chicago World Fair 1 1.jpg"
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
    exifdata.__setitem__(270, exifdata.get(270).replace('acheived', 'achieved'))
    # img.save(filename, exif=exifdata)
