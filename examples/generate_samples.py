"""Generate the two original RGB fixtures shipped with the repository."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE = 256


def color_chart():
    y, x = np.indices((SIZE, SIZE))
    image = Image.fromarray(np.stack((45 + x // 3, 52 + y // 3, 120 + (x + y) // 6), axis=-1).astype('uint8'))
    draw = ImageDraw.Draw(image)
    colors = ('#f55d53', '#fcc462', '#5cc593', '#53bdce', '#7263bc', '#ef85b7')
    for i, color in enumerate(colors):
        left = 12 + i * 39
        draw.rounded_rectangle((left, 14, left + 34, 64), radius=4, fill=color, outline='#26334c', width=2)
    for i in range(8):
        left = 12 + 29 * i
        gray = 20 + i * 32
        draw.rectangle((left, 205, left + 25, 241), fill=(gray, gray, gray))
    draw.ellipse((16, 82, 122, 190), fill='#2e365d', outline='#f9e4a7', width=3)
    for radius, color in ((39, '#70d7c9'), (24, '#ed846e'), (10, '#f7dc9a')):
        draw.ellipse((70-radius, 136-radius, 70+radius, 136+radius), fill=color)
    draw.rectangle((137, 83, 240, 189), fill='#333d59', outline='white', width=2)
    for i in range(18):
        x0 = 141 + i * 5
        draw.line((x0, 88, x0, 125), fill='#d9f5ed', width=i // 7 + 1)
    for i in range(14):
        draw.line((142, 134 + i * 4, 234, 155 + i * 2), fill=colors[i % len(colors)], width=1)
    return image


def landscape():
    y, x = np.indices((SIZE, SIZE))
    sky = np.stack((54 + y // 2, 109 + y // 3, 167 + y // 5), axis=-1).astype('uint8')
    image = Image.fromarray(sky)
    d = ImageDraw.Draw(image)
    d.ellipse((176, 21, 223, 68), fill='#ffd996')
    d.polygon([(0,124),(51,53),(87,105),(126,68),(151,111),(196,63),(255,123),(255,173),(0,173)], fill='#627b9b')
    for xy in [[(41,67),(51,53),(63,69),(52,65),(48,73)], [(187,75),(196,63),(210,78),(198,72),(193,79)]]:
        d.polygon(xy, fill='#e0edf0')
    d.polygon([(0,151),(59,106),(118,145),(183,102),(255,152),(255,203),(0,203)], fill='#375e6c')
    d.rectangle((0, 152, 255, 255), fill='#4d8c87')
    d.polygon([(0,189),(81,171),(130,199),(169,214),(255,212),(255,255),(0,255)], fill='#678a61')
    rng = np.random.default_rng(83)
    for _ in range(360):
        x0, y0 = map(int, (rng.integers(0, 255), rng.integers(155, 255)))
        d.line((x0, y0, x0+int(rng.integers(2, 15)), y0), fill=('#6caaa3','#477e78','#77a670','#96ae72')[_ % 4])
    for x0, y0, h in [(14, 233, 111), (35, 221, 70), (230, 247, 107), (206, 239, 67)]:
        d.rectangle((x0-2,y0-h,x0+2,y0), fill='#453e3d')
        for step in (0, 1, 2):
            top = y0-h+step*h//5
            half = h//7+step*3
            d.polygon([(x0,top),(x0-half,top+h//2),(x0+half,top+h//2)], fill=('#2d5a4d', '#315c50', '#264d46')[step])
    return image


if __name__ == '__main__':
    out = Path(__file__).resolve().parent / 'samples'
    out.mkdir(exist_ok=True)
    color_chart().save(out / 'sample1.png')
    landscape().save(out / 'sample2.png')
