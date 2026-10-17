"""Deterministic 224px geometry; no undocumented stochastic coordinate transforms."""
from PIL import Image

def regions(size=(224, 224)):
    w, h = size
    return [{'region_id': f'{r},{c}', 'box': [c*56,r*56,c*56+112,r*56+112],
             'original_box': [c*56*w/224,r*56*h/224,(c*56+112)*w/224,(r*56+112)*h/224],
             'transform': {'resize_from': [w,h], 'resize_to': [224,224]}}
            for r in range(3) for c in range(3)]

def crop_grid(image):
    coords = regions(image.size)
    im = image.convert('RGB').resize((224,224), Image.Resampling.BICUBIC)
    return [im.crop(x['box']) for x in coords], coords
