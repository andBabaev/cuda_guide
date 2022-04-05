from PIL import Image

def load_image(image_file: str, img_size: int=224) -> Image.Image:
    img = Image.open(image_file)
    img = img.resize((img_size, img_size))
    img = img.convert('RGB')
    return img
