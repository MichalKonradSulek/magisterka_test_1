def convert_to_savable_format(image):
    return (image * 255).clip(0, 255).astype('uint8')
