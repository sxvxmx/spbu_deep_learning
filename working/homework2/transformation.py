import random
import PIL
import PIL.Image
import numpy as np
import torch
import torchvision

class BaseTransform:
    def __init__(self, p:float) -> None:
        self.prob = p

    def activate(self):
        if random.random() <= self.prob:
            return True
        return False


class RandomCrop(BaseTransform):
    def __init__(self, p:float, **kwds) -> None:
        super().__init__(p)

    def __call__(self, image:PIL.Image, *args, **kwds) -> PIL.Image:
        image = image.copy()
        if(self.activate()):
            width, height = image.size
            size = (random.randint(int(width * 0.05), int(width * 0.95)), random.randint(int(height * 0.05), int(height * 0.95)))
            npimg = np.array(image)
            p1 = np.random.randint(0, abs(npimg.shape[0] - size[1]))
            p2 = np.random.randint(0, abs(npimg.shape[1] - size[0]))
            # for grayscale use )
            if len(npimg.shape) == 2 or (len(npimg.shape) == 3 and npimg.shape[2] == 1):
                mode = "L"
                color = 0
            else:
                mode = "RGB"
                color = (0, 0, 0)
            plate = PIL.Image.new(mode, (npimg.shape[1], npimg.shape[0]), color=color)
            plate.paste(image.crop(box=(p2, p1, p2 + size[0], p1+size[1])), (p2,p1))
            return plate
        return image
    

class RandomRotate(BaseTransform):
    def __init__(self, p: float, **kwds) -> None:
        super().__init__(p)
    
    def __call__(self,image:PIL.Image, *args, **kwds) -> PIL.Image:
        angle = random.randint(-180, 180)
        image = image.copy()
        if(self.activate()):
            return image.rotate(angle)
        return image
    

class RandomZoom(BaseTransform):
    def __init__(self, p: float, **kwds) -> None:
        super().__init__(p)

    def __call__(self, image:PIL.Image,  *args, **kwds) -> PIL.Image:
        image = image.copy()
        if self.activate():
            zoom = random.random() * 2
            width, height = image.size
            point = (random.randint(int(width * 0.15), int(width * 0.85)), random.randint(int(height * 0.15), int(height * 0.85)))
            nwidth = int(width / zoom)
            nheight = int(height / zoom)
            left = max(0, point[0] - nwidth // 2)
            top = max(0, point[1] - nheight // 2)
            right = min(width, left + nwidth)
            bottom = min(height, top + nheight)
            crop = image.crop((left, top, right, bottom))
            out = crop.resize((width, height), PIL.Image.LANCZOS)
            return out
        return image
    

class To_tensor(): 
    def __init__(self) -> None:
        pass

    def __call__(self, image:PIL.Image, *args, **kwds) -> torch.Tensor :
        np_img = np.array(image)
        if len(np_img.shape) == 2:
            np_img = np_img[None, :, :]
        else:
            np_img = np_img.transpose(2, 0, 1)
        tensor = torch.from_numpy(np_img).float() / 255.0
        return tensor
    

class Compose():
    def __init__(self, transformations:list[BaseTransform]) -> None:
        self.tran = transformations
    
    def __call__(self, image:PIL.Image, *args, **kwds) -> PIL.Image:
        image = image.copy()
        for i, item in enumerate(self.tran):
            image = item(image)
        return image
