from torchvision import transforms
from auto_augment import AutoAugment, Cutout
from PIL import Image
import numpy as np

class ISIC2019_Augmentations():
    def __init__(self, is_training, image_size=256, input_size=224):
        mdlParams = dict()
        
        mdlParams['setMean'] = np.array([0.0, 0.0, 0.0])   
        mdlParams['setStd'] = np.array([1.0, 1.0, 1.0])
        self.image_size = image_size
        mdlParams['input_size'] = [input_size, input_size, 3]
        self.input_size = (np.int32(mdlParams['input_size'][0]),np.int32(mdlParams['input_size'][1]))   

        # training augmentations
        if is_training:
            self.same_sized_crop = True
            self.only_downsmaple = False

            #mdlParams['full_color_distort'] = True
            mdlParams['autoaugment'] = True
            mdlParams['flip_lr_ud'] = True
            mdlParams['full_rot'] = 180
            mdlParams['scale'] = (0.8,1.2)
            mdlParams['shear'] = 10
            mdlParams['cutout'] = 16

            transforms = self.get_train_augmentations(mdlParams)
        else:
            # test augmentations
            transforms = self.get_test_augmentations(mdlParams)
        self.transforms = transforms
    
    def get_test_augmentations(self, mdlParams):
        all_transforms = [
                transforms.ToPILImage(),
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(np.float32(mdlParams['setMean']),np.float32(mdlParams['setStd']))]
        composed = transforms.Compose(all_transforms)
        return composed

    def get_train_augmentations(self, mdlParams):
        all_transforms = [transforms.ToPILImage()]
        # Normal train proc
        if self.same_sized_crop:
            all_transforms.append(transforms.Resize(self.image_size))
            all_transforms.append(transforms.RandomCrop(self.input_size))
        elif self.only_downsmaple:
            all_transforms.append(transforms.Resize(self.input_size))
        else:
            all_transforms.append(transforms.RandomResizedCrop(self.input_size[0],scale=(mdlParams.get('scale_min',0.08),1.0)))
        if mdlParams.get('flip_lr_ud',False):
            all_transforms.append(transforms.RandomHorizontalFlip())
            all_transforms.append(transforms.RandomVerticalFlip())
        # Full rot
        if mdlParams.get('full_rot',0) > 0:
            if mdlParams.get('scale',False):
                all_transforms.append(transforms.RandomChoice([transforms.RandomAffine(mdlParams['full_rot'], scale=mdlParams['scale'], shear=mdlParams.get('shear',0), interpolation=Image.NEAREST),
                                                            transforms.RandomAffine(mdlParams['full_rot'],scale=mdlParams['scale'],shear=mdlParams.get('shear',0), interpolation=Image.BICUBIC),
                                                            transforms.RandomAffine(mdlParams['full_rot'],scale=mdlParams['scale'],shear=mdlParams.get('shear',0), interpolation=Image.BILINEAR)]))
            else:
                all_transforms.append(transforms.RandomChoice([transforms.RandomRotation(mdlParams['full_rot'], interpolation=Image.NEAREST),
                                                            transforms.RandomRotation(mdlParams['full_rot'], interpolation=Image.BICUBIC),
                                                            transforms.RandomRotation(mdlParams['full_rot'], interpolation=Image.BILINEAR)]))
        # Color distortion
        if mdlParams.get('full_color_distort') is not None:
            all_transforms.append(transforms.ColorJitter(brightness=mdlParams.get('brightness_aug',32. / 255.),saturation=mdlParams.get('saturation_aug',0.5), contrast = mdlParams.get('contrast_aug',0.5), hue = mdlParams.get('hue_aug',0.2)))
        else:
            all_transforms.append(transforms.ColorJitter(brightness=32. / 255.,saturation=0.5))   
        # Autoaugment
        if mdlParams.get('autoaugment',False):
            all_transforms.append(AutoAugment())             
        # Cutout
        if mdlParams.get('cutout',0) > 0:
            all_transforms.append(Cutout_v0(n_holes=1,length=mdlParams['cutout']))                             
        # Normalize
        all_transforms.append(transforms.ToTensor())
        all_transforms.append(transforms.Normalize(np.float32(mdlParams['setMean']),np.float32(mdlParams['setStd'])))            
        # All transforms
        composed = transforms.Compose(all_transforms)         

        return composed


class Cutout_v0(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        img = np.array(img)
        #print(img.shape)
        h = img.shape[0]
        w = img.shape[1]

        mask = np.ones((h, w), np.uint8)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        #mask = torch.from_numpy(mask)
        #mask = mask.expand_as(img)
        img = img * np.expand_dims(mask,axis=2)
        img = Image.fromarray(img)
        return img    