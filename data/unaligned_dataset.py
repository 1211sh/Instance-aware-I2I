import os.path
from data.base_dataset import BaseDataset, get_transform, get_transform_box, BoxHorizontalFlip
from data.image_folder import make_dataset, make_init_dataset #SH_EDIT
from PIL import Image
import random
import util.util as util
#SH_EDIT & JB_EDIT
import numpy as np
import torch
import pdb
class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.dir_A = '/root/dataset1/night_to_day/night'
        self.dir_B = '/root/dataset1/night_to_day/sunny'
        print(self.dir_A)
        print(self.dir_B)

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        #Before data load

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))#[:5000]   # load images from '/path/to/data/trainA' # [:2000]
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))#[:5000]    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        #JB_EDIT
        self.use_box = opt.use_box
        self.num_box = opt.num_box
        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range

        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)

        #JB_EDIT
        if self.use_box:
            transform = get_transform_box(modified_opt)
            A = transform(A_img)
            B = transform(B_img)

            path_split_A = A_path.split('/')
            path_split_B = B_path.split('/')
            
            if self.opt.only_patch:
                direct = 'ONLYPATCH'        #없는 폴더로 빈박스만 load하게끔
            else:
                direct = 'gt_box'           #INIT Box / FasterRCNN Box: 'box'

            

            box_path_A = os.path.join('/',*path_split_A[:-1],direct,(path_split_A[-1][:-4]+'.txt')) #gt_box for INIT box, box for Faster R-CNN box
            box_path_B = os.path.join('/',*path_split_B[:-1],direct,(path_split_B[-1][:-4]+'.txt'))
            
            A_Box = torch.zeros(self.num_box, 5)
            A_Box[:, 0] = -1
            B_Box = torch.zeros(self.num_box, 5)
            B_Box[:, 0] = -1

            try:                                                                                                    

                with open(box_path_A, 'r') as f:                                                                                
                    for i, line in enumerate(f.readlines()):
                        if i >= self.num_box:
                            break
                    
                        param = line.split(' ')

                        if float(param[3]) - float(param[1]) < 0.03:
                            pass
                        else:
                            A_Box[i,0] = int(param[0]) 
                            A_Box[i,1] = float(param[1]) 
                            A_Box[i,2] = float(param[2]) 
                            A_Box[i,3] = float(param[3])
                            A_Box[i,4] = float(param[4])
            except:
                pass
             
            try:
                with open(box_path_B, 'r') as f:
                    for i, line in enumerate(f.readlines()):
                        if i >= self.num_box:
                            break
                    
                        param = line.split(' ')

                        if float(param[1]) - float(param[1]) < 0.03:
                            pass
                        else:
                            B_Box[i,0] = int(param[0]) 
                            B_Box[i,1] = float(param[1]) 
                            B_Box[i,2] = float(param[2]) 
                            B_Box[i,3] = float(param[3])
                            B_Box[i,4] = float(param[4])
            except:
                pass 


            #flip boxes and images
            if self.opt.phase == 'train':
                hor_flip = BoxHorizontalFlip(0.5)
                A, A_Box = hor_flip(A, A_Box)
                B, B_Box = hor_flip(B, B_Box)
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_box': A_Box, 'B_box': B_Box}       

        else:
            transform = get_transform_box(modified_opt)
            A = transform(A_img)
            B = transform(B_img)
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
