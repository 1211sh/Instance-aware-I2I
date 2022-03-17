import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import pdb
from .instnce import INSTNCELoss
from torch.autograd import Variable
import torch.nn as nn
import pdb

class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'real_B', 'fake_B' ]
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            # self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']
            self.loss_names += ['recon_B']
            # if opt.nce_inst:
            #     self.loss_names += ['NCE_inst_Y']
        if opt.nce_inst:
            self.loss_names += ['NCE_inst']
        if opt.use_adain:
            self.loss_names += ['style_B']
            # self.loss_names += ['style_recon_rand']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netF_inst = networks.define_F_inst(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            self.instNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            for box_idx in range(self.opt.num_box*self.opt.batch_size// max(len(self.opt.gpu_ids), 1)):    
                self.instNCE.append(INSTNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def getpath(self):
        pdb.set_trace()
        print(image_path)
        return self.image_paths

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        """ modified """
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        if self.opt.use_box:
            self.box_info_A = self.box_info_A[:bs_per_gpu]
            self.box_info_B = self.box_info_B[:bs_per_gpu]
        self.forward()                                          # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                    # calculate gradients for D
            self.compute_G_loss().backward()                    # calculate graidents for G
            
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)
                if self.opt.nce_inst:
                    self.optimizer_F_inst = torch.optim.Adam(self.netF_inst.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                    self.optimizers.append(self.optimizer_F_inst)
               

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
            if self.opt.nce_inst:
                self.optimizer_F_inst.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
            if self.opt.nce_inst:
                self.optimizer_F_inst.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        # pdb.set_trace()
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        image_path = self.image_paths
        #JB_EDIT
        if self.opt.use_box:
            self.box_info_A = input['A_box'].to(self.device)
            self.box_info_B = input['B_box'].to(self.device)

    def test_forward(self, a2b=True): #0828 JH
        if a2b:
            self.real = self.real_A
            if self.opt.use_box:
                self.box_info = self.box_info_A
                self.style_dim = 8
                self.s_b = torch.randn(self.real_B.size(0), self.style_dim, 1, 1).cuda()
                self.fake,_,_ = self.netG(self.real, self.box_info, rand_style=self.s_b,i2i=True) #networks에서 else의 output
            else:
                self.fake = self.netG(self.real)
            self.fake_B = self.fake[:self.real_A.size(0)] #realA: B,3,256,256

        else:
            self.real = self.real_B
            if self.opt.use_box:
                self.box_info_B = self.box_info_B
                self.style_dim = 8
                self.s_a = torch.randn(self.real_A.size(0), self.style_dim, 1, 1).cuda()
                # self.s_a = torch.cat((self.s_a[self.s_a.size(0)//2:],self.s_a[self.s_a.size(0)//2:]),dim=0)
                self.fake,_,_ = self.netG(self.real, self.box_info, rand_style=self.s_a) #networks에서 else의 output
            else:
                self.fake = self.netG(self.real)
            self.fake_B = self.fake[:self.real_A.size(0)] #realA: B,3,256,256
            #self.fake_B_box = self.fake_box[:self.fake_box.size(0)//2]
            #self.fake_B_box_list = self.fake_box_list[:len(self.fake_box_list)//2]
        #return output       
    
        #else:
         ##    if self.opt.use_box:
          #      self.box_info = input['B_box'].to(self.device)
          #      self.style_dim = 8
          #      self.s_A = Variable(torch.randn(input['A'].size(0), self.style_dim*4, 1, 1).cuda()) #4 or 9 - 210827 Edit #rand_style
         #       self.s_A = torch.cat((self.s_A[self.s_A.size(0)//2:],self.s_A[self.s_A.size(0)//2:]),dim=0)
          #      output = self.netG(self.realB, self.box_info, rand_style=self.s_b) #networks에서 else의 output
          #  else:
          #      output = self.netG(self.realB)
          #  self.image_paths = input['B_paths']
        #return output

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # pdb.set_trace()
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        #if self.opt.flip_equivariance:
        #    self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
        #    if self.flipped_for_equivariance:
        #        self.real = torch.flip(self.real, [3])

        #JB_EDIT
        if self.opt.use_box:
            self.box_info = torch.cat((self.box_info_A, self.box_info_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.box_info_A 
            self.style_dim = 8
            self.s_b = torch.randn(self.real_B.size(0), self.style_dim, 1, 1).cuda() #4 or 9 - 210827 Edit #rand_style
            
            self.fake_B, self.fake_B_box, self.fake_B_box_list = self.netG(self.real, self.box_info, rand_style=self.s_b,i2i=True) #networks에서 else의 output
            self.fake_B_box = self.fake_B_box.flatten(0, 1)
            self.fake_B_box_list = self.fake_B_box_list.tolist()

        else:
            self.fake = self.netG(self.real)
        
        ###210906 SH EDIT 210912 EDIT
        if self.opt.nce_idt and self.opt.phase=='train':
            # import pdb;pdb.set_trace()
            self.idt_B, self.idt_B_box, self.idt_B_box_list = self.netG(self.real, self.box_info, rand_style=self.s_b,recon=True) #networks에서 else의 output
            self.idt_B_box = self.idt_B_box.flatten(0, 1)
            self.idt_B_box_list = self.idt_B_box_list.tolist()

            
    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_GAN > 0.0:
            if self.opt.use_adain:                
                self.loss_style_B = self.calculate_style_recon_loss(self.fake_B).mean()*10 

        if self.opt.lambda_NCE > 0.0:
            if self.opt.use_box:
                self.fromBox = 'A'
                try:
                    self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B) *5
                except:
                    self.loss_NCE = 0.0
            if self.opt.nce_inst:
                try:
                    self.loss_NCE_inst = self.calculate_NCE_inst_loss(self.real_A, self.fake_B_box) *3
                except:
                    self.loss_NCE_inst = 0.0
            else:
                self.loss_NCE_inst = 0.0
        else:
            self.loss_NCE, self.loss_NCE_both = 0.0, 0.0
        
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            if self.opt.use_box:
                self.fromBox = 'B'
            self.loss_recon_B = self.calculate_b_recon_loss(self.idt_B, self.real_B).mean() * 10

            # self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            # loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5 # 211011 edit nceboth*2
            loss_NCE_both = (self.loss_NCE) * 0.25
            # if self.opt.nce_inst:
            #     self.loss_NCE_inst_Y = self.calculate_NCE_inst_loss(self.real_B,self.idt_B_box) 
            #     loss_NCE_both += self.loss_NCE_inst_Y*0.5           
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        if self.opt.nce_inst:
            if self.loss_NCE_inst is not None:
                self.loss_G += self.loss_NCE_inst*0.25
                # self.loss_G += self.loss_style_recon_rand*0.5
            else:
                self.loss_NCE_inst = 0.0
                #self.loss_G += self.NCE_inst*0.25
        if self.opt.use_adain:
            self.loss_G += self.loss_style_B
            if self.opt.nce_idt:
                # self.loss_G += self.loss_style_recon_rand*0.5
                self.loss_G += self.loss_recon_B
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        
        if self.opt.use_box:
            self.box_info = self.box_info_A  if self.fromBox == 'A' else self.box_info_B
            feat_q,_ = self.netG(tgt, self.box_info, self.nce_layers, encode_only=True)
        else:
            feat_q,_ = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        if self.opt.use_box:
            
            feat_k,_ = self.netG(src, self.box_info, self.nce_layers, encode_only=True)
        else:
            feat_k,_ = self.netG(src, self.nce_layers, encode_only=True)
        #pdb.set_trace()
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        #feat_q_pool: [512,256],[512,256],[512,256],[128,256],[128,256]
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def calculate_NCE_inst_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        # pdb.set_trace()
        if self.opt.use_box and self.opt.nce_inst:
            # box_index_list [1, 0, 0, 1, 3, 0, 15, 1] 
            self.box_info = self.box_info_A  if self.fromBox == 'A' else self.box_info_B
            if self.fromBox =='A':
                # pdb.set_trace()
                feat_q_none, feat_q_all, box_index_list = self.netG(tgt, self.box_info, self.nce_layers, self.fake_B_box_list, encode_only=True, feat_K=False, feat_Q=True)
                # pdb.set_trace()
                # self.fake_box = self.fake_box.flatten(0, 1)
                # box_index_list = box_index_list.tolist()
            elif self.fromBox =='B':
                feat_q_none, feat_q_all, box_index_list = self.netG(tgt, box_info=None, layers=self.nce_layers, fake_box_list=self.idt_B_box_list, encode_only=True, feat_K=False, feat_Q=True)
            #print("feat_q: ",feat_q.shape) #box_index_list [11, 5, 5, 3, 3, 0, 15, 2]
        else:
            pass
        # pdb.set_trace()
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            #feat_q = [torch.flip(fq, [3]) for fq in feat_q]
            feat_q_all = [torch.flip(fq, [3]) for fq in feat_q_all]
            
            

        if self.opt.use_box:
            if self.opt.nce_inst:
                feat_k_none, feat_k = self.netG(src, self.box_info, self.nce_layers, encode_only=True, feat_K=True,feat_Q=False)
            else:
                pass
        #pdb.set_trace() #feat_q torch.Size([19, 256, 8, 8]) -> [2,256,8,8]
        #FEAT_Q_ALL [80,256,8,8]
        if sum(box_index_list) ==0:
            return None
        # pdb.set_trace()
        feat_q_all = torch.split(feat_q_all,1,dim=0) #list: 21,[2,256,8,8]
        feat_k = torch.split(feat_k,1,dim=0)

        feat_k_pool, sample_ids = self.netF_inst(feat_k, 64, None)
        
        
        feat_q_pool_all, _ = self.netF_inst(feat_q_all, 64, sample_ids)
        #pdb.set_trace()
        #feat_k_pool = torch.cat(feat_k_pool,dim=0)
        feat_k_list=[]
        for i in range(self.opt.batch_size// max(len(self.opt.gpu_ids), 1)):
            feat_k_list.extend(feat_k_pool[0:box_index_list[i]]) #4,8,7,0
            # feat_k_list = feat_k_pool[self.opt.num_box*i:self.opt.num_box*i+box_index_list[i]]
            # try:
            #     feat_k_2 = torch.cat([feat_k_2,feat_k_list],dim=0)
            # except:
            #     feat_k_2 = feat_k_list
            del feat_k_pool[0:self.opt.num_box]

        feat_k_pool = feat_k_list
        feat_q_list=[]
        for i in range(self.opt.batch_size// max(len(self.opt.gpu_ids), 1)):
            feat_q_list.extend(feat_q_pool_all[0:box_index_list[i]]) #4,8,7,0
            # feat_q_list = feat_q_pool_all[self.opt.num_box*i:self.opt.num_box*i+box_index_list[i]]
            # try:
            #     feat_q_2 = torch.cat([feat_q_2,feat_q_list],dim=0)
            # except:
            #     feat_q_2 = feat_q_list
            del feat_q_pool_all[0:self.opt.num_box]
        feat_q_pool = feat_q_list
        #feat_q_pool = feat_q_2 
        #feat_k = torch.split(feat_k,2,dim=0)
        #feat_q_pool, _ = self.netF(feat_q_pool, self.opt.num_patches, sample_ids)
        ##SH_edit 0817
        #feat_q = torch.split(feat_q,2,dim=0) #list: 21,[2,256,8,8]
   

        
        # for i in range(self.opt.batch_size):
        #     feat_q_list.append(feat_q_pool[0:box_index_list[i]]) #4,8,7,0
        #     del feat_q_all[0:self.opt.num_box]
        # feat_q = feat_q_list
        #feat_q = torch.split(feat_q,box_index_list,dim=0) 
        
        total_nce_loss = 0.0

        for f_q, f_k, crit, box_idx in zip(feat_q_pool, feat_k_pool, self.instNCE, range(sum(box_index_list))):
            #pdb.set_trace()
            #print("q",f_q)
            #print("k",f_k)
            #box_count = box_idx
            if torch.eq(f_k, torch.zeros_like(f_k)).sum()>int(64*64):
                if box_idx == 0:
                    return None
                else:
                    return total_nce_loss / (box_idx)
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()


        return total_nce_loss / sum(box_index_list)


    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))
    
    def calculate_style_recon_loss(self, src):
        # pdb.set_trace()
        #idt B, real B
        # s_B_none, s_B = self.netG(tgt, self.box_info_B, self.nce_layers, encode_only=True, style_Real=True)
        # s_idt_B_none, s_idt_B = self.netG(src, self.box_info_B, self.nce_layers, encode_only=True, style_Recon=True)
        s_recon = self.netG(src, self.box_info_A, self.nce_layers, encode_only=True, style_Recon=True)
        # pdb.set_trace()
        return self.criterionIdt(s_recon, self.s_b)
        #return self.criterionIdt(s_idt_B, s_B)


#####Masking for Instance Revisied VER. (by Soohyun) 

    
    def calculate_b_recon_loss(self, src, tgt):
        # pdb.set_trace()
        return self.criterionIdt(src, tgt)

    # def calculate_rand_style_recon_loss(self, src):
    #     #pdb.set_trace()
    #     self.style_dim = 8
    #     s_b = Variable(torch.randn(self.real_B.size(0), self.style_dim, 1, 1).to(src.device))
    #     # s_b = torch.cat((s_b[s_b.size(0)//2:],s_b[s_b.size(0)//2:]),dim=0)
    #     fake_rand_s_b,fake_rand_s_b_box,fake_rand_s_b_list = self.netG(src, self.box_info, layers=[], encode_only=True, rand_style=s_b)
    #     s_idt_b_none, s_idt_b = self.netG(fake_rand_s_b[fake_rand_s_b.size(0)//2:], self.box_info_B, self.nce_layers, encode_only=True, style_Recon=True)        
    #     return self.recon_criterion(s_idt_b, s_b)

    # def calculate_content_recon_loss(self, src):
    #     #pdb.set_trace()
    #     self.style_dim = 8
    #     s_b = Variable(torch.randn(self.real_B.size(0), self.style_dim, 1, 1).to(src.device))
    #     # s_b = torch.cat((s_b[s_b.size(0)//2:],s_b[s_b.size(0)//2:]),dim=0)
    #     fake_rand_s_b,fake_rand_s_b_box,fake_rand_s_b_list = self.netG(src, self.box_info, layers=[], encode_only=True, rand_style=s_b)
    #     s_idt_b_none, s_idt_b = self.netG(fake_rand_s_b[fake_rand_s_b.size(0)//2:], self.box_info_B, self.nce_layers, encode_only=True, style_Recon=True)        
    #     return self.recon_criterion(s_idt_b, s_b)
