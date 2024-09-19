import torch
import itertools
import numpy as np
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchvision import transforms



class StainCycleModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.add_argument('--stain_task', type=str, help='specify the stain translation task (HE_HD or HD_HE)')
        parser.add_argument('--lum_mask', action='store_true', help='if specified, use luminance mask to pass background unchanged through the network')
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_SM', type=float, default=0.5, help='weight for stain matrix loss')
            parser.add_argument('--lambda_overlap', type=float, default=10.0, help='weight for overlap loss')
            parser.add_argument('--lambda_L1', type=float, default=0.0, help='weight for L1 loss. Use for approximately aligned data')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_A', 'cycle_A', 'idt_A', 'G_B', 'cycle_B', 'idt_B', 'stain_matrix', 'G_Dual', 'G_L1']
        if self.isTrain and self.opt.overlap > 0:
            self.loss_names.append('overlap')
        if self.isTrain and self.opt.feat_loss:
            self.loss_names.append('feat')
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['big_A', 'big_fake_B', 'real_A', 'rec_A']
        visual_names_B = ['big_B', 'big_fake_A', 'real_B', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'Sep_A', 'Comb_A', 'Sep_B', 'Comb_B']
            if opt.dual_D in ['dual_only', 'all']:
                self.model_names.append('D')
                self.loss_names.append('Dual')
            if opt.dual_D in ['all', 'none']:
                self.model_names += ['D_A', 'D_B']
                self.loss_names += ['D_A', 'D_B']
        else:  # during test time, dont need D's
            self.model_names = ['G_A', 'G_B', 'Sep_A', 'Comb_A', 'Sep_B', 'Comb_B']
        if self.isTrain and opt.l1_downsample > 1:
            self.resize = torch.nn.AdaptiveAvgPool2d((int(256 / opt.l1_downsample), int(256 / opt.l1_downsample)))

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        # default stain matrix
        #stain_matrix = [[0.458,0.814,0.356],[0.259,0.866,0.428],[0.269,0.568,0.778]]
        self.orig_stain_matrixA = torch.tensor([[0.65, 0.70, 0.29],
                    [0.07, 0.99, 0.11],
                    [0.27, 0.57, 0.78]])
        self.orig_stain_matrixB = torch.tensor(             
                [
                    [0.62, 0.637, 0.458],  # H
                    [0.29, 0.832, 0.473],  # CDX2 (pink)
                    [0.3, 0.491, 0.818],  # CDX8 (brown)
                    [0.033, 0.343, 0.939],  # MUC2 (yellow)
                    [0.741, 0.294, 0.604],
                ],  # MUC5 (green),
                )
        self.stain_matrixA = torch.nn.Parameter(torch.empty_like(self.orig_stain_matrixA).copy_(self.orig_stain_matrixA))
        self.stain_matrixB = torch.nn.Parameter(torch.empty_like(self.orig_stain_matrixB).copy_(self.orig_stain_matrixB))

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.stain_task.split("_")[1], opt.lum_mask)
        self.netSep_A = networks.define_stain_sep(self.stain_matrixA, self.gpu_ids)
        self.netComb_A = networks.define_stain_comb(self.stain_matrixB, opt.stain_task.split("_")[1], self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.stain_task.split("_")[0], opt.lum_mask)
        self.netSep_B = networks.define_stain_sep(self.stain_matrixB, self.gpu_ids)
        self.netComb_B = networks.define_stain_comb(self.stain_matrixA, opt.stain_task.split("_")[0], self.gpu_ids)

        if self.isTrain:  # define discriminators
            if opt.dual_D in ['dual_only', 'all']:
                self.netD = networks.define_D(opt.input_nc + 3, opt.ndf, opt.netD_Dual,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            if opt.dual_D in ['all', 'none']:
                self.netD_A = networks.define_D(3, opt.ndf, opt.netD, # hard code 3 here, but need to sort out distinction between num stains and num channels
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                print(f"input_channels: {opt.input_nc}, output_channels: {opt.output_nc}")
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionOverlap = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.netSep_A.parameters(), self.netComb_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.dual_D == 'dual_only':
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.dual_D == 'none':
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters(), self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.tl_blend_frac = torch.ones(256, 256)
            self.tl_blend_frac[:, 256-opt.overlap:] *= torch.linspace(1, 0, opt.overlap).unsqueeze(0).repeat(256, 1)
            self.tl_blend_frac[256-opt.overlap:, :] *= torch.linspace(1, 0, opt.overlap).unsqueeze(1).repeat(1, 256)
            #self.tl_blend_frac[256-opt.overlap:, 256-opt.overlap:] *= torch.linspace(1, 0, opt.overlap).unsqueeze(0).repeat(opt.overlap, 1)
            #self.tl_blend_frac[256-opt.overlap:, 256-opt.overlap:] *= torch.linspace(1, 0, opt.overlap).unsqueeze(1).repeat(1, opt.overlap)
            self.tl_blend_frac = self.tl_blend_frac.unsqueeze(0).repeat(3, 1, 1)
            # rotate the blend frac for the other 3 patches
            self.blend_frac = torch.zeros(4, 3, 256, 256).to(self.device)
            # order is tl, bl, tr, br
            self.blend_frac[0] = self.tl_blend_frac
            self.blend_frac[1] = torch.rot90(self.tl_blend_frac, 1, [1, 2])
            self.blend_frac[2] = torch.rot90(self.tl_blend_frac, -1, [1, 2])
            self.blend_frac[3] = torch.rot90(self.tl_blend_frac, 2, [1, 2])


    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        # squeeze as artificially adding batch size of 4 - needs fixing as will break if
        # use an 'actual' batch size that isn't 1
        if self.opt.isTrain:
            self.real_A = torch.squeeze(input['A' if AtoB else 'B'].to(self.device))
            self.real_B = torch.squeeze(input['B' if AtoB else 'A'].to(self.device))
            self.big_A = input['big_A' if AtoB else 'big_B'].to(self.device)
            self.big_B = input['big_B' if AtoB else 'big_A'].to(self.device)
        else:
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.in_rgb:
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        else:
            self.stain_A = self.netSep_A(self.real_A)
            self.fake_stain_B = self.netG_A(self.stain_A)  # G_A(A)
            self.fake_B = self.netComb_A(self.fake_stain_B)
            self.rec_A = self.netComb_B(self.netG_B(self.netSep_B(self.fake_B)))   # G_B(G_A(A))
            self.stain_B = self.netSep_B(self.real_B)
            self.fake_stain_A = self.netG_B(self.stain_B) 
            self.fake_A = self.netComb_B(self.fake_stain_A)  # G_B(B)
            self.rec_B = self.netComb_A(self.netG_A(self.netSep_A(self.fake_A)))  # G_A(G_B(B))
        self.big_fake_A = self.fuse_patches(self.fake_A * self.blend_frac)
        self.big_fake_B = self.fuse_patches(self.fake_B * self.blend_frac)

    def fuse_patches(self, patches):
        """Fuse the four overlapping patches into a single image"""
        ol = self.opt.overlap
        fused_size = 512 - ol
        fused = torch.zeros((1, 3, fused_size, fused_size)).to(self.device)
        i = 0
        for x in [0, 256 - ol]:
            for y in [0, 256 - ol]:
                fused[:, :, y:y+256, x:x+256] += patches[i]
                i += 1
        return fused # / self.overlap_denom
        

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_dual_D(self, fake_A, real_A, fake_B, real_B):
        """Calculate GAN loss for the discriminator"""
        # stop backprop to the generator by detaching fakes
        # Fake; 
        fake_map_AB = torch.cat((fake_A.detach(), real_B), 1)
        pred_fake = self.netD(fake_map_AB)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_map_AB = torch.cat((real_A, fake_B.detach()), 1)
        pred_real = self.netD(real_map_AB)
        loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_Dual = (loss_D_fake + loss_D_real) * 0.5
        self.loss_Dual.backward()

    def backward_dual_D_G(self, fake_A, real_A, fake_B, real_B):
        """Calculate GAN loss for the discriminator w.r.t. G"""
        # dont detach fakes
        # Fake; 
        fake_map_AB = torch.cat((fake_A, real_B), 1)
        pred_fake = self.netD(fake_map_AB)
        loss_D_fake = self.criterionGAN(pred_fake, True) # say its a real map even though it's fake as we want G to fool D
        # Real
        real_map_AB = torch.cat((real_A, fake_B), 1)
        pred_real = self.netD(real_map_AB)
        loss_D_real = self.criterionGAN(pred_real, False) # say its a fake map even though it's real
        return (loss_D_fake + loss_D_real) * 0.5

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        big_fake_B = self.fake_B_pool.query(self.big_fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.big_B, big_fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        big_fake_A = self.fake_A_pool.query(self.big_fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.big_A, big_fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_SM = self.opt.lambda_SM
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netComb_A(self.netG_A(self.netSep_A(self.real_B)))
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_idt #*lambda_B
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netComb_B(self.netG_B(self.netSep_B(self.real_A)))
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_idt #* lambda_A 
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_G_Dual = 0
        if self.opt.dual_D in ['dual_only', 'all']:
            self.loss_G_Dual = self.backward_dual_D_G()

        self.loss_G_A = self.loss_G_B = 0
        if self.opt.dual_D in ['all', 'none']:
            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(self.big_fake_B), True)
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(self.big_fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # stain matrix loss || stain_matrix - orig_stain_matrix ||
        self.loss_stain_matrix = (
                torch.nn.MSELoss()(self.stain_matrixA, self.orig_stain_matrixA.to(self.device))
                + torch.nn.MSELoss()(self.stain_matrixB, self.orig_stain_matrixB.to(self.device))
            ) * self.opt.lambda_SM
        # do we need a loss that ensures d does not become less when removing one of the other stains from hed?
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_G_Dual + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_stain_matrix
        
        if self.opt.overlap > 0:
            ol = self.opt.overlap
            ol_start = self.opt.crop_size - ol
            # overlap consistency loss. Only considers the 4 'edgewise' overlaps
            # order of 4 patches is tl, bl, tr, br
            self.loss_overlap = torch.tensor(0.0).to(self.device)
            # overlap tl with bl
            self.loss_overlap += self.criterionOverlap(self.fake_A[0, :, ol_start:, :], self.fake_A[1, :, :ol, :])
            self.loss_overlap += self.criterionOverlap(self.fake_B[0, :, ol_start:, :], self.fake_B[1, :, :ol, :])
            # overlap tl with tr
            self.loss_overlap += self.criterionOverlap(self.fake_A[0, :, :, ol_start:], self.fake_A[2, :, :, :ol])
            self.loss_overlap += self.criterionOverlap(self.fake_B[0, :, :, ol_start:], self.fake_B[2, :, :, :ol])
            # overlap tr with br
            self.loss_overlap += self.criterionOverlap(self.fake_A[2, :, ol_start:, :], self.fake_A[3, :, :ol, :])
            self.loss_overlap += self.criterionOverlap(self.fake_B[2, :, ol_start:, :], self.fake_B[3, :, :ol, :])
            # overlap bl with br
            self.loss_overlap += self.criterionOverlap(self.fake_A[1, :, :, ol_start:], self.fake_A[3, :, :, :ol])
            self.loss_overlap += self.criterionOverlap(self.fake_B[1, :, :, ol_start:], self.fake_B[3, :, :, :ol])
            self.loss_G += self.loss_overlap * self.opt.lambda_overlap

        # get the L1 loss between the real and fake images if specified
        if self.opt.lambda_L1 > 0:
            self.loss_G_L1 = self.get_l1_loss(self.real_A, self.fake_A) + self.get_l1_loss(self.real_B, self.fake_B)
            self.loss_G += self.loss_G_L1
        if self.opt.stain_loss:
            # get the L1 loss between the stain images
            pass # implement this
        if self.opt.feat_loss > 0:
            # get the feature loss from the discriminator
            feat_loss_A = self.criterionL1(self.netD_A.module.forward_first_n(self.real_B), self.netD_A.module.forward_first_n(self.fake_B))
            feat_loss_B = self.criterionL1(self.netD_B.module.forward_first_n(self.real_A), self.netD_B.module.forward_first_n(self.fake_A))
            self.loss_feat = (feat_loss_A + feat_loss_B) * self.opt.feat_loss
            self.loss_G += self.loss_feat
        
        self.loss_G.backward()

    def get_l1_loss(self, real, fake):
        if self.opt.l1_downsample > 1:
            # resize the images to calculate L1 loss
            return self.criterionL1(
                self.resize(fake), self.resize(real)
            ) * self.opt.lambda_L1
        else:
            return self.criterionL1(fake, real) * self.opt.lambda_L1
        
    def get_l1_stain_loss(self, real, fake):
        direct_channels = [0, 2 ,3, 4]
        threshold_channels = [1]
        # calculate the L1 loss on the direct channels in stain images
        loss_direct = self.criterionL1(fake[:, direct_channels], real[:, direct_channels]) * self.opt.lambda_L1
        # calculate the soft threshold loss on the threshold channels in stain images
        loss_thresh = self.get_soft_threshold_loss(real[:, threshold_channels], fake[:, threshold_channels]) * self.opt.lambda_L1
        return loss_direct + loss_thresh

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        if self.opt.dual_D in ['all', 'none']:
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        if self.opt.dual_D in ['dual_only', 'all']:
            self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        if self.opt.dual_D in ['all', 'none']:
            self.set_requires_grad([self.netD_A, self.netD_B], True)
        if self.opt.dual_D in ['dual_only', 'all']:
            self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        if self.opt.dual_D in ['dual_only', 'all']:
            self.backward_dual_D()
        if self.opt.dual_D in ['all', 'none']:
            self.backward_D_A()      # calculate gradients for D_A
            self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
