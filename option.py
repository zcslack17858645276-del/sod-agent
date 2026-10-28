import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('SOD Training', add_help=False)

    parser.add_argument(
        "--input_size", 
        type=int, 
        default=384,
        help="The size of input images"
    )

    # Training configuration
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda"
    )

    parser.add_argument(
        '--batch_size_train',
        default=4, 
        type=int,
        help='Batch size for training'
    )

    parser.add_argument(
        '--max_epoch_num', 
        default=30, 
        type=int, 
        help='Maximum number of training epochs'
    )
    
    parser.add_argument(
        '--model_save_fre',
        default=3, 
        type=int, 
        help='Frequency (in epochs) for saving model checkpoints'
    )
    
    parser.add_argument(
        '--eval_interval', 
        default=3, 
        type=int,
        help='Frequency (in epochs) for running evaluation'
    )

    parser.add_argument(
        '--warmup_epochs',
        default=5,
        type=int,
        help='The number of epochs for warmup'
    )
    
    # Path configuration
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default='./output/',
        help="Path to the directory where checkpoints and records will be output"
    )

    parser.add_argument(
        "--resume_cpt", 
        type=str,
        default = '',
        help="The path to the resume checkpoint"
    )

    parser.add_argument(
        "--visualize", 
        type=str,
        default = False,
        help="Whether to save the prediction"
    )

    parser.add_argument(
        "--pred_dir", 
        type=str,
        default = './pred/',
        help="Path to the directory where predictions will be saved"
    )

    # model configuration
    parser.add_argument(
        "--lr", 
        type=float,
        default=1e-4,
        help="The learning rate for training"
    )

    parser.add_argument(
        "--learning_rate", 
        type=float,
        default=1e-4,
        help="The learning rate for training"
    )

    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-4,
        help="The weight decay for training"
    )

    parser.add_argument(
        "--decay_epoch",
        type=int,
        default=30,
        help="The epoch at which to decay the learning rate"
    )

    parser.add_argument(
        "--sm_loss_weight",
        type=float,
        default=0.5,
        help="The weight for the structure loss"
    )

    parser.add_argument(
        "--freeze_image_encoder",
        type=bool,
        default=True,
        help="Whether to freeze the image encoder during training"
    )

    parser.add_argument(
        "--freeze_dformer",
        type=bool,
        default=True,
        help="Whether to freeze DFormer completely during training"
    )

    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="sam2_hiera_b+.yaml",
        #default="sam2_hiera_l.yaml",
        help="The configuration file for SAM2"
    )

    parser.add_argument(
        "--sam2_ckpt",
        type=str,
        default="checkpoints/sam2_hiera_base_plus.pt",
        #default="checkpoints/sam2_hiera_l.pt",
        help="The checkpoint path for SAM2; if not specified, the model will be loaded from Hugging Face Hub"
    )

    parser.add_argument(
        "--backbone_variant",
        type=str,
        default='hiera_b+',
        #default='hiera_l',
        help="The variant of the backbone model to use"
    )

    parser.add_argument(
        '--res_checkpoint', 
        type=str, 
        default='./res_checkpoints/best_epoch.pth', 
        help='test from checkpoints'
    )

    # configuration for DFormer backbone
    parser.add_argument(
        '--dformer_backbone',
        type=str,
        default='DFormer-Large',
        help='The backbone of DFormer: DFormer-Base, DFormer-Small, DFormer-Tiny, DFormerv2_L, DFormerv2_B, DFormerv2_S'
    )

    parser.add_argument(
        '--dformer_pretrained_model',
        type=str,
        default='checkpoints/dformer_checkpoints/DFormer_Large.pth.tar',
        help='The checkpoint path for DFormer backbone; if not specified, the model will be loaded from ImageNet pretrained weights'
    )

    parser.add_argument(
        '--dformer_drop_path_rate',
        type=float,
        default=0.1,
        help='The drop path rate for DFormer backbone'
    )

    parser.add_argument(
        '--vmamba_pretrained_model',
        type=str,
        default='checkpoints/vmamba_checkpoints/vssmsmall_dp03_ckpt_epoch_238.pth',
        help='The checkpoint path for Vmamba backbone; if not specified, the model will be loaded from ImageNet pretrained weights'
    )

    parser.add_argument(
        '--pvt_pretrained_model',
        type=str,
        default='checkpoints/pvtv2_checkpoints/pvt_v2_b3.pth',
        help='The checkpoint path for PVTv2 backbone; if not specified, the model will be loaded from ImageNet pretrained weights'
    )

    parser.add_argument(
        '--swin_pretrained_model',
        type=str,
        default='checkpoints/swin_checkpoints/swin_base_patch4_window7_224.pth',
        help='The checkpoint path for SwinTransformer backbone; if not specified, the model will be loaded from ImageNet pretrained weights'
    )

    #return parser.parse_args() 
    args = parser.parse_args()

    return args