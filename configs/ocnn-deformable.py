model = dict(
    type='OcnnDeformableDETR',
    voxel_size=.01,
    backbone=dict(type='OcnnResNet3D', in_channels=3, depth=34),
    head=dict(
        type='Deformable3DHead',
        in_channels=(64, 128, 256, 512),
        out_channels=128,
        n_classes=18,
        n_reg_outs=6,
        voxel_size=.01,),
    
    depth=11,
    full_depth=2,
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))

find_unused_parameters = True

