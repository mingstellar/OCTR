model = dict(
    type='SingleStageOcnn3DDetector',
    voxel_size=.01,
    backbone=dict(type='OcnnResNet3D', in_channels=3, depth=34),
    head=dict(
        type='FCAF3DHeadOcnn',
        in_channels=(64, 128, 256, 512),
        out_channels=128,
        voxel_size=.01,
        pts_prune_threshold=100000,
        pts_assign_threshold=27,
        pts_center_threshold=18,
        n_classes=18,
        n_reg_outs=6),
    depth=11,
    full_depth=2,
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))