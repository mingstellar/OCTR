_base_ = ['fcaf3d-ocnn-scannet.py']
n_points = 100000

model = dict(
    head=dict(
        type='FCAF3DHeadOcnnMS'
    )
)
