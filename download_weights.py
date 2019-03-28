"""Download the pretrained weights of the networks"""

import os
import sys

import config

def download_weights():
    if sys.version_info >= (3, 0):
        import urllib.request as urltool
    else:
        import urllib as urltool

    code_dir = config.code_dir


    # ROI-net
    print("Downloading pretrained ROI-net")
    roi_net_source = 'http://www-sop.inria.fr/members/Qiao.Zheng/CardiacMotionFlow/ROI/model_roi_net_epoch050.h5'
    roi_net_destination = os.path.join(code_dir, 'ROI', 'model_roi_net_epoch050.h5')
    urltool.urlretrieve(roi_net_source, roi_net_destination)


    # LVRV-net
    print("Downloading pretrained LVRV-net")

    lvrv_net_source = 'http://www-sop.inria.fr/members/Qiao.Zheng/CardiacMotionFlow/segmentation/model_lvrv_net_epoch080.h5'
    lvrv_net_destination = os.path.join(code_dir, 'segmentation', 'model_lvrv_net_epoch080.h5')
    urltool.urlretrieve(lvrv_net_source, lvrv_net_destination)

    for f in range(0, 6):
        lvrv_net_source = 'http://www-sop.inria.fr/members/Qiao.Zheng/CardiacMotionFlow/segmentation/model_lvrv_net_finetune_fold{}_epoch1000.h5'.format(f)
        lvrv_net_destination = os.path.join(code_dir, 'segmentation', 'model_lvrv_net_finetune_fold{}_epoch1000.h5'.format(f))
        urltool.urlretrieve(lvrv_net_source, lvrv_net_destination)


    # ApparentFlow-net
    print("Downloading pretrained ApparentFlow-net")
    for f in range(0, 6):
        apparentflow_net_source = 'http://www-sop.inria.fr/members/Qiao.Zheng/CardiacMotionFlow/flow/model_apparentflow_net_fold{}_epoch050.h5'.format(f)
        apparentflow_net_destination = os.path.join(code_dir, 'flow', 'model_apparentflow_net_fold{}_epoch050.h5'.format(f))
        urltool.urlretrieve(apparentflow_net_source, apparentflow_net_destination)




if __name__ == '__main__':
    download_weights()
