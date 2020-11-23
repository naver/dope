# Copyright 2020-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

import sys, os
import argparse
import os.path as osp
from PIL import Image
import cv2
import numpy as np

import torch
from torchvision.transforms import ToTensor

_thisdir = osp.realpath(osp.dirname(__file__))

from model import dope_resnet50, num_joints
import postprocess

import visu

def dope(imagename, modelname, postprocessing='ppi'):
    if postprocessing=='ppi':
      sys.path.append( _thisdir+'/lcrnet-v2-improved-ppi/')
      try:
        from lcr_net_ppi_improved import LCRNet_PPI_improved
      except ModuleNotFoundError:
        raise Exception('To use the pose proposals integration (ppi) as postprocessing, please follow the readme instruction by cloning our modified version of LCRNet_v2.0 here. Alternatively, you can use --postprocess nms without any installation, with a slight decrease of performance.')

    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
      
    # load model
    ckpt_fname = osp.join(_thisdir, 'models', modelname+'.pth.tgz')
    if not os.path.isfile(ckpt_fname):
        raise Exception('{:s} does not exist, please download the model first and place it in the models/ folder'.format(ckpt_fname))
    print('Loading model', modelname)
    ckpt = torch.load(ckpt_fname, map_location=device)
    #ckpt['half'] = False # uncomment this line in case your device cannot handle half computation
    ckpt['dope_kwargs']['rpn_post_nms_top_n_test'] = 1000
    model = dope_resnet50(**ckpt['dope_kwargs'])
    if ckpt['half']: model = model.half()
    model = model.eval()
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
        
    # load the image
    print('Loading image', imagename)
    image = Image.open(imagename)
    imlist = [ToTensor()(image).to(device)]
    if ckpt['half']: imlist = [im.half() for im in imlist]
    resolution = imlist[0].size()[-2:]
    
    # forward pass of the dope network
    print('Running DOPE')
    with torch.no_grad():
        results = model(imlist, None)[0]
    
    # postprocess results (pose proposals integration, wrists/head assignment)
    print('Postprocessing')
    assert postprocessing in ['nms','ppi']
    parts = ['body','hand','face']
    if postprocessing=='ppi':
        res = {k: v.float().data.cpu().numpy() for k,v in results.items()}
        detections = {}
        for part in parts:
            detections[part] = LCRNet_PPI_improved(res[part+'_scores'], res['boxes'], res[part+'_pose2d'], res[part+'_pose3d'], resolution, **ckpt[part+'_ppi_kwargs'])
    else: # nms
        detections = {}
        for part in parts:
            dets, indices, bestcls = postprocess.DOPE_NMS(results[part+'_scores'], results['boxes'], results[part+'_pose2d'], results[part+'_pose3d'], min_score=0.3)
            dets = {k: v.float().data.cpu().numpy() for k,v in dets.items()}
            detections[part] = [{'score': dets['score'][i], 'pose2d': dets['pose2d'][i,...], 'pose3d': dets['pose3d'][i,...]} for i in range(dets['score'].size)]
            if part=='hand':
                for i in range(len(detections[part])): 
                    detections[part][i]['hand_isright'] = bestcls<ckpt['hand_ppi_kwargs']['K']
    
    # assignment of hands and head to body
    detections, body_with_wrists, body_with_head = postprocess.assign_hands_and_head_to_body(detections)
    
    # display results
    print('Displaying results')
    det_poses2d = {part: np.stack([d['pose2d'] for d in part_detections], axis=0) if len(part_detections)>0 else np.empty( (0,num_joints[part],2), dtype=np.float32) for part, part_detections in detections.items()}
    scores = {part: [d['score'] for d in part_detections] for part,part_detections in detections.items()}
    imout = visu.visualize_bodyhandface2d(np.asarray(image)[:,:,::-1],
                                          det_poses2d,
                                          dict_scores=scores,
                                         )
    outfile = imagename+'_{:s}.jpg'.format(modelname)
    cv2.imwrite(outfile, imout)
    print('\t', outfile)
    
    # display results in 3D
    if args.do_visu3d:
        print('Displaying results in 3D')
        import visu3d
        viewer3d = visu3d.Viewer3d()
        img3d, img2d = viewer3d.plot3d(image, 
           bodies={'pose3d': np.stack([d['pose3d'] for d in detections['body']]), 'pose2d' : np.stack([d['pose2d'] for d in detections['body']])},
           hands={'pose3d': np.stack([d['pose3d'] for d in detections['hand']]), 'pose2d' : np.stack([d['pose2d'] for d in detections['hand']])},
           faces={'pose3d': np.stack([d['pose3d'] for d in detections['face']]), 'pose2d' : np.stack([d['pose2d'] for d in detections['face']])},
           body_with_wrists=body_with_wrists,
           body_with_head=body_with_head,
           interactive=False)
        outfile3d = imagename+'_{:s}_visu3d.jpg'.format(modelname)
        cv2.imwrite(outfile3d, img3d[:,:,::-1])
        print('\t', outfile3d)
    
    
    
    



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='running DOPE on an image: python dope.py --model <modelname> --image <imagename>')
    parser.add_argument('--model', required=True, type=str, help='name of the model to use (eg DOPE_v1_0_0)')
    parser.add_argument('--image', required=True, type=str , help='path to the image')
    parser.add_argument('--postprocess', default='ppi', choices=['ppi','nms'], help='postprocessing method')
    parser.add_argument('--visu3d', dest='do_visu3d', default=False, action='store_true')
    args = parser.parse_args()
    dope(args.image, args.model, postprocessing=args.postprocess)
