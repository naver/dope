# Copyright 2020-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

import numpy as np
import cv2

def _get_bones_and_colors(J, ignore_neck=False): # colors in BGR
    """
    param J: number of joints -- used to deduce the body part considered.
    param ignore_neck: if True, the neck bone of won't be returned in case of a body (J==13)
    """
    if J==13: # full body (similar to LCR-Net)
        lbones = [(9,11),(7,9),(1,3),(3,5)]
        if ignore_neck:
            rbones = [(0,2),(2,4),(8,10),(6,8)] + [(4,5),(10,11)] + [([4,5],[10,11])]
        else:
            rbones = [(0,2),(2,4),(8,10),(6,8)] + [(4,5),(10,11)] + [([4,5],[10,11]),(12,[10,11])]
        bonecolors = [ [0,255,0] ] * len(lbones) + [ [255,0,0] ] * len(rbones)  
        pltcolors = [ 'g-' ] * len(lbones) + [ 'b-' ] * len(rbones)  
        bones = lbones + rbones
    elif J==21: # hand (format similar to HO3D dataset)
        bones = [ [(0,n+1),(n+1,3*n+6),(3*n+6,3*n+7),(3*n+7,3*n+8)] for n in range(5)]
        bones = sum(bones,[])
        bonecolors = [(255,0,255)]*4 + [(255,0,0)]*4 + [(0,255,0)]*4 + [(0,255,255)]*4 + [(0,0,255)] *4
        pltcolors = ['m']*4 + ['b']*4 + ['g']*4 + ['y']*4 + ['r']*4
    elif J==84: # face (ibug format)
        bones = [ (n,n+1) for n in range(83) if n not in [32,37,42,46,51,57,63,75]] + [(52,57),(58,63),(64,75),(76,83)]
        # 32 x contour + 4 x r-sourcil +  4 x l-sourcil + 7 x nose + 5 x l-eye + 5 x r-eye +20 x lip + l-eye + r-eye + lip + lip
        bonecolors = 32 * [(255,0,0)] + 4*[(255,0,0)] + 4*[(255,255,0)] + 7*[(255,0,255)] + 5*[(0,255,255)] + 5*[(0,255,0)] + 18*[(0,0,255)] + [(0,255,255),(0,255,0),(0,0,255),(0,0,255)]
        pltcolors = 32  * ['b']       + 4*['b']       + 4*['c']         + 7*['m']         + 5*['y']         + 5*['g']       + 18*['r']       + ['y','g','r','r']
    else:
        raise NotImplementedError('unknown bones/colors for J='+str(J))
    return bones, bonecolors, pltcolors
    
def _get_xy(pose2d, i):
    if isinstance(i,int):
        return pose2d[i,:]
    else:
        return np.mean(pose2d[i,:], axis=0)
        
def _get_xy_tupleint(pose2d, i):
    return tuple(map(int,_get_xy(pose2d, i)))
    
def _get_xyz(pose3d, i):
    if isinstance(i,int):
        return pose3d[i,:]
    else:
        return np.mean(pose3d[i,:], axis=0)
        
def visualize_bodyhandface2d(im, dict_poses2d, dict_scores=None, lw=2, max_padding=100, bgr=True):
    """
    bgr: whether input/output is bgr or rgb
    
    dict_poses2d: some key/value among {'body': body_pose2d, 'hand': hand_pose2d, 'face': face_pose2d}
    """
    if all(v.size==0 for v in dict_poses2d.values()): return im
     
    h,w = im.shape[:2]
    bones = {}
    bonecolors = {}
    for k,v in dict_poses2d.items():
        bones[k], bonecolors[k], _ = _get_bones_and_colors(v.shape[1])
    
    # pad if necessary (if some joints are outside image boundaries)
    pad_top, pad_bot, pad_lft, pad_rgt = 0, 0, 0, 0
    for poses2d in dict_poses2d.values():
        if poses2d.size==0: continue
        xmin, ymin = np.min(poses2d.reshape(-1,2), axis=0)
        xmax, ymax = np.max(poses2d.reshape(-1,2), axis=0)
        pad_top = max(pad_top, min(max_padding, max(0, int(-ymin-5))))
        pad_bot = max(pad_bot, min(max_padding, max(0, int(ymax+5-h))))
        pad_lft = max(pad_lft, min(max_padding, max(0, int(-xmin-5))))
        pad_rgt = max(pad_rgt, min(max_padding, max(0, int(xmax+5-w))))

    imout = cv2.copyMakeBorder(im, top=pad_top, bottom=pad_bot, left=pad_lft, right=pad_rgt, borderType=cv2.BORDER_CONSTANT, value=[0,0,0] )
    if not bgr: imout = np.ascontiguousarray(imout[:,:,::-1])
    outposes2d = {}
    for part,poses2d in dict_poses2d.items():
        outposes2d[part] = poses2d.copy()
        outposes2d[part][:,:,0] += pad_lft
        outposes2d[part][:,:,1] += pad_top
  
    # for each part
    for part, poses2d in outposes2d.items():
    
        # draw each detection
        for ipose in range(poses2d.shape[0]): # bones
            pose2d = poses2d[ipose,...]

            # draw poses
            for ii, (i,j) in enumerate(bones[part]):    
                p1 = _get_xy_tupleint(pose2d, i)
                p2 = _get_xy_tupleint(pose2d, j)
                cv2.line(imout, p1, p2, bonecolors[part][ii], thickness=lw*2)
            for j in range(pose2d.shape[0]):
                p = _get_xy_tupleint(pose2d, j)
                cv2.circle(imout, p, (2 if part!='face' else 1)*lw, (0,0,255), thickness=-1)
          
            # draw scores
            if dict_scores is not None: cv2.putText(imout, '{:.2f}'.format(dict_scores[part][ipose]), (int(pose2d[12,0]-10),int(pose2d[12,1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0) )
      
    if not bgr: imout = imout[:,:,::-1]
    
    return imout 
