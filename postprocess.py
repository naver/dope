# Copyright 2020-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

import numpy as np 
import torch
from torchvision.ops import nms

def _boxes_from_poses(poses, margin=0.1): # pytorch version
  x1y1,_ = torch.min(poses, dim=1) # N x 2
  x2y2,_ = torch.max(poses, dim=1) # N x 2
  coords = torch.cat( (x1y1,x2y2), dim=1)
  sizes = x2y2-x1y1
  coords[:,0:2] -= margin * sizes
  coords[:,2:4] += margin * sizes
  return coords
  
def DOPE_NMS(scores, boxes, pose2d, pose3d, min_score=0.5, iou_threshold=0.1):
  if scores.numel()==0:
    return torch.LongTensor([]), torch.LongTensor([])
  maxscores, bestcls = torch.max(scores[:,1:], dim=1)
  valid_indices = torch.nonzero(maxscores>=min_score)
  if valid_indices.numel()==0:
    return torch.LongTensor([]), torch.LongTensor([])
  else:
    valid_indices = valid_indices[:,0]
  
  boxes = _boxes_from_poses(pose2d[valid_indices,bestcls[valid_indices],:,:], margin=0.1)
  indices = valid_indices[ nms(boxes, maxscores[valid_indices,...], iou_threshold) ]
  bestcls = bestcls[indices]
  
  return {'score': scores[indices, bestcls+1], 'pose2d': pose2d[indices, bestcls, :, :], 'pose3d': pose3d[indices, bestcls, :, :]}, indices, bestcls
  
  
  


def _get_bbox_from_points(points2d, margin):
    """ 
    Compute a bounding box around 2D keypoints, with a margin.
    margin: the margin is relative to the size of the tight bounding box

    """
    assert (len(points2d.shape)==2 and points2d.shape[1] == 2)
    mini = np.min(points2d, axis=0)
    maxi = np.max(points2d, axis=0)
    size = maxi-mini
    lower = mini - margin*size
    upper = maxi + margin*size
    box = np.concatenate((lower, upper)).astype(np.float32)
    return box  

def assign_hands_to_body(body_poses, hand_poses, hand_isright, margin=1):
  if body_poses.size==0: return []
  if hand_poses.size==0: return [(-1,-1) for i in range(body_poses.shape[0])]
  from scipy.spatial.distance import cdist
  body_rwrist = body_poses[:,6,:]
  body_lwrist = body_poses[:,7,:]
  hand_wrist = hand_poses[:,0,:]
  hand_boxes = np.concatenate([_get_bbox_from_points(hand_poses[i,:,:], margin=0.1)[None,:] for i in range(hand_poses.shape[0])], axis=0)
  hand_size = np.max(hand_boxes[:,2:4]-hand_boxes[:,0:2], axis=1)
  # associate body and hand if the distance hand-body and body-hand is the smallest one and is this distance is smaller than 3*hand_size
  wrists_from_body = [(-1,-1) for i in range(body_poses.shape[0])] # pair of (left_hand_id, right_hand_id)
  dist_lwrist = cdist(body_lwrist,hand_wrist)
  dist_rwrist = cdist(body_rwrist,hand_wrist)
  for i in range(body_poses.shape[0]):
    lwrist = -1
    rwrist = -1
    if hand_wrist.size>0:
      best_lwrist = np.argmin(dist_lwrist[i,:])
      if np.argmin(dist_lwrist[:,best_lwrist])==i and dist_lwrist[i,best_lwrist] <= margin * hand_size[best_lwrist]:
        lwrist = best_lwrist
      best_rwrist = np.argmin(dist_rwrist[i,:])
      if np.argmin(dist_rwrist[:,best_rwrist])==i and dist_rwrist[i,best_rwrist] <= margin * hand_size[best_rwrist]:
        rwrist = best_rwrist
    wrists_from_body[i] = (lwrist,rwrist)
  return wrists_from_body # pair of (left_hand_id, right_hand_id) for each body pose (-1 means no association)

def assign_head_to_body(body_poses, head_poses):
  if body_poses.size==0: return []
  if head_poses.size==0: return [-1 for i in range(body_poses.shape[0])]
  head_boxes = np.concatenate([_get_bbox_from_points(head_poses[i,:,:], margin=0.1)[None,:] for i in range(head_poses.shape[0])], axis=0)
  body_heads = body_poses[:,12,:]
  bodyhead_in_headboxes = np.empty( (body_poses.shape[0], head_boxes.shape[0]), dtype=np.bool)
  for i in range(body_poses.shape[0]):
    bodyhead = body_heads[i,:]
    bodyhead_in_headboxes[i,:] = (bodyhead[0]>=head_boxes[:,0]) * (bodyhead[0]<=head_boxes[:,2]) * (bodyhead[1]>=head_boxes[:,1]) * (bodyhead[1]<=head_boxes[:,3])
  head_for_body = []
  for i in range(body_poses.shape[0]):
    if np.sum(bodyhead_in_headboxes[i,:])==1:
      j = np.where(bodyhead_in_headboxes[i,:])[0][0]
      if np.sum(bodyhead_in_headboxes[:,j])==1:
        head_for_body.append(j)
      else:
        head_for_body.append(-1)
    else:
      head_for_body.append(-1)
  return head_for_body
  
  
  
  
def assign_hands_and_head_to_body(detections):
  det_poses2d = {part: np.stack([d['pose2d'] for d in part_detections], axis=0) if len(part_detections)>0 else np.empty( (0,0,2), dtype=np.float32) for part, part_detections in detections.items()}
  hand_isright = np.array([ d['hand_isright'] for d in detections['hand']])
  body_with_wrists = assign_hands_to_body(det_poses2d['body'], det_poses2d['hand'], hand_isright, margin=1)
  BODY_RIGHT_WRIST_KPT_ID = 6
  BODY_LEFT_WRIST_KPT_ID = 7
  for i,(lwrist,rwrist) in enumerate(body_with_wrists):
      if lwrist != -1: detections['body'][i]['pose2d'][BODY_LEFT_WRIST_KPT_ID,:] = detections['hand'][lwrist]['pose2d'][0,:]
      if rwrist != -1: detections['body'][i]['pose2d'][BODY_RIGHT_WRIST_KPT_ID,:] = detections['hand'][rwrist]['pose2d'][0,:]
  body_with_head = assign_head_to_body(det_poses2d['body'], det_poses2d['face'])
  return detections, body_with_wrists, body_with_head
  
