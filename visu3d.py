# Copyright 2020-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use


import numpy as np
import visvis as vv
import visu
import scipy.optimize
import copy
import PIL

def scale_orthographic(points3d, points2d):
    """
    Return a scaled set of 3D points, offseted in XY direction so as to minimize the distance to the 2D points
    """
    residuals_func1 = lambda x : ((x[None,:2] + points3d[:,:2]) - points2d).flatten()
    res1 = scipy.optimize.least_squares(residuals_func1, x0=[0,0], method='lm')
    residuals_func2 = lambda x : (np.exp(x[2]) * (x[None,:2] + points3d[:,:2]) - points2d).flatten()
    res2 = scipy.optimize.least_squares(residuals_func2, x0=np.concatenate((res1['x'],[0])), method='lm')
    x=res2['x']
    output3d = points3d.copy()
    output3d[:,:2]+=x[None,:2]
    output3d*=np.exp(x[2])
    return output3d
            


class Viewer3d:
    def __init__(self, display2d=True, camera_zoom=None, camera_location=None):
        self.app = vv.use('qt5')
        self.figsize = (1280+20,720+20)
        self.display2d = display2d
        self.camera_zoom = camera_zoom
        self.camera_location = camera_location


    def plot3d(self, img, 
               bodies={'pose3d':np.empty((0,13,3)), 'pose2d':np.empty((0,13,2))},
               hands={'pose3d':np.empty((0,21,3)), 'pose2d':np.empty((0,21,2))}, 
               faces={'pose3d':np.empty((0,84,3)), 'pose2d':np.empty((0,84,2))}, 
               body_with_wrists=[],
               body_with_head=[],
               interactive=False):
        """
        :param img: a HxWx3 numpy array
        :param bodies: dictionnaroes with 'pose3d' (resp 'pose2d') with the body 3D (resp 2D) pose
        :param faces: same with face pose
        :param hands: same with hand pose
        :param body_with_wrists: list with for each body, a tuple (left_hand_id, right_hand_id) of the index of the hand detection attached to this body detection (-1 if none) for left and right hands
        :parma body_with_head: list with for each body, the index of the face detection attached to this body detection (-1 if none)
        :param interactive: whether to open the viewer in an interactive manner or not
        """
        
        # body pose do not use the same coordinate systems        
        bodies['pose3d'][:,:,0] *= -1
        bodies['pose3d'][:,:,1] *= -1
                
        # Compute 3D scaled representation of each part, stored in "points3d"
        hands, bodies, faces = [copy.copy(s) for s in (hands, bodies, faces)]
        parts = (hands, bodies, faces)
        for part in parts:
            part['points3d'] = np.zeros_like(part['pose3d'])
            for part_idx in range(len(part['pose3d'])):
                points3d = scale_orthographic(part['pose3d'][part_idx], part['pose2d'][part_idx])  
                part['points3d'][part_idx] = points3d
            
        # Various display tricks to make the 3D visualization of full-body nice
        # (1) for faces, add a Z offset to faces to align them with the body
        for body_id, face_id in enumerate(body_with_head):
            if face_id !=-1:
                z_offset = bodies['points3d'][body_id,12,2] - np.mean(faces['points3d'][face_id,:,2])
                faces['points3d'][face_id,:,2] += z_offset
        # (2) for hands, add a 3D offset to put them at the wrist location
        for body_id, (lwrist_id, rwrist_id) in enumerate(body_with_wrists):
            if lwrist_id != -1:
                hands['points3d'][lwrist_id,:,:] = bodies['points3d'][body_id,7,:] - hands['points3d'][lwrist_id,0,:]
            if rwrist_id != -1:
                hands['points3d'][rwrist_id,:,:] = bodies['points3d'][body_id,6,:] - hands['points3d'][rwrist_id,0,:]                
        
        img = np.asarray(img)
        height, width = img.shape[:2]
        

        fig=vv.figure(1)
        fig.Clear()        
        
        fig._SetPosition(0,0,self.figsize[0], self.figsize[1])
        if not interactive:
            fig._enableUserInteraction=False

        axes = vv.gca()
        # Hide axis
        axes.axis.visible = False
        
        scaling_factor = 1.0/height
        
        # Camera interaction is not intuitive along z axis
        # We reference every object to a parent frame that is rotated to circumvent the issue
        ref_frame = vv.Wobject(axes)
        ref_frame.transformations.append(vv.Transform_Rotate(-90, 1,0,0))
        ref_frame.transformations.append(vv.Transform_Translate(-0.5*width*scaling_factor, -0.5, 0))
        
        # Draw image
        if self.display2d:
            # Display pose in 2D
            img = visu.visualize_bodyhandface2d(img, 
                                            dict_poses2d={'body': bodies['pose2d'],
                                                          'hand': hands['pose2d'],
                                                          'face': faces['pose2d']},
                                            lw=2, max_padding=0, bgr=False)
            
            XX, YY = np.meshgrid([0,width*scaling_factor],[0, 1])
            img_z_offset = 0.5
            ZZ = img_z_offset * np.ones(XX.shape)
            # Draw image
            embedded_img = vv.surf(XX, YY, ZZ, img)
            embedded_img.parent = ref_frame
            embedded_img.ambientAndDiffuse=1.0

            # Draw a grid on the bottom floor to get a sense of depth
            XX, ZZ = np.meshgrid(np.linspace(0, width*scaling_factor, 10), img_z_offset - np.linspace(0, width*scaling_factor, 10))
            YY = np.ones_like(XX)
            grid3d = vv.surf(XX, YY, ZZ)
            grid3d.parent = ref_frame
            grid3d.edgeColor=(0.1,0.1,0.1,1.0)
            grid3d.edgeShading='plain'
            grid3d.faceShading=None

        
        # Draw pose
        for part in parts:

            for part_idx in range(len(part['points3d'])):
                points3d = part['points3d'][part_idx]*scaling_factor
                # Draw bones
                J = len(points3d)
                is_body = (J==13)
                ignore_neck = False if not is_body else body_with_head[part_idx]!=-1
                bones, bonecolors, pltcolors = visu._get_bones_and_colors(J, ignore_neck=ignore_neck)
                for (kpt_id1, kpt_id2), color in zip(bones, bonecolors):
                    color = color[2], color[1], color[0] # BGR vs RGB
                    p1 = visu._get_xyz(points3d, kpt_id1)
                    p2 = visu._get_xyz(points3d, kpt_id2)
                    pointset=vv.Pointset(3)
                    pointset.append(p1)
                    pointset.append(p2)

                    # Draw bones as solid capsules
                    bone_radius = 0.005
                    line = vv.solidLine(pointset, radius=bone_radius)
                    line.faceColor = color
                    line.ambientAndDiffuse=1.0

                    line.parent = ref_frame
                
                # Draw keypoints, except for faces
                if J != 84:
                    keypoints_to_plot = points3d
                    if ignore_neck:
                        # for a nicer display, ignore head keypoint
                        keypoints_to_plot=keypoints_to_plot[:12,:]
                    # Use solid spheres
                    for i in range(len(keypoints_to_plot)):
                        kpt_wobject = vv.solidSphere(translation=keypoints_to_plot[i,:].tolist(), scaling=1.5*bone_radius)
                        kpt_wobject.faceColor = (255,0,0)
                        kpt_wobject.ambientAndDiffuse=1.0
                        kpt_wobject.parent = ref_frame
        
        # Use just an ambient lighting
        axes.light0.ambient=0.8
        axes.light0.diffuse=0.2
        axes.light0.specular=0.0
        
        cam = vv.cameras.ThreeDCamera()
        axes.camera=cam
        #z axis
        cam.azimuth=-45
        cam.elevation=20
        cam.roll=0
        # Orthographic camera
        cam.fov=0
        if self.camera_zoom is None:
            cam.zoom *= 1.3 # Zoom a bit more
        else:
            cam.zoom = self.camera_zoom
        if self.camera_location is not None:
            cam.loc = self.camera_location
        cam.SetView()
        

        if interactive:
            self.app.Run()
        else:
            fig._widget.update()
            self.app.ProcessEvents()
            
            img3d = vv.getframe(vv.gcf())
            img3d = np.clip(img3d * 255, 0, 255).astype(np.uint8)
            # Crop gray borders
            img3d = img3d[10:-10, 10:-10,:]
            
            return img3d, img


