# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import time

import numpy as np

import torch
import torch.nn as nn

from mesh_viewer import MeshViewer
import utils


@torch.no_grad()
def guess_init(model,
			   joints_2d,
			   edge_idxs,
			   focal_length=5000,
			   pose_embedding=None,
			   vposer=None,
			   use_vposer=True,
			   dtype=torch.float32,
			   model_type='smpl',
			   **kwargs):
	''' Initializes the camera translation vector

		Parameters
		----------
		model: nn.Module
			The PyTorch module of the body
		joints_2d: torch.tensor 1xJx2
			The 2D tensor of the joints
		edge_idxs: list of lists
			A list of pairs, each of which represents a limb used to estimate
			the camera translation
		focal_length: float, optional (default = 5000)
			The focal length of the camera
		pose_embedding: torch.tensor 1x32
			The tensor that contains the embedding of V-Poser that is used to
			generate the pose of the model
		dtype: torch.dtype, optional (torch.float32)
			The floating point type used
		vposer: nn.Module, optional (None)
			The PyTorch module that implements the V-Poser decoder
		Returns
		-------
		init_t: torch.tensor 1x3, dtype = torch.float32
			The vector with the estimated camera location

	'''

	body_pose = vposer.decode(
		pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
	if use_vposer and model_type == 'smpl':
		wrist_pose = torch.zeros([body_pose.shape[0], 6],
								 dtype=body_pose.dtype,
								 device=body_pose.device)
		body_pose = torch.cat([body_pose, wrist_pose], dim=1)

	output = model(body_pose=body_pose, return_verts=False,
				   return_full_pose=False)
	joints_3d = output.joints
	joints_2d = joints_2d.to(device=joints_3d.device)

	diff3d = []
	diff2d = []
	for edge in edge_idxs:
		diff3d.append(joints_3d[:, edge[0]] - joints_3d[:, edge[1]])
		diff2d.append(joints_2d[:, edge[0]] - joints_2d[:, edge[1]])

	diff3d = torch.stack(diff3d, dim=1)
	diff2d = torch.stack(diff2d, dim=1)

	length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
	length_3d = diff3d.pow(2).sum(dim=-1).sqrt()

	height2d = length_2d.mean(dim=1)
	height3d = length_3d.mean(dim=1)

	est_d = focal_length * (height3d / height2d)

	# just set the z value
	batch_size = joints_3d.shape[0]
	x_coord = torch.zeros([batch_size], device=joints_3d.device,
						  dtype=dtype)
	y_coord = x_coord.clone()
	init_t = torch.stack([x_coord, y_coord, est_d], dim=1)
	return init_t


class FittingMonitor(object):
	def __init__(self, summary_steps=1, visualize=False,
				 maxiters=100, ftol=2e-09, gtol=1e-05,
				 body_color=(1.0, 1.0, 0.9, 1.0),
				 model_type='smpl',
				 **kwargs):
		super(FittingMonitor, self).__init__()

		self.maxiters = maxiters
		self.ftol = ftol
		self.gtol = gtol

		self.visualize = visualize
		self.summary_steps = summary_steps
		self.body_color = body_color
		self.model_type = model_type

	def __enter__(self):
		self.steps = 0
		if self.visualize:
			self.mv = MeshViewer(body_color=self.body_color)
		return self

	def __exit__(self, exception_type, exception_value, traceback):
		if self.visualize:
			self.mv.close_viewer()

	def set_colors(self, vertex_color):
		batch_size = self.colors.shape[0]

		self.colors = np.tile(
			np.array(vertex_color).reshape(1, 3),
			[batch_size, 1])

	def run_fitting(self, optimizer, closure, params, body_model,
					use_vposer=True, pose_embedding=None, vposer=None, 
					camera_fitting = False, **kwargs):
		''' Helper function for running an optimization process
			Parameters
			----------
				optimizer: torch.optim.Optimizer
					The PyTorch optimizer object
				closure: function
					The function used to calculate the gradients
				params: list
					List containing the parameters that will be optimized
				body_model: nn.Module
					The body model PyTorch module
				use_vposer: bool
					Flag on whether to use VPoser (default=True).
				pose_embedding: torch.tensor, BxN
					The tensor that contains the latent pose variable.
				vposer: nn.Module
					The VPoser module
			Returns
			-------
				loss: float
				The final loss value
		'''
		append_wrists = self.model_type == 'smpl' and use_vposer
		prev_loss, min_dist = None, None
		for n in range(self.maxiters):
			loss = optimizer.step(closure)
			if type(loss) == tuple:
				loss, min_dist = loss

			if torch.isnan(loss).sum() > 0:
				print('NaN loss value, stopping!')
				break

			if torch.isinf(loss).sum() > 0:
				print('Infinite loss value, stopping!')
				break

			if n > 0 and prev_loss is not None and self.ftol > 0:
				loss_rel_change = utils.rel_change(prev_loss, loss.item())

				if loss_rel_change <= self.ftol:
					break

			if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
					for var in params if var.grad is not None]):
				break

			if self.visualize and n % self.summary_steps == 0:
				body_pose = vposer.decode(
					pose_embedding, output_type='aa').view(
						1, -1) if use_vposer else None

				if append_wrists:
					wrist_pose = torch.zeros([body_pose.shape[0], 6],
											 dtype=body_pose.dtype,
											 device=body_pose.device)
					body_pose = torch.cat([body_pose, wrist_pose], dim=1)
				model_output = body_model(
					return_verts=True, body_pose=body_pose)
				vertices = model_output.vertices.detach().cpu().numpy()

				self.mv.update_mesh(vertices.squeeze(),
									body_model.faces)

			prev_loss = loss.item()

		return prev_loss, min_dist

	def create_fitting_closure(self,
							   optimizer, body_model, camera=None,
							   gt_joints=None, loss=None,
							   joints_conf=None,
							   joint_weights=None,
							   return_verts=True, return_full_pose=False,
							   use_vposer=False, vposer=None,
							   pose_embedding=None,
							   create_graph=False,
							   **kwargs):
		faces_tensor = body_model.faces_tensor.view(-1)
		append_wrists = self.model_type == 'smpl' and use_vposer

		def fitting_func(backward=True):
			if backward:
				optimizer.zero_grad()

			body_pose = vposer.decode(
				pose_embedding, output_type='aa').view(
					1, -1) if use_vposer else None

			if append_wrists:
				wrist_pose = torch.zeros([body_pose.shape[0], 6],
										 dtype=body_pose.dtype,
										 device=body_pose.device)
				body_pose = torch.cat([body_pose, wrist_pose], dim=1)

			body_model_output = body_model(return_verts=return_verts,
										   body_pose=body_pose,
										   return_full_pose=return_full_pose)
			min_dist = None
			total_loss = loss(body_model_output, camera=camera,
						  gt_joints=gt_joints,
						  body_model_faces=faces_tensor,
						  joints_conf=joints_conf,
						  joint_weights=joint_weights,
						  pose_embedding=pose_embedding,
						  use_vposer=use_vposer,
						  **kwargs)
			if type(total_loss) == tuple:
				total_loss, min_dist = total_loss

			if backward:
				total_loss.backward(create_graph=create_graph)

			self.steps += 1
			if self.visualize and self.steps % self.summary_steps == 0:
				model_output = body_model(return_verts=True,
										  body_pose=body_pose)
				vertices = model_output.vertices.detach().cpu().numpy()

				self.mv.update_mesh(vertices.squeeze(),
									body_model.faces)
			return total_loss, min_dist

		return fitting_func


def create_loss(loss_type='smplify', **kwargs):
	if loss_type == 'smplify':
		return SMPLifyLoss(**kwargs)
	elif loss_type == 'camera_init':
		return SMPLifyCameraInitLoss(**kwargs)
	else:
		raise ValueError('Unknown loss type: {}'.format(loss_type))


class SMPLifyLoss(nn.Module):

	def __init__(self, search_tree=None,
				 pen_distance=None, tri_filtering_module=None,
				 rho=100,
				 body_pose_prior=None,
				 shape_prior=None,
				 expr_prior=None,
				 angle_prior=None,
				 jaw_prior=None,
				 use_joints_conf=True,
				 use_face=True, use_hands=True,
				 left_hand_prior=None, right_hand_prior=None,
				 interpenetration=True, dtype=torch.float32,
				 data_weight=1.0,
				 body_pose_weight=0.0,
				 shape_weight=0.0,
				 bending_prior_weight=0.0,
				 hand_prior_weight=0.0,
				 expr_prior_weight=0.0, jaw_prior_weight=0.0,
				 coll_loss_weight=0.0,
				 reduction='sum',
				 use_densepose=False,
				 densepose_i=None,
				 face_idx_to_atlas_i=None,
				 densepose_loss_weight=0.0,
				 **kwargs):

		super(SMPLifyLoss, self).__init__()

		self.use_joints_conf = use_joints_conf
		self.angle_prior = angle_prior

		self.robustifier = utils.GMoF(rho=rho)
		self.rho = rho

		self.body_pose_prior = body_pose_prior

		self.shape_prior = shape_prior

		self.interpenetration = interpenetration
		if self.interpenetration:
			self.search_tree = search_tree
			self.tri_filtering_module = tri_filtering_module
			self.pen_distance = pen_distance

		self.use_hands = use_hands
		if self.use_hands:
			self.left_hand_prior = left_hand_prior
			self.right_hand_prior = right_hand_prior

		self.use_face = use_face
		if self.use_face:
			self.expr_prior = expr_prior
			self.jaw_prior = jaw_prior

		self.use_densepose = use_densepose
		if self.use_densepose:
		  self.densepose_i_arr = densepose_i[0]
		  self.densepose_i_dict = densepose_i[1]
		  self.face_idx_to_atlas_i_arr = face_idx_to_atlas_i[0]
		  self.face_idx_to_atlas_i_dict = face_idx_to_atlas_i[1]

		self.register_buffer('data_weight',
							 torch.tensor(data_weight, dtype=dtype))
		self.register_buffer('body_pose_weight',
							 torch.tensor(body_pose_weight, dtype=dtype))
		self.register_buffer('shape_weight',
							 torch.tensor(shape_weight, dtype=dtype))
		self.register_buffer('bending_prior_weight',
							 torch.tensor(bending_prior_weight, dtype=dtype))
		if self.use_hands:
			self.register_buffer('hand_prior_weight',
								 torch.tensor(hand_prior_weight, dtype=dtype))
		if self.use_face:
			self.register_buffer('expr_prior_weight',
								 torch.tensor(expr_prior_weight, dtype=dtype))
			self.register_buffer('jaw_prior_weight',
								 torch.tensor(jaw_prior_weight, dtype=dtype))
		if self.interpenetration:
			self.register_buffer('coll_loss_weight',
								 torch.tensor(coll_loss_weight, dtype=dtype))
		if self.use_densepose:
			self.register_buffer('densepose_loss_weight',
								 torch.tensor(coll_loss_weight, dtype=dtype))

	def reset_loss_weights(self, loss_weight_dict):
		for key in loss_weight_dict:
			if hasattr(self, key):
				weight_tensor = getattr(self, key)
				if 'torch.Tensor' in str(type(loss_weight_dict[key])):
					weight_tensor = loss_weight_dict[key].clone().detach()
				else:
					weight_tensor = torch.tensor(loss_weight_dict[key],
												 dtype=weight_tensor.dtype,
												 device=weight_tensor.device)
				setattr(self, key, weight_tensor)

	def forward(self, body_model_output, camera, gt_joints, joints_conf,
				body_model_faces, joint_weights,
				use_vposer=False, pose_embedding=None,
				**kwargs):
		#print(body_model_output.vertices.shape, body_model_faces.shape)
		#torch.Size([1, 10475, 3]) torch.Size([62724])

		projected_joints = camera(body_model_output.joints)
		#print(body_model_output.joints.shape, projected_joints.shape)
		#torch.Size([1, 118, 3]) torch.Size([1, 118, 2])

		# Calculate the weights for each joints
		weights = (joint_weights * joints_conf
				   if self.use_joints_conf else
				   joint_weights).unsqueeze(dim=-1)

		# Calculate the distance of the projected joints from
		# the ground truth 2D detections
		joint_diff = self.robustifier(gt_joints - projected_joints)
		joint_loss = (torch.sum(weights ** 2 * joint_diff) *
					  self.data_weight ** 2)

		# Calculate the loss from the Pose prior
		if use_vposer:
			pprior_loss = (pose_embedding.pow(2).sum() *
						   self.body_pose_weight ** 2)
		else:
			pprior_loss = torch.sum(self.body_pose_prior(
				body_model_output.body_pose,
				body_model_output.betas)) * self.body_pose_weight ** 2

		shape_loss = torch.sum(self.shape_prior(
			body_model_output.betas)) * self.shape_weight ** 2
		# Calculate the prior over the joint rotations. This a heuristic used
		# to prevent extreme rotation of the elbows and knees
		body_pose = body_model_output.full_pose[:, 3:66]
		angle_prior_loss = torch.sum(
			self.angle_prior(body_pose)) * self.bending_prior_weight

		# Apply the prior on the pose space of the hand
		left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
		if self.use_hands and self.left_hand_prior is not None:
			left_hand_prior_loss = torch.sum(
				self.left_hand_prior(
					body_model_output.left_hand_pose)) * \
				self.hand_prior_weight ** 2

		if self.use_hands and self.right_hand_prior is not None:
			right_hand_prior_loss = torch.sum(
				self.right_hand_prior(
					body_model_output.right_hand_pose)) * \
				self.hand_prior_weight ** 2

		expression_loss = 0.0
		jaw_prior_loss = 0.0
		if self.use_face:
			expression_loss = torch.sum(self.expr_prior(
				body_model_output.expression)) * \
				self.expr_prior_weight ** 2

			if hasattr(self, 'jaw_prior'):
				jaw_prior_loss = torch.sum(
					self.jaw_prior(
						body_model_output.jaw_pose.mul(
							self.jaw_prior_weight)))

		pen_loss = 0.0
		# Calculate the loss due to interpenetration
		if (self.interpenetration and self.coll_loss_weight.item() > 0):
			batch_size = projected_joints.shape[0]
			triangles = torch.index_select(
				body_model_output.vertices, 1,
				body_model_faces).view(batch_size, -1, 3, 3)
			#print(triangles.shape)
			#torch.Size([1, 20908, 3, 3])

			with torch.no_grad():
				collision_idxs = self.search_tree(triangles)

			# Remove unwanted collisions
			if self.tri_filtering_module is not None:
				collision_idxs = self.tri_filtering_module(collision_idxs)

			if collision_idxs.ge(0).sum().item() > 0:
				pen_loss = torch.sum(
					self.coll_loss_weight *
					self.pen_distance(triangles, collision_idxs))

		densepose_loss = 0.0
		min_dist = None
		if self.use_densepose:
			densepose_loss, min_dist = self.densepose_loss_min_dist(projected_joints.shape[0], body_model_output, 
														body_model_faces, camera)
			densepose_loss *= self.densepose_loss_weight


		total_loss = (joint_loss + pprior_loss + shape_loss +
					  angle_prior_loss + pen_loss +
					  jaw_prior_loss + expression_loss +
					  left_hand_prior_loss + right_hand_prior_loss + 
					  densepose_loss)
		#print('total_loss: {}, joint loss: {}, densepose_loss: {}'\
		#  .format(total_loss.data.cpu().numpy(), joint_loss.data.cpu().numpy(), densepose_loss.data.cpu().numpy()))
		return total_loss, min_dist

	def densepose_loss_min_dist(self, batch_size, body_model_output, body_model_faces, camera):
		densepose_loss = 0.0
		dist_threshold = 1.0
		#batch_size = projected_joints.shape[0]
		faces = torch.index_select(
			  body_model_output.vertices, 1,
			  body_model_faces).view(batch_size, -1, 3, 3)
		face_pts = torch.mean(faces, dim = 2, keepdim = False)
		H, W = self.densepose_i_arr.shape
		projected_face_pts = camera(face_pts)
		#print('projected_face_pixels:', projected_face_pixels.shape)
		projected_face_pixel_x, projected_face_pixel_y = torch.split(projected_face_pts, 1, dim = 2)
		#print('image_shape:', H, W)
		#print('projected_face_pixel_x:', projected_face_pixel_x.shape, projected_face_pixel_x.min(), projected_face_pixel_x.max())
		#print('projected_face_pixel_y:', projected_face_pixel_y.shape, projected_face_pixel_y.min(), projected_face_pixel_y.max())
		projected_face_pixel_x = projected_face_pixel_x.squeeze()
		projected_face_pixel_y = projected_face_pixel_y.squeeze()

		#  #calculate the face_mask
		face_mask = self.face_idx_to_atlas_i_arr != -1
		#face_mask = torch.unsqueeze(face_mask, 0)
		p1 = faces[:, :, 0, :]
		p2 = faces[:, :, 1, :]
		p3 = faces[:, :, 2, :]
		p1p2 = p2 - p1
		p2p3 = p3 - p1
		n = torch.cross(p1p2, p2p3)
		n = torch.div(n, torch.norm(n))
		visibility_mask = n[:, :, -1] < 0
		visibility_mask = torch.squeeze(visibility_mask, 0)
		face_mask = face_mask & visibility_mask
		#print('{} out of {} faces are not masked:'.format(torch.sum(face_mask), len(self.face_idx_to_atlas_i_arr)))

		n_faces = len(self.face_idx_to_atlas_i_arr)
		min_dist = -torch.ones(n_faces, dtype = faces.dtype, device = faces.device)

		for part_id in range(1, 7):
			if part_id != 1:
				continue
			target_pixels = self.densepose_i_dict[part_id]
			#target_pixels = torch.unsqueeze(target_pixels, dim = 0)
			#print('target_pixels:', target_pixels.shape)

			face_ids_per_part = self.face_idx_to_atlas_i_dict[part_id]
			#print('part id:', part_id, 'size of face_ids_per_part:', len(face_ids_per_part))
			face_mask_per_part = torch.nonzero(face_mask[face_ids_per_part])[:, 0]
			face_ids_per_part_masked = face_ids_per_part[face_mask_per_part]
			#print('# of faces of part {} before mask: {}, after mask: {}.'.format(
			#part_id, len(face_ids_per_part), len(face_ids_per_part_masked)))
			#print('face_ids_per_part_masked:', face_ids_per_part_masked.shape)

			predicted_pixel_x = projected_face_pixel_x[face_ids_per_part_masked]
			predicted_pixel_y = projected_face_pixel_y[face_ids_per_part_masked]
			predicted_pixels = torch.stack([predicted_pixel_y, predicted_pixel_x], dim = 1)

			N = target_pixels.shape[0]
			M = predicted_pixels.shape[0]
			#print('M, N:', M, N)
			predicted_pixels_tiled = torch.unsqueeze(predicted_pixels, dim = 1).repeat(1, N, 1) # M*N*2
			#print('predicted_pixels_tiled:', predicted_pixels_tiled.shape)
			target_pixels_tiled = torch.unsqueeze(target_pixels, dim = 1).repeat(1, M, 1) # N*M*2
			#print('target_pixels_tiled:', target_pixels_tiled.shape)
			dist_matrix = predicted_pixels_tiled - torch.transpose(target_pixels_tiled, 0, 1) # M*N*2
			#print('dist_matrix:', dist_matrix.shape)
			ones = torch.ones_like(dist_matrix)
			dist_matrix_min, _ = torch.min(torch.sum(torch.relu(dist_matrix - ones), 2), 1)
			#print('part id:', part_id, dist_matrix_min.min(), torch.mean(dist_matrix_min), dist_matrix_min.max())
			min_dist[face_ids_per_part_masked] = dist_matrix_min
			densepose_loss += torch.mean(dist_matrix_min) * 100.
			
			#predicted_pixels = torch.unsqueeze(predicted_pixels, dim = 0)
			#dist_matrix = torch.cdist(predicted_pixels, target_pixels)
			#print('dist_matrix.shape:', dist_matrix.shape)
			#dist_matrix_min, _ = torch.min(dist_matrix, dim = 1)
			#print('part_id:', part_id, 'dist_matrix_min:', dist_matrix_min.min(), dist_matrix_min.max(), dist_matrix_min.mean())
			#densepose_loss += torch.mean(torch.relu(dist_matrix_min - dist_matrix_min.mean().detach()) ** 2) * 100.0

		#predicted value
		#projected_face_densepose_i = torch.unsqueeze(self.densepose_i[projected_face_pixel_y, \
		#                              projected_face_pixel_x], 0).float()
		#projected_face_densepose_i = torch.unsqueeze(self.face_idx_to_atlas_i, 0).float()
		#print('projected_face_densepose_i:', projected_face_densepose_i.grad_fn)

		#target value
		#face_idx_to_atlas_i = torch.unsqueeze(self.face_idx_to_atlas_i, 0)
		#face_idx_to_atlas_i[face_idx_to_atlas_i == -1] = 0
		#face_idx_to_atlas_i = face_idx_to_atlas_i.float()
		#print('face_idx_to_atlas_i:', face_idx_to_atlas_i.min(), face_idx_to_atlas_i.max())
		#face_mask = face_mask.float()
		#target = torch.ones_like(projected_face_densepose_i)
		#+densepose_loss = torch.mean((projected_face_densepose_i - face_idx_to_atlas_i) ** 2) * 100.
		return densepose_loss, min_dist

class SMPLifyCameraInitLoss(nn.Module):

	def __init__(self, init_joints_idxs, trans_estimation=None,
				 reduction='sum',
				 data_weight=1.0,
				 depth_loss_weight=1e2, dtype=torch.float32,
				 **kwargs):
		super(SMPLifyCameraInitLoss, self).__init__()
		self.dtype = dtype

		if trans_estimation is not None:
			self.register_buffer(
				'trans_estimation',
				utils.to_tensor(trans_estimation, dtype=dtype))
		else:
			self.trans_estimation = trans_estimation

		self.register_buffer('data_weight',
							 torch.tensor(data_weight, dtype=dtype))
		self.register_buffer(
			'init_joints_idxs',
			utils.to_tensor(init_joints_idxs, dtype=torch.long))
		self.register_buffer('depth_loss_weight',
							 torch.tensor(depth_loss_weight, dtype=dtype))

	def reset_loss_weights(self, loss_weight_dict):
		for key in loss_weight_dict:
			if hasattr(self, key):
				weight_tensor = getattr(self, key)
				weight_tensor = torch.tensor(loss_weight_dict[key],
											 dtype=weight_tensor.dtype,
											 device=weight_tensor.device)
				setattr(self, key, weight_tensor)

	def forward(self, body_model_output, camera, gt_joints,
				**kwargs):

		projected_joints = camera(body_model_output.joints)

		joint_error = torch.pow(
			torch.index_select(gt_joints, 1, self.init_joints_idxs) -
			torch.index_select(projected_joints, 1, self.init_joints_idxs),
			2)
		joint_loss = torch.sum(joint_error) * self.data_weight ** 2

		depth_loss = 0.0
		if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
				None):
			depth_loss = self.depth_loss_weight ** 2 * torch.sum((
				camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))

		return joint_loss + depth_loss
