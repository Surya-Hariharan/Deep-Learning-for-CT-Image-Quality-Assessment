"""Prediction Uncertainty (U) -- architecture boundary.

Status: planned. Deliberately unformulated: NGP-Net outputs a warped
image, per-voxel segmentation logits, and a deformation field -- it does
not classify -- so the usual binary-classification entropy formulation of
"uncertainty" does not apply directly. A formulation should be chosen
after the NGP-Net baseline is trained and measured, not before.
"""
