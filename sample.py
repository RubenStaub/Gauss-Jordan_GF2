#!/usr/bin/env python3

import numpy as np

def row_pivoting(A, col=0, slicing=slice(None)):
	# Apply row pivoting if first element of column is zero
	if not A[0, col]:
		# Get index of first non zero element along column
		ind = np.argmax(A[slicing, col])
		
		# Check that max pivot is non-zero
		if not A[slicing, col][ind]:
			return(False)
		
		# Swap rows
		A[[0, ind]] = A[[ind, 0]]
	
	return(True)

def xor_nullify_column(sub_A, row, col=0):
	# Get rows needing xor nullifying
	selected_rows = sub_A[:, col]
	
	# Apply xor row to selected rows
	sub_A[selected_rows] = np.logical_xor(sub_A[selected_rows], row)

def FG_partial_pivot(A):
	"""Forward elimination step of Gauss-Jordan elimination in GF(2)
	
	Parameters:
		A (array): Matrix to apply forward elimination.
	
	Return:
		A_row_ech (boolean array): A in row echelon form.
	"""
	# Convert to boolean
	full_A = np.array(A, dtype=bool)
	
	# Apply recursion until submatrix is empty or has single row
	sub_A = A_row_ech
	while sub_A.size > 0 and sub_A.shape[0] > 1:
		# Partial pivoting (i.e. rows only)
		pivot_success = row_pivoting(sub_A)
		
		# If best pivot is zero, retry on right submatrix
		if not pivot_success:
			sub_A = sub_A[:, 1:]
			continue
		
		# Nullify column elements below using elemental row operations
		xor_nullify_column(sub_A[1:], sub_A[0])
		
		# Move to bottom-right submatrix
		sub_A = sub_A[1:, 1:]
	
	return(A_row_ech)

def BG(A):
	"""Backward substitution part of Gauss-Jordan elimination in GF(2)
	
	Parameters:
		A (array): Matrix to apply backward substitution, in row-echelon form.
	
	Return:
		A_reduced (boolean array): A in reduced row echelon form.
	"""
	# Convert to boolean
	A_reduced = np.array(A, dtype=bool)
	
	# Apply recursion until submatrix is empty or has single row
	sub_A = A_reduced
	while sub_A.size > 0 and sub_A.shape[0] > 1:
		# Get first non-zero pivot on row
		ind = np.argmax(sub_A[-1])
		
		# Nullify column elements above using elemental row operations
		xor_nullify_column(sub_A[:-1], sub_A[-1], col=ind)
		
		assert(np.count_nonzero(sub_A[:-1, ind]) == 0)
		
		# Move to top submatrix
		sub_A = sub_A[:-1]
	
	return(A_reduced)

if __name__ == '__main__':
	#
	# Testing on example from:
	# Jun Kurihara, Shinsaku Kiyomoto, Kazuhide Fukushima, and Toshiaki Tanaka
	# A New (k, n)-Threshold Secret Sharing Scheme and Its Extension
	# (https://eprint.iacr.org/2008/409.pdf)
	#
	
	# Define test matrix G
	I = np.eye(4, dtype=bool)
	J = np.eye(5, k=-1, dtype=bool)[1:]
	K = np.eye(4, k=-1, dtype=bool)
	G = np.block([[I, J, J, K],
	              [I, np.roll(J, 1, axis=1), np.roll(J, 2, axis=1), np.roll(K, -1)],
	              [I, np.roll(J, 2, axis=1), np.roll(J, 4, axis=1), np.roll(K, -2)],
	              [I, np.roll(J, 4, axis=1), np.roll(J, 8, axis=1), I]])
	
	# Augmented matrix
	G_aug = np.hstack((G, np.eye(16, dtype=bool)))
	
	# Forward Gaussian elimination
	full_FG = FG_partial_pivot(G_aug)
	
	# Keep last rows
	FG0 = full_FG[-4:, 14:]
	
	# Backward Gaussian elimination
	M_aug = BG(FG0)
	
	# Augmented only
	M = M_aug[:, 4:]


