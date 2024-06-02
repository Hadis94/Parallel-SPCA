import pandas as pd
import numpy as np
from mpi4py import MPI
import random

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ensure that we have 6 processes running
if size != 6:
    if rank == 0:
        print("This program needs to be run with 6 processes.")
    exit()

# Process 0 will create the matrix, calculate its covariance, adjust it, distribute the rows, gather the results, and sort them, and ...
if rank == 0:
    # Create a matrix with 85% zeros and 20% random numbers between 0 and 1
    matrix = np.zeros((5000, 5000))
    num_non_zero = int(0.5 * matrix.size)
    indices = random.sample(range(matrix.size), num_non_zero)
    for idx in indices:
        matrix[np.unravel_index(idx, matrix.shape)] = random.random()
    
    # Calculate the covariance matrix
    covariance_matrix = np.cov(matrix, rowvar=False)
    #print(covariance_matrix)
    
    sigma_squared = 0.01
    covariance_matrix += sigma_squared * np.eye(covariance_matrix.shape[0])
    
    # Adjust covariance matrix: set elements with absolute value less than 0.01 to 0
    covariance_matrix[np.abs(covariance_matrix) < 0.000995] = 0
    #print("covariance_matrix", covariance_matrix)
    
    # Calculate the sparsity percentage of the covariance matrix
    sparsity_percentage = 100 * np.sum(covariance_matrix == 0) / covariance_matrix.size
    print(f"The covariance matrix is {sparsity_percentage:.2f}% sparse.")
    #print("covariance_matrix.size=", covariance_matrix.size)
    
    # Create a random vector v with dimensions equal to the number of columns in the covariance matrix
    v = np.random.random(covariance_matrix.shape[1])
    
    # number of elements to keep in each iteration
    k = 3500
    
    # 1rst PC by covariance method
    #eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)    
    #sorted_indices_evs = np.argsort(eigenvalues)[::-1]
    #sorted_eigenvalues = eigenvalues[sorted_indices_evs]
    #sorted_eigenvectors = eigenvectors[:, sorted_indices_evs]
    #first_principal_component_covmethod = sorted_eigenvectors[:, 0]
    #print("First Principal Component by Covariance Method:")
    #print(first_principal_component_covmethod)
    #print("\n")
    # Keep only top k elements and set others to zero
    #if k < first_principal_component_covmethod.size:
        # Keep only the top k elements in the first_principal_component_covmethod vector and set the rest to 0
        #k_largest_indices = np.argsort(-np.abs(first_principal_component_covmethod))[:k]
        #print("k_largest_indices", k_largest_indices)
        #result_vector = np.zeros_like(first_principal_component_covmethod)
        #result_vector[k_largest_indices] = first_principal_component_covmethod[k_largest_indices]
        # Print the result_vector containing only the top k elements
        #print("Top k elements of first_principal_component_covmethod:")
        #print(result_vector)
        #print('\n')
        
        
    # 1rst PC by SVD method
    start_time = MPI.Wtime()
    # Perform Singular Value Decomposition (SVD)
    u, s, vh = np.linalg.svd(covariance_matrix, full_matrices=False)
    # First principal component (PC1)
    first_principal_component_svd = u[:, 0]
    #print("First Principal Component by SVD Method:")
    #print(first_principal_component_svd)
    end_time = MPI.Wtime()
    print(f"Total time taken by svd: {end_time - start_time} seconds")
    print("\n")
    # Keep only top k elements and set others to zero
    if k < first_principal_component_svd.size:
        # Keep only the top k elements in the first_principal_component_svd vector and set the rest to 0
        k_largest_indices_svd = np.argsort(-np.abs(first_principal_component_svd))[:k]
        #print("k_largest_indices_svd", k_largest_indices_svd)
        result_vector_svd = np.zeros_like(first_principal_component_svd)
        result_vector_svd[k_largest_indices_svd] = first_principal_component_svd[k_largest_indices_svd]
        #Print the result_vector_svd containing only the top k elements
        #print("Top k elements of first_principal_component_svd:")
        #print(result_vector_svd)
        #print("\n") 
        
        
    
    # Send slices of the adjusted covariance matrix to other processes
    rows_per_process = covariance_matrix.shape[0] // (size - 1)
    for i in range(1, size):
        start_row = (i - 1) * rows_per_process
        end_row = start_row + rows_per_process if i < size - 1 else covariance_matrix.shape[0]
        comm.send(covariance_matrix[start_row:end_row], dest=i, tag=11)

    # Initialize an array to store the non-zero counts from all processes
    total_non_zero_counts = np.zeros((covariance_matrix.shape[0], 2), dtype=int)

    # Receive non-zero counts from other processes and place them in the total array
    for i in range(1, size):
        received_non_zero_counts = comm.recv(source=i, tag=i)
        start_row = (i - 1) * rows_per_process
        end_row = start_row + len(received_non_zero_counts)
        total_non_zero_counts[start_row:end_row, :] = received_non_zero_counts
    #print("Receive non-zero counts from other processes and place them in the total array", total_non_zero_counts)    

    # Sort the rows based on non-zero counts in ascending order and keep the sorted indices
    sorted_indices = np.argsort(total_non_zero_counts[:, 1])
    #print("sorted indices", sorted_indices)

    # Calculate the indices for the distribution based on sorted_indices
    chunk_size = covariance_matrix.shape[0] // 10
    for i in range(1, size):
        if i == 1:
            indices_for_core = np.concatenate((
                sorted_indices[:chunk_size], 
                sorted_indices[-chunk_size:]
            ))
        else:
            indices_for_core = np.concatenate((
                sorted_indices[(i - 1) * chunk_size : i * chunk_size], 
                sorted_indices[-i * chunk_size : -(i - 1) * chunk_size]
            ))
        rows_to_send = covariance_matrix[indices_for_core, :]
        comm.send((rows_to_send, indices_for_core), dest=i, tag=12)
        #print(i, indices_for_core)
        
        # Send vector v to each process
        comm.send(v, dest=i, tag=14)    
        
    # Initialize an array to store the final_result
    final_result = np.zeros(covariance_matrix.shape[0])

    # Gather the results from all processes
    for i in range(1, size):
        received_result_with_indices = comm.recv(source=i, tag=i)
        for index, value in received_result_with_indices:
            final_result[int(index)] = value
    
    #print("Final result of distributed:")
    #print(final_result)
    #print("\n")   
    
    # Keep only the top k elements in the final_result vector and set the rest to 0
    k_largest_indices_final_result = np.argsort(-np.abs(final_result))[:k]
    result_vector_final_result = np.zeros_like(final_result)
    result_vector_final_result[k_largest_indices_final_result] = final_result[k_largest_indices_final_result]
    norm_final_result = np.linalg.norm(result_vector_final_result, ord=2)
    final_result = result_vector_final_result / norm_final_result
    #print("final_result=", final_result)
    
    
    start_time = MPI.Wtime()
    result = np.dot(covariance_matrix, final_result)
    #print("result of dot =", result)
    end_time = MPI.Wtime()
    #print(f"Total time taken by dot: {end_time - start_time} seconds")
    
    
    
    start_time = MPI.Wtime()
    iterations = 100
    iteration_count = 0  
    while iteration_count < iterations:
        #Send final_result to all cores
        for i in range(1, size):
            comm.send(final_result, dest=i, tag=15)
        #Gather results from all cores with indices
        collected_result = np.zeros_like(final_result)
        for i in range(1, size):
            received_result_with_indices = comm.recv(source=i, tag=i)
            for index, value in received_result_with_indices:
                collected_result[int(index)] += value
        final_result = collected_result
        #print(final_result)
        if k < final_result.size:
            k_largest_indices_final_result = np.argsort(-np.abs(final_result))[:k]
            result_vector_final_result = np.zeros_like(final_result)
            result_vector_final_result[k_largest_indices_final_result] = final_result[k_largest_indices_final_result]
            norm_final_result = np.linalg.norm(result_vector_final_result, ord=2)
            final_result = result_vector_final_result / norm_final_result
        else:
            print("Warning: k is larger than the size of the result array.")
        iteration_count += 1
    end_time = MPI.Wtime()
    print(f"Total time taken by power method: {end_time - start_time} seconds")
    #print("Final result after 100 iterations by distributed version:")
    #print(final_result)
     
    
    
    final_result_normalized = final_result / np.linalg.norm(final_result, 2)
    #print("final_result=", final_result)
    #print("final_result_normalized=", final_result_normalized)
    #print("norm_final_result=", np.linalg.norm(final_result, 2))
    result_vector_svd_normalized = first_principal_component_svd / np.linalg.norm(first_principal_component_svd, 2)
    #print("first_principal_component_svd", first_principal_component_svd)
    #print("result_vector_svd_normalized=", result_vector_svd_normalized)
    #print("norm_first_principal_component_svd=", np.linalg.norm(first_principal_component_svd, 2))
    cosine_similarity = np.dot(final_result_normalized, result_vector_svd_normalized)
    print(f"Cosine Similarity between final_result and result_vector_svd: {cosine_similarity}")
    angle_radians = np.arccos(cosine_similarity)
    angle_degrees = np.degrees(angle_radians)
    print(f"Angle between vectors in degrees: {angle_degrees}")
    
    
    
# Other processes will receive their respective slices of the adjusted covariance matrix and print them
else:
    # Receive the slice of the covariance matrix
    received_matrix = comm.recv(source=0, tag=11)
    #print(rank, received_matrix)

    # Count the number of non-zero elements in each row
    non_zero_counts = np.count_nonzero(received_matrix, axis=1)
    #print(rank, non_zero_counts)

    # Store the non-zero counts in an array with indices corresponding to row indices
    original_row_indices = np.array(range((rank - 1) * len(received_matrix), rank * len(received_matrix)))
    non_zero_counts_with_indices = np.array(list(zip(original_row_indices, non_zero_counts)))

    # Send the non-zero counts back to process 0
    comm.send(non_zero_counts_with_indices, dest=0, tag=rank)

    # Receive for the second time part of the covariance matrix and the original indices based on sorted indices
    received_data = comm.recv(source=0, tag=12)
    received_matrix, row_indices = received_data
    #print(rank, row_indices)
    
    # Receive vector v
    v = comm.recv(source=0, tag=14) 
    #print(v)
        
    # Print the received rows and their indices
    #print(f"Core {rank} received rows with indices: {received_data}")   
    
    # Initialize an array to store the result of matrix-vector multiplication
    result_vector = np.zeros(received_matrix.shape[0])
    
    # Store non-zero indices for each row
    non_zero_indices_list = [np.nonzero(row)[0] for row in received_matrix]
   
    # Perform the matrix-vector multiplication using non-zero indices
    for i, non_zero_indices in enumerate(non_zero_indices_list):
        result_vector[i] = np.sum(received_matrix[i, non_zero_indices] * v[non_zero_indices])
        
    result_with_indices = np.array(list(zip(row_indices, result_vector)))
    #print(result_with_indices)
    comm.send(result_with_indices, dest=0, tag=rank)
    
    iterations = 100
    iteration_counter = 0
    for _ in range(iterations):
        final_result = comm.recv(source=0, tag=15)
        #print("after recieve= ", final_result)
        common_non_zero_indices_all_rows = []
        for i, row_non_zero_indices in enumerate(non_zero_indices_list):
            common_non_zero_indices = np.intersect1d(row_non_zero_indices, np.nonzero(final_result)[0])
            common_non_zero_indices_all_rows.append(common_non_zero_indices)
    
        result_vector = np.zeros(received_matrix.shape[0])
        
        if iteration_counter == 0:
            start_time = MPI.Wtime()
    
        # Now perform the multiplication using the pre-computed common non-zero indices
        for i, common_non_zero_indices in enumerate(common_non_zero_indices_all_rows):
            # Ensure there are non-zero indices to avoid unnecessary computation
            if len(common_non_zero_indices) > 0: 
                result_vector[i] = np.sum(received_matrix[i, common_non_zero_indices] * final_result[common_non_zero_indices])
        end_time = MPI.Wtime()
        if iteration_counter == 0:
             #print(f"Total time taken by multiplication in core {rank}: {end_time - start_time} seconds")
             iteration_counter += 1

        result_with_indices = np.array(list(zip(row_indices, result_vector)))
        #print("result =",  result_with_indices)
        comm.send(result_with_indices, dest=0, tag=rank)

        
# Finalize the MPI environment
MPI.Finalize()
