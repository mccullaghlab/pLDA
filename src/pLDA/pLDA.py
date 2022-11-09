import numpy as np
import torch
from shapeGMMTorch import torch_align
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')
import random

# class
class pLDA:
    """
    pLDA is a class that can be used to perform Linear Discriminant Analysis in size-and-shape space.
    The class is designed to mimic similar methods implemented in sklearn.  The model is first initialized
    and then fit with supplied data.  Fit parameters for the model include average structures and (co)variances.
    Once fit, the model can be used to predict clustering on an alternative (but same feature space size) data set.
    Author: Martin McCullagh
    Date: 11/8/2022
    """
    def __init__(self, n_clusters, rot_type="uniform", kabsch_thresh=1E-1, dtype=torch.float32, device=torch.device("cpu")):
        """
        Initialize size-and-shape GMM.
        rot_type                - string defining the type of rotational alignment to use.  Options are 'kronecker' and 'uniform'.  Defualt is 'uniform'.
        kabsch_thresh           - float dictating convergence criteria for each alignment step.  Default value is 1e-1.
        dtype                   - Data type to be used.  Default is torch.float32.
        device                  - device to be used.  Default is torch.device('cuda:0') device.
        verbose                 - boolean dictating whether to print various things at every step. Defulat is False.
        """
        
        self.rot_type = rot_type                                # string
        self.kabsch_thresh = kabsch_thresh                      # float
        self.dtype = dtype                                      # torch dtype
        self.device = device                                    # torch device
        self.verbose = verbose                                  # boolean
        self._plda_fit_flag = False                              # boolean tracking if GMM has been fit.

    # fit the model
    def fit(self, traj_data, cluster_ids):
        """
        Fit pLDA using traj_data and cluster_ids as the training data.
        traj_data (required)   - (n_frames, n_atoms, 3) float32 or float64 numpy array of particle positions. 
        cluster_ids               - (n_frames) integer numpy array of initial cluster assignments.  
        """

        # pass trajectory data to device
        traj_tensor = torch.tensor(traj_data,dtype=self.dtype,device=self.device)
        # make sure trajectory is centered
        torch_align.torch_remove_center_of_geometry(traj_tensor,dtype=self.dtype,device=self.device)
        # perform global alignment
        if self.covar_type == 'uniform':
            aligned_traj_tensor, center_tensor, var_tensor = torch_align.torch_iterative_align_uniform(traj_tensor,thresh=self.kabsch_thresh,device=self.device,dtype=self.dtype)
            self.var = var_tensor.cpu().numpy()
            del var_tensor
        else:
            aligned_traj_tensor, centers_tensor, precisions_tensor, lpdet_tensor = torch_align.torch_iterative_align_kronecker(traj_tensor,thresh=self.kabsch_thresh,device=self.device,dtype=self.dtype)
            self.precision = precision_tensor.cpu().numpy() 
            del precision_tensor
        self.center = center_tensor.cpu().numpy()
        aligned_traj = aligned_traj_tensor.cpu().numpy()
        del traj_tensor
        del centers_tensor
        del aligned_traj_tensor
        torch.cuda.empty_cache()

        # perform LDA using sklearn package
        self.lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
        self.lda_projection = self.lda.fit_transform(aligned_traj.reshape(aligned_traj.shape[0],aligned_traj.shape[1]*aligned_traj.shape[2]), cluster_ids)
        self.lda_vecs = self.lda.scalings_

        

    # predict clustering of provided data based on prefit parameters from fit_weighted
    def transform(self,traj_data):
        """
        Transform/project traj_data into Linear Discriminant vectors

        traj_data (required)   - (n_frames, n_atoms, 3) float32 or float64 numpy array of particle positions. 

        Returns:
        lda_projection             - (n_frames,n_clusters-1) float array
        """

        if self._gmm_fit_flag == True:
            # send data to device
            traj_tensor = torch.tensor(traj_data,dtype=self.dtype,device=self.device)
            center_tensor = torch.tensor(self.center,dtype=self.dtype,device=self.device)
            # kronecker specific variables
            if self.rot_type == 'kronecker':
                # declare precision matrix (inverse covariances)
                precision_tensor = torch.tensor(self.precision,dtype=torch.float64,device=self.device)
            # make sure trajectory is centered
            torch_align.torch_remove_center_of_geometry(traj_tensor,dtype=self.dtype,device=self.device)
            # perform alignment to global average
            if self.covar_type == 'uniform':
                traj_tensor = torch_align.torch_align_uniform(traj_tensor,center_tensor)
            else:
                traj_tensor = torch_align.torch_align_kronecker(traj_tensor,center_tensor,precision_tensor,device=self.device,dtype=self.dtype)
            traj_data = traj_tensor.cpu().numpy() 
            lda_projection = self.lda.transform(traj_data.reshape(traj_data.shape[0],traj_data.shape[1]*traj_data.shape[2]))
            del traj_tensor
            del center_tensor
            if self.rot_type == 'kronecker':
                del precision_tensor 
            torch.cuda.empty_cache()
            # return values
            return lda_projection
        else:
            print("pLDA must be fit before it can predict.")

