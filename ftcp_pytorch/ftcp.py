import joblib, json
import numpy as np
from functools import partial
from tqdm import tqdm
import joblib, os
import numpy as np
from tqdm import tqdm
from ase.io import write
from ase import spacegroup
from pymatgen.core import Structure
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
tqdm = partial(tqdm, position=0, leave=True)

class FTCPDataSet(Dataset):
    def __init__(self, dataframe, pad_width=2, max_elms=102, max_sites=40, predict_property=False, property_name=None):
        FTCP_representation, self.Nsites = FTCP_represent(dataframe, max_elms=max_elms, max_sites=max_sites, return_Nsites=True)
        FTCP_representation = np.pad(FTCP_representation, ((0, 0), (0, pad_width), (0, 0)), constant_values=0)
        self.data, self.scaler = minmax(FTCP_representation)

        self.predict_property = predict_property

        if predict_property:
            Y = dataframe[[property_name]].values
            scaler_y = MinMaxScaler()
            self.Y = scaler_y.fit_transform(Y)
            print(self.Y.shape)
            assert len(self.Y) == len(self.data)
            self.Y = self.Y.squeeze(-1)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.predict_property:
            prop = self.Y[idx]
        else:
            prop = 0.
        return self.data[idx], prop

def minmax(FTCP):
    '''
    This function performs data normalization for FTCP representation along the second dimension

    Parameters
    ----------
    FTCP : numpy ndarray
        FTCP representation as numpy ndarray.

    Returns
    -------
    FTCP_normed : numpy ndarray
        Normalized FTCP representation.
    scaler : sklearn MinMaxScaler object
        MinMaxScaler used for the normalization.

    '''
    
    dim0, dim1, dim2 = FTCP.shape
    scaler = MinMaxScaler()
    FTCP_ = np.transpose(FTCP, (1, 0, 2))
    FTCP_ = FTCP_.reshape(dim1, dim0*dim2)
    FTCP_ = scaler.fit_transform(FTCP_.T)
    FTCP_ = FTCP_.T
    FTCP_ = FTCP_.reshape(dim1, dim0, dim2)
    FTCP_normed = np.transpose(FTCP_, (1, 0, 2))
    
    return FTCP_normed, scaler


def FTCP_represent(dataframe, max_elms=3, max_sites=20, return_Nsites=False):
    '''
    This function represents crystals in the dataframe to their FTCP representations.

    Parameters
    ----------
    dataframe : pandas dataframe
        Dataframe containing cyrstals to be converted; 
        CIFs need to be included under column 'cif'.
    max_elms : int, optional
        Maximum number of components/elements for crystals in the dataframe. 
        The default is 3.
    max_sites : int, optional
        Maximum number of sites for crystals in the dataframe.
        The default is 20.
    return_Nsites : bool, optional
        Whether to return number of sites to be used in the error calculation
        of reconstructed site coordinate matrix
    
    Returns
    -------
    FTCP : numpy ndarray
        FTCP representation as numpy array for crystals in the dataframe.

    '''
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    # Read string of elements considered in the study
    elm_str = joblib.load('data/element.pkl')
    # Build one-hot vectors for the elements
    elm_onehot = np.arange(1, len(elm_str)+1)[:,np.newaxis]
    elm_onehot = OneHotEncoder().fit_transform(elm_onehot).toarray()
    
    # Read elemental properties from atom_init.json from CGCNN (https://github.com/txie-93/cgcnn)
    with open('data/atom_init.json') as f:
        elm_prop = json.load(f)
    elm_prop = {int(key): value for key, value in elm_prop.items()}
    
    # Initialize FTCP array
    FTCP = []
    if return_Nsites:
        Nsites = []
    # Represent dataframe
    op = tqdm(dataframe.index)
    for idx in op:
        op.set_description('representing data as FTCP ...')
        
        crystal = Structure.from_str(dataframe['cif'][idx],fmt="cif")
        
        # Obtain element matrix
        elm, elm_idx = np.unique(crystal.atomic_numbers, return_index=True)
        # Sort elm to the order of sites in the CIF
        site_elm = np.array(crystal.atomic_numbers)
        elm = site_elm[np.sort(elm_idx)]
        # Zero pad element matrix to have at least 3 columns
        ELM = np.zeros((len(elm_onehot), max(max_elms, 3),))
        ELM[:, :len(elm)] = elm_onehot[elm-1,:].T
        
        # Obtain lattice matrix
        latt = crystal.lattice
        LATT = np.array((latt.abc, latt.angles))
        LATT = np.pad(LATT, ((0, 0), (0, max(max_elms, 3)-LATT.shape[1])), constant_values=0)
        
        # Obtain site coordinate matrix
        SITE_COOR = np.array([site.frac_coords for site in crystal])
        # Pad site coordinate matrix up to max_sites rows and max_elms columns
        SITE_COOR = np.pad(SITE_COOR, ((0, max_sites-SITE_COOR.shape[0]), 
                                       (0, max(max_elms, 3)-SITE_COOR.shape[1])), constant_values=0)
        
        # Obtain site occupancy matrix
        elm_inverse = np.zeros(len(crystal), dtype=int) # Get the indices of elm that can be used to reconstruct site_elm
        for count, e in enumerate(elm):
            elm_inverse[np.argwhere(site_elm == e)] = count
        SITE_OCCU = OneHotEncoder().fit_transform(elm_inverse[:,np.newaxis]).toarray()
        # Zero pad site occupancy matrix to have at least 3 columns, and max_elms rows
        SITE_OCCU = np.pad(SITE_OCCU, ((0, max_sites-SITE_OCCU.shape[0]),
                                       (0, max(max_elms, 3)-SITE_OCCU.shape[1])), constant_values=0)
        
        # Obtain elemental property matrix
        ELM_PROP = np.zeros((len(elm_prop[1]), max(max_elms, 3),))
        ELM_PROP[:, :len(elm)] = np.array([elm_prop[e] for e in elm]).T
        
        # Obtain real-space features; note the zero padding is to cater for the distance of k point in the reciprocal space
        REAL = np.concatenate((ELM, LATT, SITE_COOR, SITE_OCCU, np.zeros((1, max(max_elms, 3))), ELM_PROP), axis=0)
        
        # Obtain FTCP matrix
        recip_latt = latt.reciprocal_lattice_crystallographic
        # First use a smaller radius, if not enough k points, then proceed with a larger radius
        hkl, g_hkl, ind, _ = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 1.297, zip_results=False)
        if len(hkl) < 60:
            hkl, g_hkl, ind, _ = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 1.4, zip_results=False)
        # Drop (000)
        not_zero = g_hkl!=0
        hkl = hkl[not_zero,:]
        g_hkl = g_hkl[not_zero]
        # Convert miller indices to be type int
        hkl = hkl.astype('int16')
        # Sort hkl
        hkl_sum = np.sum(np.abs(hkl),axis=1)
        h = -hkl[:,0]
        k = -hkl[:,1]
        l = -hkl[:,2]
        hkl_idx = np.lexsort((l,k,h,hkl_sum))
        # Take the closest 59 k points (to origin)
        hkl_idx = hkl_idx[:59]
        hkl = hkl[hkl_idx,:]
        g_hkl = g_hkl[hkl_idx]
        # Vectorized computation of (k dot r) for all hkls and fractional coordinates
        k_dot_r = np.einsum('ij,kj->ik', hkl, SITE_COOR[:, :3]) # num_hkl x num_sites
        # Obtain FTCP matrix
        F_hkl = np.matmul(np.pad(ELM_PROP[:,elm_inverse], ((0, 0),
                                                           (0, max_sites-len(elm_inverse))), constant_values=0),
                          np.pi*k_dot_r.T)
        
        # Obtain reciprocal-space features
        RECIP = np.zeros((REAL.shape[0], 59,))
        # Prepend distances of k points to the FTCP matrix in the reciprocal-space features
        RECIP[-ELM_PROP.shape[0]-1, :] = g_hkl
        RECIP[-ELM_PROP.shape[0]:, :] = F_hkl
        
        # Obtain FTCP representation, and add to FTCP array
        FTCP.append(np.concatenate([REAL, RECIP], axis=1))
        
        # print(F_hkl.shape,g_hkl.shape)
        # print(g_hkl.shape, F_hkl.shape)
        if return_Nsites:
            Nsites.append(len(crystal))
    FTCP = np.stack(FTCP)
    
    if not return_Nsites:
        return FTCP
    else:
        return FTCP, np.array(Nsites)



def get_info(ftcp_designs, 
             max_elms=3, 
             max_sites=20, 
             elm_str=joblib.load('data/element.pkl'),
             to_CIF=True,
             check_uniqueness=True,
             mp_api_key=None,
             ):
    
    '''
    This function gets chemical information for designed FTCP representations, 
    i.e., formulas, lattice parameters, site fractional coordinates.
    (decoded sampled latent points/vectors).

    Parameters
    ----------
    ftcp_designs : numpy ndarray
        Designed FTCP representations for decoded sampled latent points/vectors.
        The dimensions of the ndarray are number of designs x latent dimension.
    max_elms : int, optional
        Maximum number of components/elements for designed crystals. 
        The default is 3.
    max_sites : int, optional
        Maximum number of sites for designed crystals.
        The default is 20.
    elm_str : list of element strings, optional
        A list of element strings containing elements considered in the design.
        The default is from "elements.pkl".
    to_CIF : bool, optional
        Whether to output CIFs to "designed_CIFs" folder. The default is true.
    check_uniqueness : bool, optional
        Whether to check the uniqueness of the designed composition is contained in the Materials Project.
    mp_api_key : str, optional
        The API key for Mateirals Project. Required if check_uniqueness is True. 
        The default is None.
    

    Returns
    -------
    pred_formula : list of predicted sites
        List of predicted formulas as lists of predicted sites.
    pred_abc : numpy ndarray
        Predicted lattice constants, abc, of designed crystals; 
        Dimensions are number of designs x 3
    pred_ang : numpy ndarray
        Predicted lattice angles, alpha, beta, and gamma, of designed crystals; 
        Dimensions are number of designs x 3
    pred_latt : numpy ndarray
        Predicted lattice parameters (concatenation of pred_abc and pred_ang);
        Dimensions are number of designs x 6
    pred_site_coor : list
        List of predicted site coordinates, of length number of designs;
        The component site coordinates are in numpy ndarray of number_of_sites x 3
    ind_unique : list
        Index for unique designs. Will only be returned if check_uniqueness is True.
    
    '''
    
    Ntotal_elms = len(elm_str)
    # Get predicted elements of designed crystals
    pred_elm = np.argmax(ftcp_designs[:, :Ntotal_elms, :max_elms], axis=1)
    
    def get_formula(ftcp_designs, ):
        
        # Initialize predicted formulas
        pred_for_array = np.zeros((ftcp_designs.shape[0], max_sites))
        pred_formula = []
        # Get predicted site occupancy of designed crystals
        pred_site_occu = ftcp_designs[:, Ntotal_elms+2+max_sites:Ntotal_elms+2+2*max_sites, :max_elms]
        # Zero non-max values per site in the site occupancy matrix
        temp = np.repeat(np.expand_dims(np.max(pred_site_occu, axis=2), axis=2), max_elms, axis=2)
        pred_site_occu[pred_site_occu < temp]=0
        # Put a threshold to zero empty sites (namely, the sites due to zero padding)
        pred_site_occu[pred_site_occu < 0.05] = 0
        # Ceil the max per site to ones to obtain one-hot vectors
        pred_site_occu = np.ceil(pred_site_occu)
        # Get predicted formulas
        for i in range(len(ftcp_designs)):
            pred_for_array[i] = pred_site_occu[i].dot(pred_elm[i])
            
            if np.all(pred_for_array[i] == 0):
                pred_formula.append([elm_str[0]])
            else:
                temp = pred_for_array[i]
                temp = temp[:np.where(temp>0)[0][-1]+1]
                temp = temp.tolist()
                pred_formula.append([elm_str[int(j)] for j in temp])
        return pred_formula
    
    pred_formula = get_formula(ftcp_designs)
    # Get predicted lattice of designed crystals
    pred_abc = ftcp_designs[:, Ntotal_elms, :3]
    pred_ang = ftcp_designs[:, Ntotal_elms+1,:3]
    pred_latt = np.concatenate((pred_abc, pred_ang), axis=1)
    # Get predicted site coordinates of designed crystals
    pred_site_coor = []
    pred_site_coor_ = ftcp_designs[:, Ntotal_elms+2:Ntotal_elms+2+max_sites, :3]
    for i, c in enumerate(pred_formula):
        Nsites = len(c)
        pred_site_coor.append(pred_site_coor_[i, :Nsites, :])
    

    ind = list(np.arange(len(pred_formula)))
    
    if to_CIF:
        os.makedirs('designed_CIFs', exist_ok=True)
        
        op = tqdm(ind)
        for i, j in enumerate(op):
            op.set_description("Writing designed crystals as CIFs")
            
            try:
                crystal = spacegroup.crystal(pred_formula[j],
                                             basis=pred_site_coor[j],
                                             cellpar=pred_latt[j])
                write('designed_CIFs/'+str(i)+'.cif', crystal)
            except:
                pass
    
    if check_uniqueness:
        ind_unique = ind
        return pred_formula, pred_abc, pred_ang, pred_latt, pred_site_coor, ind_unique
    else:
        return pred_formula, pred_abc, pred_ang, pred_latt, pred_site_coor
    