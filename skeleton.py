
## utility functions
def edit_driving_data_config(config_path, which_param, new_varval_dict):
    pass

def edit_namelist(nml_path, which_nml, new_varval_dict):
    pass

def create_prior_dists():
    pass

def sample_priors():
    pass

def run_jules():
    pass

def evaluate(jules_res, eval_data):
    pass

def propose_params():
    # bounds-aware
    pass

def update_chain():
    # do accept/reject devision
    
    # update params and chain
    
    pass




## define paths for jules exe and original namelists


## define paths for driving data and evaluation data


## ensure site-specific parameters for each site are correctly loaded
## in the config/namelist files (including data path for each site,
## dates for running, and plant functional type presence / land cover)
edit_driving_data_config()
edit_namelist()

## define randomised folder to work in (to save edited namelists
## and temporary jules output). Might need to edit config files
## or namelists here to edit 
edit_driving_data_config()
edit_namelist()

## define parameters that we wish to infer and find them in the 
## namelists (i.e. create dict of params with correct jules naming)


## define priors over the parameters and note physical bounds
create_prior_dists()    

## sample from priors as initial conditions and output 
## modified namelists
sample_priors()
edit_namelist(nml_path, which_nml, new_varval_dict)

## run jules with initial parameter set at all sites
run_jules()


## calculate loss against evaluation data
evaluate()


## start MCMC loop:

    ## create new parameters set using Metropolis-Hastings proposal
    propose_params()        

    ## run jules with new parameter set at all sites
    run_jules()
    
    ## calculate loss against evaluation data
    evaluate()
    
    ## accept or reject new parameter set and update chain
    update_chain()
    
