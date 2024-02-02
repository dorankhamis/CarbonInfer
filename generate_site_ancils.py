import os
import glob
from jules import Jules

if __name__=="__main__":
    # clean paths
    a_files = glob.glob('./ancils/*')
    [os.remove(af) for af in a_files]
    c_files = glob.glob('./configs/*')
    [os.remove(cf) for cf in c_files]
    
    # re-populate
    j = Jules(prepare_nmls=True)
    
    
    
