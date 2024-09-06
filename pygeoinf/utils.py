import numpy as np


def run_randomised_checks(checks, trials, rtol):
    """
    Repeated calls a randomised check and returns true if all cases pass. 

    Args:
        check (callable): A functor that takes in rtol and returns true if 
            a randomised check has been passed. 
        trails (int): Number of checks to perform. 
        rtol (float): relative tolerance used within the check. 
    """        
    for check in checks:
        for _ in range(trials):
            if not check(rtol):
                return False
    return True


