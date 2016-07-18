v1.2.2
======

 * All encoders handle missing values and are tested for their handling
 
v1.2.1
======

 * Better handling for missing values in hashing encoder
 
v1.2.0
======

 * Testing enhancements
 * Hash type in hashing encoder now defaults to md5 using hashlib, but can be set to any valid hashlib hash

v1.1.2
======

 * Added optional parameter to return a numpy array rather than a dataframe from all transformers.
 
v1.1.1
======

 * Immediately return if cols is empty.
 

v1.1.0
======

 * Optionally pass drop_invariant to any encoder to consistently drop columns with 0 variance from the output (based on training set data in fit())
 * If None is passed as the cols param, every string column will be encoded (pandas type = object).
 
v1.0.5
======

 * Changed setup.py to not explicitly force reinstalls of other packages
 
v1.0.4
======

 * Bugfixes
 
v1.0.0
======

 * First real usable release, includes sklearn compatible encoders.
 
v0.0.1
======

 * Basic library of encoders, no automated testing.