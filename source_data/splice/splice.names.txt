
1. Title of Database: Primate splice-junction gene sequences (DNA)
                      with associated imperfect domain theory

2. Sources:
   (a) Creators: 
       - all examples taken from Genbank 64.1 (ftp site: genbank.bio.net)
       - categories "ei" and "ie" include every "split-gene" 
         for primates in Genbank 64.1
       - non-splice examples taken from sequences known not to include
         a splicing site 
   (b) Donor: G. Towell, M. Noordewier, and J. Shavlik, 
              {towell,shavlik}@cs.wisc.edu, noordewi@cs.rutgers.edu
   (c) Date received: 1/1/92

3. Past Usage:
   (a) machine learning:
       	-- M. O. Noordewier and G. G. Towell and J. W. Shavlik, 1991; 
           "Training Knowledge-Based Neural Networks to Recognize Genes in 
           DNA Sequences".  Advances in Neural Information Processing Systems,
           volume 3, Morgan Kaufmann.

	-- G. G. Towell and J. W. Shavlik and M. W. Craven, 1991;  
           "Constructive Induction in Knowledge-Based Neural Networks",  
           In Proceedings of the Eighth International Machine Learning
	   Workshop, Morgan Kaufmann.

        -- G. G. Towell, 1991;
           "Symbolic Knowledge and Neural Networks: Insertion, Refinement, and
           Extraction", PhD Thesis, University of Wisconsin - Madison.

        -- G. G. Towell and J. W. Shavlik, 1992;
           "Interpretation of Artificial Neural Networks: Mapping 
           Knowledge-based Neural Networks into Rules", In Advances in Neural
           Information Processing Systems, volume 4, Morgan Kaufmann.

   (b) attributes predicted: given a position in the middle of a window
       60 DNA sequence elements (called "nucleotides" or "base-pairs"),
       decide if this is a
	a) "intron -> exon" boundary (ie) [These are sometimes called "donors"]
	b) "exon -> intron" boundary (ei) [These are sometimes called "acceptors"]
	c) neither                      (n)
   (c) Results of study indicated that machine learning techniques (neural
       networks, nearest neighbor, contributors' KBANN system) performed as
       well/better than classification based on canonical pattern matching
       (method used in biological literature).

4. Relevant Information Paragraph:
   Problem Description: 
      Splice junctions are points on a DNA sequence at which `superfluous' DNA is
      removed during the process of protein creation in higher organisms.  The
      problem posed in this dataset is to recognize, given a sequence of DNA, the
      boundaries between exons (the parts of the DNA sequence retained after
      splicing) and introns (the parts of the DNA sequence that are spliced
      out). This problem consists of two subtasks: recognizing exon/intron
      boundaries (referred to as EI sites), and recognizing intron/exon boundaries
      (IE sites). (In the biological community, IE borders are referred to
      a ``acceptors'' while EI borders are referred to as ``donors''.)

   This dataset has been developed to help evaluate a "hybrid" learning
   algorithm (KBANN) that uses examples to inductively refine preexisting
   knowledge.  Using a "ten-fold cross-validation" methodology on 1000
   examples randomly selected from the complete set of 3190, the following 
   error rates were produced by various ML algorithms (all experiments
   run at the Univ of Wisconsin, sometimes with local implementations
   of published algorithms). 

                System	       Neither    EI      IE
                ----------     -------  -----   -----
		KBANN    	 4.62    7.56    8.47
		BACKPROP    	 5.29    5.74   10.75
		PEBLS    	 6.86    8.18    7.55
		PERCEPTRON    	 3.99   16.32   17.41
		ID3    		 8.84   10.58   13.99
		COBWEB   	11.80   15.04    9.46
		Near. Neighbor	31.11   11.65    9.09
	     	
   Type of domain: non-numeric, nominal (one of A, G, T, C)
 
5. Number of Instances: 3190

6. Number of Attributes: 62
   -- class (one of n, ei, ie)
   -- instance name
   -- 60 sequential DNA nucleotide positions

7. Attribute information:
   -- Statistics for numeric domains: No numeric features used.
   -- Statistics for non-numeric domains
      -- Frequencies:  Neither	    EI	      IE
                       -------	  ------     -----
		A  	24.984%	  22.153%   20.577%
		G  	25.653%	  31.415%   22.383%
		T  	24.273%	  21.771%   26.445%
		C  	25.077%	  24.561%   30.588%
		D	 0.001%    --        0.002%
		N	 0.010%    0.010%    --
                S        --        --        0.002%
                R        --        --        0.002% 

   Attribute #:  Description:
   ============  ============
             1   One of {n ei ie}, indicating the class.
             2   The instance name.
          3-62   The remaining 60 fields are the sequence, starting at 
                 position -30 and ending at position +30. Each of
                 these fields is almost always filled by one of 
                 {a, g, t, c}. Other characters indicate ambiguity among
                 the standard characters according to the following table:
			character	    meaning
			---------	----------------
			    D		  A or G or T
			    N           A or G or C or T
			    S                C or G
			    R		     A or G

8. Missing Attribute Values: none

9. Class Distribution: 
	EI:       767  (25%)
        IE:       768  (25%)
	Neither: 1655  (50%)

