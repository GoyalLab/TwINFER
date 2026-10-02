1. Check gene list choice again. 
2. Check score on simulations.
    2.1 Remove twins that are not similar - also are cells of the same cell type identifying the same gene-pairs
3. Figures
    
    4.   remake figures 2/3/4 with z-scores and 1000 simulations.
        
    5.  benchmark figures - network_sweep, mixed_networks, real_network_sims
4. Test this on LARRY high-correlation/and others
    Two filters come first: |z_rho| above 2.326 at either timepoint, and z_reg_gated above 2.326 at either timepoint. Otherwise, score is -inf (bottom n).
    The score is then z(|rho(t1)|) + z(|rho(t2)|) + s(z_flux), with z_flux = reg(x) − reg(y). 
    Three hinge penalties follow: minus I(z_stable > 2.326)·z_stable, plus I(max(z_het) < −2.326)·max(z_het), and minus I(|z_dagger| > 2.326)|z_dagger| #COMMENT check this, and gamma.
    z_dagger is the z-score of ρ†, the cross-time correlation between clone-mates. 
    z_flux is the difference in the number of targets between the two genes.
5. Perturbation papers - there are gene lists + 5x (compare to Collectri and alternative benchmark dataset)
6. (LARRY, CellTag), (FateMap, SpaceBar).
   
    7. Fatemap 08 - primary melanocyte
    8. Fatemap 06 - untreated tumor - matched with SpaceBar


9. Michaels et al has the 14 day da