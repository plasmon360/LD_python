clear all
material='Au'
lambda=[200:5:2000]*1e-9;
[epsilon_Re epsilon_Im N] = LD(lambda,material,'LD'); % change material type here.
data = [lambda; epsilon_Re; epsilon_Im]'
save(strcat('eps_',material,'_ld_bora.dat'),'data','-ascii') 

