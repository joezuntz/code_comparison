import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from cosmosis.datablock import option_section
import os

def setup(options):
	outdir = options[option_section, 'outdir']
	try:
		os.mkdir(outdir)
	except:
		pass
	return [outdir]


def execute(block, config):
	outdir, = config
	# Growth factor at z = 0,1,2,3,4,5
	d = block['growth_parameters', 'd_z']
	z = block['growth_parameters', 'z']
	np.savetxt(outdir+"/growth.txt", np.transpose([z,d]))

	#b) Comoving radial distance [Mpc/h] at z = 0,1,2,3,4,
	z = block['distances', 'z'][::-1]
	d_m = block['distances', 'd_m'][::-1]
	h0 = block['cosmological_parameters', 'h0']
	d_m *= h0
	z_sample = [0., 1., 2., 3., 4.]
	d_m = interp1d(z, d_m, kind='cubic')(z_sample)
	np.savetxt(outdir+"/comoving_distance.txt", np.transpose([z_sample,d_m]))

	#c) Linear matter power spectrum [(Mpc/h)^3] 
	#at z = 0,1,2,3; 
	#1.e-3 h/Mpc <= k <= 10, 10 bins/decade

	k=block["matter_power_lin", "k_h"]
	z=block["matter_power_lin", "z"]
	P=block["matter_power_lin", "P_k"]
	R=RectBivariateSpline(k,z,P.T)
	z_sample = [0.0, 1.0, 2.0, 3.0]
	k_sample = np.logspace(-3, 1, 40)
	z_out = []
	k_out = []
	p_out = []
	for z in z_sample:
		for k in k_sample:
			p_out.append([z, k, R(k, z)])

	np.savetxt(outdir+"/linear_power.txt", p_out)

	#d) Halo mass function at \Delta = 200 \bar{\rho}_m: n(M)*M^2/\bar{\rho}_m 
	#at z = 0,1,2,3; 1.e+10 M_sun/h <= M <= 1.e+16 M_sun/h, 10 bins/decade

	#Do not have n(M), just dN/d(log M)
	M=block["mass_function", "m_h"]
	z=block["mass_function", "z"]
	N=block["mass_function", "dndlnmh"]
	R=RectBivariateSpline(M,z,N.T)

	z_sample = [0.0, 1.0, 2.0, 3.0]
	m_sample = np.logspace(10., 16., 60)
	z_out = []
	k_out = []
	n_out = []
	for z in z_sample:
		for m in m_sample:
			n_out.append([z, m, R(m, z)])

	np.savetxt(outdir+"/mass_function.txt", n_out)

	return 0
