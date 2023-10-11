"""
IMU

    Velocity (ENU) : Note velocities are in ENU coordinates, but are converted to ECEF
    				during execution of the Kalman filter.
    - ve
    - vn
    - vu
    
    Velocity STD (ENU)
    - v_sde
    - v_sdn
    - v_sdu
    - v_sden
    - v_sdeu
    - v_sdnu

GNSS
    Position (ECEF) : Note GNSS will normally provide position in ECEF of LLH coordinates
                     The measurements in the data are in ECEF
    - pos_x
    - pos_y
    - pos_z
    
    Position STD (ENU)
    - sde
    - sdn
    - sdu
    - sdne
    - sdeu
    - sdun

X: states:
    - pos_n
    - pos_e
    - pos_d
    - vn
    - ve
    - vd

"""

from numpy.linalg import inv
import numpy as np
import pandas as pd
import pymap3d

class GNSS_filter:		
	def __init__(self, row):
		dt = 5e-2

		self.Nx = 6 # number of states: pos, vel
		self.Nu = 3 # number of input var: vel

		self.X = np.zeros([self.Nx, 1]) # error states: pos_xyz, v_ned, eul, bias_acc, bias_gyro

		# Initial antenna location
		pos_xyz = np.array((row['x'],row['y'],row['z'])).reshape(3,1)
		self.X[0:3] = pos_xyz
		# Initial antenna velocity
		rot, v_xyz = self.rot_vel(row)
		
		self.X[3:6] = v_xyz

	# Process model

		# State transition matrix
		self.F = np.identity(self.Nx)

		self.F[0:3, 3:6] = np.identity(3)*dt

		self.P = np.identity(self.Nx) # Set initial values to 0.25?

		# Process noise matrix
		self.gnss_pos_psd = 3.125e-5
		self.vel_psd = 0.0025

		self.Q = np.zeros([self.Nx, self.Nx])

		self.updateQ(dt)

		# Position measurement noise
		self.R_position = np.identity(3)
		self.R_position[0,0] = row.sdx**2
		self.R_position[1,1] = row.sdy**2
		self.R_position[2,2] = row.sdz**2

		self.R_position[0,1] = np.sign(row.sdxy)*row.sdxy**2
		self.R_position[0,2] = np.sign(row.sdxz)*row.sdxz**2
		self.R_position[1,2] = np.sign(row.sdyz)*row.sdyz**2

		self.R_position[1,0] = self.R_position[0,1]
		self.R_position[2,0] = self.R_position[0,2]
		self.R_position[2,1] = self.R_position[1,2]

		# Velocity measurement noise
		self.R_velocity = np.identity(3) * 4e-8
		
		self.R_velocity[0,1] = 1.5e-9
		self.R_velocity[0,2] = 1.5e-9
		self.R_velocity[1,2] = 1.5e-9

		self.R_velocity[1,0] = self.R_velocity[0,1]
		self.R_velocity[2,0] = self.R_velocity[0,2]
		self.R_velocity[2,1] = self.R_velocity[1,2]

		if 'v_sde' in row:
			self.R_velocity[0,0] = row.v_sde**2
			self.R_velocity[1,1] = row.v_sdn**2
			self.R_velocity[2,2] = row.v_sdu**2

			self.R_velocity[0,1] = row.v_sden
			self.R_velocity[0,2] = row.v_sdeu
			self.R_velocity[1,2] = row.v_sdnu

			self.R_velocity[1,0] = self.R_velocity[0,1]
			self.R_velocity[2,0] = self.R_velocity[0,2]
			self.R_velocity[2,1] = self.R_velocity[1,2]
			
		self.R_velocity = rot@self.R_velocity@rot.transpose()

	def rot_vel(self,row):
		"""
		------------------- Rotate ENU velocity into ECEF velocity --------------------------------
			   dX = |  -sg   -sa*cg  ca*cg | | de |        de = |  -sg       cg      0 | | dX |
			   dY = |   cg   -sa*sg  ca*sg | | dn |  and   dn = |-sa*cg  -sa*sg     ca | | dY |
			   dZ = |    0    ca     sa    | | du |        du = | ca*cg   ca*sg     sa | | dZ |
		-------------------------------------------------------------------------------------------
		"""
		v_enu = np.array((row['vel_e'],row['vel_n'],row['vel_u'])).reshape(3,1)
		lat, lon, alt = pymap3d.ecef2geodetic(self.X[0],self.X[1],self.X[2])
		lat=np.deg2rad(lat)
		lam=np.deg2rad(lon)

		ca = np.cos(lat)
		sa = np.sin(lat)

		cg = np.cos(lam)
		sg = np.sin(lam)

		rot = np.zeros((3,3))
		rot[0,0] = -sg
		rot[0,1] = -sa*cg
		rot[0,2] =  ca*cg
		rot[1,0] =  cg
		rot[1,1] = -sa*sg
		rot[1,2] =  ca*sg
		rot[2,0] =   0
		rot[2,1] =  ca
		rot[2,2] =  sa
		v_xyz = rot@v_enu
	
		return rot, v_xyz

	def updateQ(self,dt):
		# Position estimation noise
		# Initial Q values from Chadwell code 3.125d-5 3.125d-5 3.125d-5 0.0025 0.0025 0.0025, assumes white noise of 2.5 cm over a second
		self.Q[0:3, 0:3] = np.identity(3)*self.gnss_pos_psd*dt

		# Velocity estimation noise (acc psd)
		self.Q[3:6, 3:6] = np.identity(3)*self.vel_psd

	def predict(self, dt): # w is the angular rate vector

		self.F[0:3, 3:6] = np.identity(3)*dt
   
		self.updateQ(dt)

		self.X = self.F@self.X
		self.P = self.F@self.P@self.F.transpose() + self.Q

	def updateVel_cov(self, row):
		rot, v_xyz = self.rot_vel(row)
		
		self.R_velocity[0,0] = row.v_sde**2
		self.R_velocity[1,1] = row.v_sdn**2
		self.R_velocity[2,2] = row.v_sdu**2

		self.R_velocity[0,1] = row.v_sden
		self.R_velocity[0,2] = row.v_sdeu
		self.R_velocity[1,2] = row.v_sdnu
	
		self.R_velocity[1,0] = self.R_velocity[0,1]
		self.R_velocity[2,0] = self.R_velocity[0,2]
		self.R_velocity[2,1] = self.R_velocity[1,2]    
		self.R_velocity = rot@self.R_velocity@rot.transpose()

	def updatePosition(self, row):
# 		pos_ned = row[['pos_n','pos_e','pos_d']].values.reshape(3,1)
		pos_xyz = np.array((row['x'],row['y'],row['z'])).reshape(3,1)

		H = np.zeros([3, self.Nx])
		H[0:3, 0:3] = np.identity(3)

		self.R_position[0,0] = row.sdx**2
		self.R_position[1,1] = row.sdy**2
		self.R_position[2,2] = row.sdz**2

		self.R_position[0,1] = np.sign(row.sdxy)*row.sdxy**2
		self.R_position[0,2] = np.sign(row.sdxz)*row.sdxz**2
		self.R_position[1,2] = np.sign(row.sdyz)*row.sdyz**2

		self.R_position[1,0] = self.R_position[0,1]
		self.R_position[2,0] = self.R_position[0,2]
		self.R_position[2,1] = self.R_position[1,2]

		y = pos_xyz - H @ self.X
		S = H @ self.P @ H.transpose() + self.R_position
		K = (self.P @ H.transpose()) @ inv(S)
		self.X = self.X + K @ y

		I = np.identity(self.Nx)
		self.P = (I - K @ H) @ self.P

	def updateVelocity(self, row):
# 		v_ned = row[['vel_n', 'vel_e', 'vel_d']].values.reshape(3,1)
		rot, v_xyz = self.rot_vel(row)
	
		H = np.zeros([3, self.Nx])
		H[0:3,3:6] = np.identity(3)

		y = v_xyz - H @ self.X
		S = H @ self.P @ H.transpose() + self.R_velocity
		K = (self.P @ H.transpose()) @ inv(S)
		self.X = self.X + K @ y

		I = np.identity(self.Nx)
		self.P = (I - K @ H) @ self.P
	
	def get_states(self):
		return {
				"x": np.ndarray.item(self.X[0]),
				"y": np.ndarray.item(self.X[1]),
				"z": np.ndarray.item(self.X[2]),
				"vx": np.ndarray.item(self.X[3]),
				"vy": np.ndarray.item(self.X[4]),
				"vz": np.ndarray.item(self.X[5]),
		}
	def return_states(self):
		return self.X
	
	def return_covariance(self):
		return self.P

	def rts_smoother(self, Xs, Ps):
	# Rauch, Tung, and Striebel smoother
	
		n, dim_x, _ = Xs.shape

		# smoother gain
		K = np.zeros((n,dim_x, dim_x))
		x, P, Pp = Xs.copy(), Ps.copy(), Ps.copy()

		i = 0
		for k in range(n-2,-1,-1):
			Pp[k] = self.F @ P[k] @ self.F.T + self.Q # predicted covariance

			K[k]  = P[k] @ self.F.T @ inv(Pp[k])
			x[k] += K[k] @ (x[k+1] - (self.F @ x[k]))     
			P[k] += K[k] @ (P[k+1] - Pp[k]) @ K[k].T
			i += 1
			print(str(i+1)+'/'+str(n),end='\r')
		print('')
			
		return x, P, K, Pp

def run_filter_simulation(df):
	import time

	init = True

	results = []
	Xs = np.zeros([len(df),6,1])
	Ps = np.zeros([len(df),6,6])

	all_results = pd.DataFrame()
	
	# Process data through forward Kalman filter
	for i in range(len(df)):
		row = df[i]
		print(str(i+1)+'/'+str(len(df)),end='\r')

		if init:
			gnss_kf = GNSS_filter(row)
			init = False
			last_time = row.dts
		else:
			dt = abs(row.dts - last_time) # This helps to stabilize the solution, abs ensures reverse filtering works.
			gnss_kf.predict(dt)
			last_time = row.dts

			# New velocity standard deviations
			if 'v_sde' in row.dtype.names:
				if not np.isnan(row['v_sde']):
					gnss_kf.updateVel_cov(row)
			# new GNSS measurement
			if not np.isnan(row['x']):
				gnss_kf.updatePosition(row)
			# new velocity measurement
			if not np.isnan(row['vel_e']):
				gnss_kf.updateVelocity(row)

		X = gnss_kf.return_states()
		P = gnss_kf.return_covariance()
		Xs[i] = X
		Ps[i] = P
	
	print('')
	print('Smoothing results')
	x, P, K, Pp = gnss_kf.rts_smoother(Xs,Ps)
	
	return x, P, K, Pp

def main():
	import pymap3d

	vel_df = pd.read_csv('/Volumes/SeaJade 2 Backup/ONC/Data/John_GNSSA/2018/NCL1_2018/NCL1/261/WG_20180918/INS_VEL',
		header=None, sep='\s+', names=['dts','dtype','lat','lon','hae','vel_n','vel_e','vel_u'],
		low_memory=False)
		

	vel_std = pd.read_csv('/Volumes/SeaJade 2 Backup/ONC/Data/John_GNSSA/2018/NCL1_2018/NCL1/261/WG_20180918/COV_VEL',
		header=None, sep='\s+', names=['dts','dtype','v_sde','v_sden','v_sdeu','v_sdne',
		'v_sdn','v_sdnu','v_sdue','v_sdun','v_sdu'],
		low_memory=False)
	
	pos_df = pd.read_csv('/Volumes/SeaJade 2 Backup/ONC/Data/John_GNSSA/2018/NCL1_2018/NCL1/261/GPS_PPP/GPS_POS_FREED',
		header=None, names=['dts','dtype','x','y','z','sdx','sdy','sdz'], sep='\s+',
		low_memory=False)
	pos_df['sdxy'] = np.sqrt(pos_df.sdx * pos_df.sdy)
	pos_df['sdxz'] = np.sqrt(pos_df.sdx * pos_df.sdz)
	pos_df['sdyz'] = np.sqrt(pos_df.sdy * pos_df.sdz)
	
	merged_df = vel_df.merge(pos_df,on='dts',how='left')
	merged_df = merged_df.merge(vel_std,on='dts',how='left')
	merged_df.reset_index(drop=True)
	merged_df.sort_values('dts').reset_index(drop=True)
	first_pos = merged_df[~merged_df.x.isnull()].iloc[0].name
	merged_df = merged_df.loc[first_pos:].reset_index(drop=True)
	
	df = merged_df.to_records()

	x, P, K, Pp = run_filter_simulation(df)

	smoothed_results = pd.DataFrame(x.reshape(x.shape[0],-1),columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])
	lats, lons, alts = pymap3d.ecef2geodetic(smoothed_results.x.values,smoothed_results.y.values,smoothed_results.z.values)
	smoothed_results['lat'] = lats
	smoothed_results['lon'] = lons
	smoothed_results['hae'] = alts
	smoothed_results['dts'] = merged_df[0:len(smoothed_results)].dts

	import matplotlib.pyplot as plt

	fig,ax = plt.subplots()
	smoothed_results.plot('lon','lat',ax=ax,c='green',label='smoothed')
	compare = merged_df[0:len(smoothed_results)][~merged_df[0:len(smoothed_results)].z.isnull()]
	compare.reset_index(drop=True,inplace=True)
	lats, lons, alts = pymap3d.ecef2geodetic(compare.x.values,compare.y.values,compare.z.values)
	compare['lat'] = lats
	compare['lon'] = lons
	compare['hae'] = alts
	compare.plot('lon','lat',ax=ax,c='blue',label='GNSS')
	plt.show()

	fig,ax = plt.subplots()
	smoothed_results.plot('dts','hae',ax=ax,c='green',label='smoothed')
	compare = merged_df[0:len(smoothed_results)][~merged_df[0:len(smoothed_results)].z.isnull()]
	compare.reset_index(drop=True,inplace=True)
	lats, lons, alts = pymap3d.ecef2geodetic(compare.x.values,compare.y.values,compare.z.values)
	compare['lat'] = lats
	compare['lon'] = lons
	compare['hae'] = alts
	compare.plot('dts','hae',ax=ax,c='blue',label='GNSS')
	plt.show()