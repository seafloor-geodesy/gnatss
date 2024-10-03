# GNSS-A Pre-processing Overview

The GNSS-Acoustic technique requires three fundamental measurements in order to
position an array of seafloor transponders:

- The position of a transducer mounted to a sea surface platform
- The two-way travel time between the transducer and each seafloor transponder
- The sound velocity profile of the water column

However, when conducting GNSS-Acoustic surveys an extra complication arises
because the transducer position is itself a derived product that must be derived
from the GNSS antenna positions on the surface platform. This operation requires
information about the instantaneous orientation (roll, pitch, and heading) of
the platform as well as the ATD offsets between the GNSS antenna and the
transducer in the local body frame coordinates of the platform, also called the
"lever arms".

Furthermore, there is a discrepancy in the time sampling of these instruments
and the epochs when the transducer receives an acoustic reply since the two-way
travel times are of arbitrary length dependent on the state of the ocean at any
particular moment. GNATSS is tuned to process data collected by the Sonardyne
GNSS-Acoustic payload mounted on an LRI model SV-3 wave glider, as currently
employed in the Near Trench Community Geodetic Experiment. This instrument
collects the following data:

- Pseudorange data at a 10 Hz sampling rate, which the user may choose to
  process at 1 or 10 HZ for GNSS antenna positions
- Wave glider velocities and roll, pitch, heading values at a 20 Hz sampling
  rate
- Acoustic interrogation and rely times. The interrogation times have a sampling
  rate of 1 ping every 15 seconds, and there is one reply for each transponder
  in the array at arbitrary times after interrogation dependent on the ray path
  length.

The posfilter mode of GNATSS will perform the pre-processing required to
generate transducer positions from the raw data collected by the wave glider,
which consists of two steps:

1. Interpolating the positions and orientations to estimate values during
   interrogation and reply epochs
2. Calculating the transducer positions

## Interpolation

The interpolation is performed using one of two methods depending on the data.
Because the roll, pitch, and heading values are collected at 20 Hz, they are
interpolated using a simple spline. However, the interpolated antenna positions
are calculated using a Kalman filter, described below.

The Kalman filter is a tool for predicting the state of a system based upon
previous measurements that is particularly widespread in navigation. GNATSS uses
this algorithm to estimate the antenna position of the surface platform at
arbitrary ping reply epochs based upon the user-processed antenna positions
(assumed to have a sampling rate of 1 Hz) and the INS velocities (assumed to
have a sampling rate of 20 Hz). The Kalman filter consists of two steps, a
forward-filter and back-smoother, and is as follows:

We define the state vector $Y$ as a vector of the position and velocity of the
antenna and $P$ as the covariance matrix of $Y$. We also use subscripts to
distinguish between the _measurement update_ $Y_{(k|k)}$ of the state derived
from a measurement $z_k$ and the _predicted_ value $Y_{(k|k-1)}$ at time $k$,
based upon an estimated state $Y_{(k-1|k-1)}$ at a previous time $k-1$.

The forward filter first uses the previous measurement update, $Y_{(k-1|k-1)}$
and $P_{(k-1|k-1)}$, to predict the state vector and covariance:

$$ Y_{(k|k-1)} = \Phi_{(k|k-1)} \cdot Y_{(k-1|k-1)} $$

$$
P_{(k|k-1)} = \Phi_{(k|k-1)} \cdot P_{(k-1|k-1)} \cdot \Phi^T_{(k|k-1)} +
Q_{(k|k-1)}
$$

In the above equations, $\Phi_{(k|k-1)}$ describes the physics of the system
(assuming the surface platform moves at a constant velocity between times $t_k$
and $t_{k-1}$) and $Q_{(k|k-1)}$ is a noise parameter associated with
$\Phi_{(k|k-1)}$. Based upon this prediction, we calculate the Kalman gain:

$$
K_{(k|k)} = P_{(k|k-1)} \cdot H^T_{(k|k)} \cdot \left( H_{(k|k)} \cdot
P_{(k|k-1)} \cdot H^T_{(k|k)} + V_{(k|k)} \right)^{-1}
$$

$V_{(k|k)}$ is the measurement covariance at time $t_k$ and $H_{(k|k)}$ is a
matrix denoting which elements of $Y$ should be updated based upon new
measurements $z_k$ at time $t_k$. From this, the measurement update of $Y$ and
$P$ is calculated as:

$$
Y_{(k|k)} = Y_{(k|k-1)} + K_{(k|k)} \cdot \left( z_{(k|k)} - H_{(k|k)} \cdot
Y_{(k|k-1)} \right)
$$

$$ P_{(k|k)} = \left( I - K_{(k|k)} \cdot H_{(k|k)} \right) \cdot P_{(k|k-1)} $$

Note that $H_{(k|k)}=0$ during ping reply epochs since no new positions or
velocities are collected at those times. Thus, there is no measurement update at
these times, only a prediction.

The back-smoother is a Rauch-Tung-Striebel smoother (Rauch et al., 1965) that
uses a smoothing gain $A_{(k)}$ to update $Y$ and $P$ based upon future
measurement updates calculated by the forward filter:

$$ A_{(k)} = P_{(k|k)} \cdot \Phi^T_{(k+1|k)} \cdot P^{-1}_{(k+1|k)} $$

$$
Y_{(k|N)} = Y_{(k|k)} + A_{(k)} \cdot \left( Y_{(k+1|N)} - Y_{(k+1|k)}
\right)
$$

$$
P_{(k|N)} = P_{(k|k)} + A_{(k)} \cdot \left( P_{(k+1|N)} - P_{(k+1|k)}
\right) \cdot A_{(k)}^T
$$

## Antenna-Transducer Rotation

The transducer positions $X_{trans}$ must be calculated and depends upon the
roll $r$, pitch $p$, and heading $h$ recorded by the INS, the antenna positions
$X_{ant}$ recorded by the GNSS, and the body frame offsets between the antenna
and transducer on the surface platform. $X_{ant}$ must also be computed by the
user using a kinematic GNSS processing algorithm.

Computing $X_{trans}$ is done in two steps. First, $r$, $p$, $h$, and $X_{ant}$
are interpolated to the times at ping transmit and receive. Because $r$, $p$,
and $h$ are collected at a higher sampling rate than $X_{ant}$, GNATSS employs
different interpolation strategies for these parameters. The orientation
parameters are interpolated with a simple spline. The $X_{ant}$ positions are
interpolated using a Kalman filter with the INS velocities as an additional
constraint.

Once interpolated values are computed for the orientation parameters and antenna
positions, the transducer positions $X_{trans}$ are calculated by adding an
offset to the $X_{ant}$ positions. The method to compute this offset is also
described in Watanabe et al. (2020), and relies upon a body-frame coordinate
system defined for the surface platform. The body-frame axes are oriented toward
the aft, port and vertical (positive-down) of the platform about which the
platforms rotates when imparted a roll, pitch, or heading change. Within this
frame, if we know the offset between the antenna and transducer, $X_{body}$, the
transducer position may be computed by rotating $X_{body}$ about the three
body-frame axes:

$$
X_{trans} = X_{ant} + \mathbf{R}_3(h-360) \mathbf{R}_2(-p) \mathbf{R}_1(-r)
X_{body}
$$

$$
\mathbf{R}_1(\theta) = \left( \begin{matrix}
				\cos{\theta} & 0 & -\sin{\theta} \\
				0 & 1 & 0 \\
				\sin{\theta} & 0 & \cos{\theta} \\
			\end{matrix} \right)
$$

$$
\mathbf{R}_2(\theta) = \left( \begin{matrix}
				1 & 0 & 0 \\
				0 & \cos{\theta} & \sin{\theta} \\
				0 & -\sin{\theta} & \cos{\theta} \\
			\end{matrix} \right)
$$

$$
\mathbf{R}_3(\theta) = \left( \begin{matrix}
				\cos{\theta} & \sin{\theta} & 0 \\
				-\sin{\theta} & \cos{\theta} & 0 \\
				0 & 0 & 1 \\
			\end{matrix} \right)
$$

Note that the rotations use negative $r$, $p$, and $h$ values due to the
orientation of the body-frame axes. For a surface platform such as a wave
glider, $X_{body}$ is constant and may be carefully surveyed before a GNSS-A
deployment.

## References

- Rauch, H. E., Tongue, F., and Striebel, C. T. _Maximum likelihood estimates of
  linear dynamic systems_. AIAA Journal, 3(8):1445–1450, 1965.
  [doi: 10.2514/3.3166](https://doi.org/10.2514/3.3166).

- Watanabe, S. I., Ishikawa, T., Yokota, Y., & Nakamura, Y. (2020). _GARPOS:
  Analysis software for the GNSS‐A seafloor positioning with simultaneous
  estimation of sound speed structure_. Frontiers in Earth Science, 8, 597532.
