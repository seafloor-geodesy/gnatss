# GNSS-Acoustic

## Overview
Global Navigation Satellite System - Acoustic (GNSS-A) is a seafloor geodetic tool roughly analogous to GNSS. Standard GNSS is unable to position points on the seafloor because the ocean is opaque to the satellite radar used to communicate with GNSS recievers. To address this, the GNSS-A technique involves deploying two sets of instruments: acoustic transponders that rest on the seafloor and a sea surface platform with a transducer and standard GNSS. The surface platform utilizes acoustics to position the seafloor transponder relative to itself and GNSS to anchor the entire system (seafloor transponder and surface platform) in a terrestrial refence frame such as ITRF.

## Experimental Setup
A GNSS-A site consists of an array of seafloor transponders. These transponders rest on the seafloor in a "listen" mode, replying with an acoustic pulse when they recieve a like pulse with a specific address. This allows a user to measure the two-way travel time (TWTT) between a surface platform and a seafloor transponder. Because the TWTT is sensitive to the speed of sound in seawater, a value that may change due to instantaneous oceanographic sources such as internal gravity waves, a single transponder may not be positioned to much better than ~10 cm due to oceanographic signals. This may be mitigated by deploying acoustic transponders in arrays of 3 or more and averaging the apparent transponder positions in space and time, which allows us to position the geometric center of a transponder array to ~1 cm.

The sea surface platform may be a wave glider, research vessel, or buoy. GNATSS assumes that the surface platform surveys the acoustic transponders from the center of the array and interrogates each transponder simultaneously at regular intervals, typically every 15-20 seconds. Surveying from the array center yields robust array positions but less information about sound speed variations in the water column. Thus, we typically recommend surveying for 3-5 days at a single site and averaging the array position over this time period to mitigate oceanographic signals.

## GNSS-A Derivation
Consider a transponder sitting on the seafloor at some known position $X_1$ and a transducer at the sea surface at some known position $X_G$. Let the line-of-sight distance between the transponder and transducer be denoted by $\vec{D}_1=X_1-X_G$, and assume the water column consists of a single layer with sound velocity $c$. Under such conditions, the one-way travel time of an acoustic pulse a1 between the transponder and the transducer may be written as

$$\frac{||\vec{D}_1||}{c}=a+1,$$

$$\frac{\hat{D}_1}{c} \cdot \vec{D}_1=a_1$$

As a quick unit check, note that $c$ has units of velocity (nominally m/s), $\vec{D}_1$ has units of length (nominally meters), and $\hat{D}_1$ is unitless, so $a_1$ has units of time (nominally s) as expected.

Now let us assume that the seafloor transponder remains stationary but is offset from $X_1$, while holding the transducer position $X_G$ constant. In this case, let $a_{mod}$ be the expected travel time had the acoustic pulse traveled along $\vec{D}_1$ and $a_{meas}$ be the measured travel time. If ameasamod, then the “true” raypath D1t must differ from D1 by offset X1. Assuming that ||X1||<<||D1||, we may approximate D1tD1. This lets us write a simplified equation solving for the travel time residual a1=ameas-amod,

D1cX1=a1.

If we instead consider the two-way travel time, the only difference is that the raypath is multiplied by 2/c instead of 1/c.

Now, let us relax the constraint that XG is constant and allow the transducer at the sea surface to move, defining the transducer position when it sends an acoustic pulse to the transponder as XS and its position when it receives a reply as XR. During this time, the transponder on the seafloor is stationary but may still be offset from its assumed position by X1. In this case the travel time residual is the sum of the residual from the raypath D1S from the initial transducer position to the transponder and the residual from the return raypath D1R from the transponder to the final transducer position:

D1S+D1RcX1=a1S+a1R=a1.

Until now we have only considered a single transponder, but we may generalize this to an array with i transponders. Assuming that the array moves as a block, each transponder has the same offset, so X1=X2==Xi=X. Furthermore, since there may be subtle oceanographic variations across the array, let us assume that the raypath from the transducer to each transponder travels through a different mean sound velocity ci. With this assumption, we can define the variable Pi as

Pi=DiS+DiRci.

Now, we can write a generalized equation relating the change in array position to the travel time residuals:

AX=a,
A=P1; P2;; Pi,
a=a1; a2;; ai.

Note that in the equations above, A is an i3 matrix and a has a length of i.

You will no doubt have noticed at this point that I have very deliberately structured the relationship between the transponder and array offsets and the travel time residuals as a system of linear equations. This is to imply that if we measure the two-way travel time between a sea surface transducer and an array of seafloor transponders, we may infer the offset of the array from an assumed a priori position as long as we have a good estimate of the sound velocity profile across the array and the transducer position when sending and receiving acoustic pulses. This exercise forms the backbone of the GNSS-Acoustic technique, although in practice there are some more steps we must take when interpreting some data.

The observations used in this inversion are not perfect, so it is a good idea to construct a weighting matrix W which depends on the uncertainties of the travel time residuals. The travel time residuals have uncertainties dependent on three sources: the acoustic measurement uncertainty of the transducer, the position uncertainty of the transducer when the acoustic pulse is sent, and the position uncertainty of the transducer when the return acoustic pulse is received. Let a2 be the variance of an acoustic measurement, CS be the 3x3 covariance matrix of the transducer position Xs, and CR be the 3x3 covariance matrix of the transducer position XR. Assuming that a is a stochastic variable, the error propagation to derive the ii matrix W may be written out as:

W=a2I+aXSTCSaXS+aXRTCRaXR.

Note that in the above formulation, the partial derivatives simplify to:

aXS=D1Sc1, D2Sc2,,DiSci,
aXR=D1Rc1, D2Rc2,,DiRci.

As another unit check, the elements in the partial derivatives above all have units of slowness (nominally s/m) and the elements of Cs and CR all have units of length squared (nominally m2), so the entire matrix multiplication will result in a covariance matrix whose elements have units of time squared (nominally s2). Likewise, a2 also has units of time squared so the units of W are consistent.

As one final note for constructing W, it is possible to save some computing time by assuming that XS and XR are close to each other and that CSCR. In this case you only have to compute the above matrix multiplication once and can write W as

W=a2I+2aXSTCSaXS.

Following this, the weighted inversion may be written out as

ATWAX=ATWa.

This inversion may be solved according to the user’s preference, such as with the method of least squares.

One more consideration is that in the real world we do not collect a single travel time measurement for each transponder but rather a time series of measurements during which the seafloor transponders are stationary but the sea surface transducer moves. In this case, we can construct the A, W, and a variables by calculating them at each epoch j as detailed above and then summing them prior to the inversion. Thus,

jAjTWjAjX=jAjTWjaj.

You may also want to construct a pseudo-constraint matrix Q in addition to the above variables to perform a constrained inversion. For instance, you could construct a Q matrix to keep the baselines between transponders constant and force them to resolve the same X.

In addition, the inversion may not always converge immediately. In this case you may iterate the inversion until its solution converges. Let the solution of the kth inversion be Xk. Simply repeat the inversion on subsequent iterations while updating the transponder positions such that 

Xki=Xi+0k-1Xk.

The final transponder offsets will be

X=kXk.

