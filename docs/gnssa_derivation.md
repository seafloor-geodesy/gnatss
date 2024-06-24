## GNSS-A Positioning Derivation

Consider a transponder sitting on the seafloor at some known position $X_1$ and
a transducer at the sea surface at some known position $X_G$. Let the
line-of-sight distance between the transponder and transducer be denoted by
$\vec{D}_1=X_1-X_G$, and assume the water column consists of a single layer with
sound velocity $c$. Under such conditions, the one-way travel time of an
acoustic pulse a1 between the transponder and the transducer may be written as

$$\frac{||\vec{D}_1||}{c}=a+1,$$

$$\frac{\hat{D}_1}{c} \cdot \vec{D}_1=a_1$$

As a quick unit check, note that $c$ has units of velocity (nominally m/s),
$\vec{D}_1$ has units of length (nominally meters), and $\hat{D}_1$ is unitless,
so $a_1$ has units of time (nominally s) as expected.

Now let us assume that the seafloor transponder remains stationary but is offset
from $X_1$, while holding the transducer position $X_G$ constant. In this case,
let $a_{mod}$ be the expected travel time had the acoustic pulse traveled along
$\vec{D}_1$ and $a_{meas}$ be the measured travel time. If
$a_{meas} \neq a_{mod}$, then the “true” raypath $\vec{D}_{1t}$ must differ from
$\vec{D}_1$ by offset $\Delta X_1$. Assuming that
$||\Delta X_1|| << ||\vec{D}_1||,$ we may approximate
$\hat{D}_{1t} \approx \hat{D}_1$. This lets us write a simplified equation
solving for the travel time residual $\Delta a_1 = a_{meas}-a_{mod}$,

$$\frac{\hat{D}_1}{c} \cdot \Delta X_1 = \Delta a_1$$

If we instead consider the two-way travel time, the only difference is that the
raypath is multiplied by $2/c$ instead of $1/c$.

Now, let us relax the constraint that $X_G$ is constant and allow the transducer
at the sea surface to move, defining the transducer position when it sends an
acoustic pulse to the transponder as $X_S$ and its position when it receives a
reply as $X_R$. During this time, the transponder on the seafloor is stationary
but may still be offset from its assumed position by $\Delta X_1$. In this case
the travel time residual is the sum of the residual from the raypath
$\vec{D}_{1S}$ from the initial transducer position to the transponder and the
residual from the return raypath $\vec{D}_{1R}$ from the transponder to the
final transducer position:

$$\left( \frac{\vec{D}_{1S} + \vec{D}_{1R}}{c} \right) \cdot \Delta X_1 = \Delta a_{1S} + \Delta a_{1R} = \Delta a_1$$

Until now we have only considered a single transponder, but we may generalize
this to an array with $i$ transponders. Assuming that the array moves as a
block, each transponder has the same offset, so
$\Delta X_1 = \Delta X_2 = \cdots = \Delta X_i = \Delta X$. Furthermore, since
there may be subtle oceanographic variations across the array, let us assume
that the raypath from the transducer to each transponder travels through a
different mean sound velocity $c_i$. With this assumption, we can define the
variable $\vec{P}_i$ as

$$\vec{P}_i = \left( \frac{\vec{D}_{iS} + \vec{D}_{iR}}{c_i} \right)$$

Now, we can write a generalized equation relating the change in array position
to the travel time residuals:

$$
A \cdot \Delta X = \Delta \vec{a},
$$

$$
A =
\begin{bmatrix}
\vec{P}_1 \\
\vec{P}_2 \\
\vdots \\
\vec{P}_i
\end{bmatrix},
$$

$$
\Delta \vec{a} =
\begin{bmatrix}
\Delta a_1 \\
\Delta a_2 \\
\vdots \\
\Delta a_i
\end{bmatrix}
$$

Note that in the equations above, $A$ is an $i \times 3$ matrix and a has a
length of $i$.

You will no doubt have noticed at this point that I have very deliberately
structured the relationship between the transponder and array offsets and the
travel time residuals as a system of linear equations. This is to imply that if
we measure the two-way travel time between a sea surface transducer and an array
of seafloor transponders, we may infer the offset of the array from an assumed a
priori position as long as we have a good estimate of the sound velocity profile
across the array and the transducer position when sending and receiving acoustic
pulses. This exercise forms the backbone of the GNSS-Acoustic technique,
although in practice there are some more steps we must take when interpreting
some data.

The observations used in this inversion are not perfect, so it is a good idea to
construct a weighting matrix W which depends on the uncertainties of the travel
time residuals. The travel time residuals have uncertainties dependent on three
sources: the acoustic measurement uncertainty of the transducer, the position
uncertainty of the transducer when the acoustic pulse is sent, and the position
uncertainty of the transducer when the return acoustic pulse is received. Let
$\sigma_a^2$ be the variance of an acoustic measurement, $C_S$ be the
$3 \times 3$ covariance matrix of the transducer position $X_S$, and $C_R$ be
the $3 \times 3$ covariance matrix of the transducer position $X_R$. Assuming
that $\Delta a$ is a stochastic variable, the error propagation to derive the
$i \times i$ matrix $W$ may be written out as:

$$
W = \sigma_a^2 I + \left( \left( \frac{\partial \Delta \vec{a}}{\partial X_S} \right)^T \cdot C_S \cdot \left( \frac{\partial \Delta \vec{a}}{\partial X_S} \right) \right) + \left( \left( \frac{\partial \Delta \vec{a}}{\partial X_R} \right)^T \cdot C_R \cdot \left( \frac{\partial \Delta \vec{a}}{\partial X_R} \right) \right)
$$

Note that in the above formulation, the partial derivatives simplify to:

$$
\frac{\partial \Delta \vec{a}}{\partial X_S} = \left( \frac{\hat{D}_{1S}}{c_1}, \frac{\hat{D}_{2S}}{c2}, \cdots , \frac{\hat{D}_{iS}}{ci} \right),
$$

$$
\frac{\partial \Delta \vec{a}}{\partial X_R} = \left( \frac{\hat{D}_{1R}}{c_1}, \frac{\hat{D}_{2R}}{c2}, \cdots , \frac{\hat{D}_{iR}}{ci} \right)
$$

As another unit check, the elements in the partial derivatives above all have
units of slowness (nominally s/m) and the elements of $C_S$ and $C_R$ all have
units of length squared (nominally m{sup}`2`), so the entire matrix
multiplication will result in a covariance matrix whose elements have units of
time squared (nominally s{sup}`2`). Likewise, $\sigma_a^2$ also has units of
time squared so the units of $W$ are consistent.

As one final note for constructing $W$, it is possible to save some computing
time by assuming that $X_S$ and $X_R$ are close to each other and that
$C_S \approx C_R$. In this case you only have to compute the above matrix
multiplication once and can write $W$ as

$$
W = \sigma_a^2 I + 2 \left( \left( \frac{\partial \Delta \vec{a}}{\partial X_S} \right)^T \cdot C_S \cdot \left( \frac{\partial \Delta \vec{a}}{\partial X_S} \right) \right)
$$

Following this, the weighted inversion may be written out as

$$
A^T W A \cdot \Delta X = A^T W \cdot \Delta \vec{a}
$$

This inversion may be solved according to the user’s preference, such as with
the method of least squares.

One more consideration is that in the real world we do not collect a single
travel time measurement for each transponder but rather a time series of
measurements during which the seafloor transponders are stationary but the sea
surface transducer moves. In this case, we can construct the $A$, $W$, and
$\Delta \vec{a}$ variables by calculating them at each epoch $j$ as detailed
above and then summing them prior to the inversion. Thus,

$$
\left( \sum_j A_j^T W_j A_j \right) \cdot \Delta X = \left( \sum_j A_j^T W_j \Delta \vec{a}_j \right)
$$

You may also want to construct a pseudo-constraint matrix $Q$ in addition to the
above variables to perform a constrained inversion. For instance, you could
construct a $Q$ matrix to keep the baselines between transponders constant and
force them to resolve the same $\Delta X$.

In addition, the inversion may not always converge immediately. In this case you
may iterate the inversion until its solution converges. Let the solution of the
$k$th inversion be $\Delta X_k$. Simply repeat the inversion on subsequent
iterations while updating the transponder positions such that

$$
\Delta X_{ki} = \Delta X_i + \sum_0^{k-1} \Delta X_k
$$

The final transponder offsets will be

$$
\Delta X = \sum_k \Delta X_k
$$
