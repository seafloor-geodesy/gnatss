# Example rotation procedure

import pandas as pd
from scipy.spatial.transform import Rotation as rot

### You will need to load your own data into a compatible pandas dataframe. Note that you
### will need antenna positions at send and receive times, as well as attitudes

atd_offset = np.array(
    [0.0053, 0, 0.92813]
)  # Antenna-transducer offset; forward, rightward, and downward

# Compute transducer offset from the antenna
r = rot.from_euler("xyz", df[["roll0", "pitch0", "head0"]].values, degrees=True)
offset = r.as_matrix() @ atd_offset
d_e0 = offset[:, 1]
d_n0 = offset[:, 0]
d_u0 = -offset[:, 2]
r = rot.from_euler("xyz", df[["roll1", "pitch1", "head1"]].values, degrees=True)
offset = r.as_matrix() @ atd_offset
d_e1 = offset[:, 1]
d_n1 = offset[:, 0]
d_u1 = -offset[:, 2]

df["d_e0"] = d_e0
df["d_n0"] = d_n0
df["d_u0"] = d_u0
df["d_e1"] = d_e1
df["d_n1"] = d_n1
df["d_u1"] = d_u1

# Compute location of transducer for send (0) and receive(1) times
df["td_e0"] = df.ant_e0 + df.d_e0
df["td_n0"] = df.ant_n0 + df.d_n0
df["td_u0"] = df.ant_u0 + df.d_u0
df["td_e1"] = df.ant_e1 + df.d_e1
df["td_n1"] = df.ant_n1 + df.d_n1
df["td_u1"] = df.ant_u1 + df.d_u1
