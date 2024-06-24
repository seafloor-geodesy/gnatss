# Experimental Design

## Overview

Global Navigation Satellite System - Acoustic (GNSS-A) is a seafloor geodetic
tool roughly analogous to GNSS. Standard GNSS is unable to position points on
the seafloor because the ocean is opaque to the satellite radar used to
communicate with GNSS receivers. To address this, the GNSS-A technique involves
deploying two sets of instruments: acoustic transponders that rest on the
seafloor and a sea surface platform with a transducer and standard GNSS. The
surface platform utilizes acoustics to position the seafloor transponder
relative to itself and GNSS to anchor the entire system (seafloor transponder
and surface platform) in a terrestrial reference frame such as ITRF.

## Experimental Setup

A GNSS-A site consists of an array of seafloor transponders. These transponders
rest on the seafloor in a "listen" mode, replying with an acoustic pulse when
they receive a like pulse with a specific address. This allows a user to measure
the two-way travel time (TWTT) between a surface platform and a seafloor
transponder. Because the TWTT is sensitive to the speed of sound in seawater, a
value that may change due to instantaneous oceanographic sources such as
internal gravity waves, a single transponder may not be positioned to much
better than ~10 cm due to oceanographic signals. This may be mitigated by
deploying acoustic transponders in arrays of 3 or more and averaging the
apparent transponder positions in space and time, which allows us to position
the geometric center of a transponder array to ~1 cm.

The sea surface platform may be a wave glider, research vessel, or buoy. GNATSS
assumes that the surface platform surveys the acoustic transponders from the
center of the array and interrogates each transponder simultaneously at regular
intervals, typically every 15-20 seconds. Surveying from the array center yields
robust array positions but less information about sound speed variations in the
water column. Thus, we typically recommend surveying for 3-5 days at a single
site and averaging the array position over this time period to mitigate
oceanographic signals.
