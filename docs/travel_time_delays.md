## A Note on Travel Time Delays

In general, a GNSS-Acoustic Two-Way Travel Time (TWTT) measurement will include
two additional timing delays in addition to the time when the acoustic pulse is
traveling through the water column. We refer to these as:

- **The Transducer Delay Time** The delay time of the surface platform between
  an interrogate command being sent to the transducer and the interrogation ping
  being emitted. Different surface platforms will have different delays.

- **The Transponder Turn-Around Time (TAT)** The TAT is a user defined delay
  programmed into each individual transponder. When a transponder receives an
  acoustic pulse, it will wait for the TAT before sending a reply pulse. The
  goal is to program each transponder in an array with a different TAT so that
  when the surface platform interrogates them from the array center, the reply
  pulses do not interfere.

The structure of a GNSS-Acoustic ping can be broken down into the following
phases:

1. On-board computer sends interrogate command to the transducer.
2. Surface platform waits for the transducer delay time.
3. Transducer emits interrogate ping.
4. Interrogate ping travels through the water column.
5. Seafloor transponder receives interrogate ping.
6. Transponder waits until the TAT has elapsed.
7. Transponder emits reply ping.
8. Reply ping travels through the water column.
9. Transducer on surface platform receives reply ping.

In GNSS-Acoustic processing, the critical transducer positions that must be
known are the position at ping send (Phase 3) and ping reply (Phase 9) since
these positions are when the acoustic pulse physically starts and finishes
traveling through the water column to and from the transducer. Consistent with
this, by convention the TAT is thus included in the TWTT measurement while the
transducer delay is excluded from the TWTT measurement. GNATSS requires the
TWTTs submitted in the _pxp_tt_ file to adhere to this convention, in which case
the transducer delay in the config file may be set to 0.

### The SV-3 Wave Glider Platform

The Sonardyne GNSS-A platform integrated in the model SV-3 Wave Gliders used in
the Near-trench Community Geodetic Experiment include the transducer delay in
the raw TWTT measurement, so the user must manually remove it when generating
_pxp_tt_ files. See the User manual for more information.
