- **Transducer Delay Time** The delay time of the surface platform between an
  interrogate command being sent to the transducer and the interrogation ping
  being emitted, in seconds. Different surface platforms will have different
  delays. In general, GNATSS assumes that:
  - The time variable in the _pxp_tt_ file is when the interrogate command is
    sent (before the start of the transducer delay time)
  - The two-way travel times logged in the _pxp_tt_ file do not include the
    transducer delay time since the interrogation ping was not in the water
    during this delay.
  - The internal delay of the seafloor transponders _is_ included in the two-way
    travel times by convention since the interrogation ping is in the water
    during this delay.
  - The transducer positions in the _POS_FREED_TRANS_TWTT_ file are at times
    when the interrogation ping is sent and the replies received, and thus after
    the transducer delay.
  - You can set the transducer delay to zero if you remove the delay time prior
    to running GNATSS. GNATSS will run a cross-check before processing data and
    return an error if the ping timings of the _pxp_tt_ and
    _POS_FREED_TRANS_TWTT_ files do not align.
