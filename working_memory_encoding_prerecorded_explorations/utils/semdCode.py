code = """
if (ISynFac > 0.00001) {
   ISynTrigger += trigInit * Isyn;
}
if (RefracTime <= 0.0) {
   const scalar inputCurrent = ISynTrigger * ISynFac;
   const scalar alpha = ((inputCurrent + Ioffset) * Rmembrane) + Vrest;
   V = alpha - (ExpTC * (alpha - V));
}
else {
  RefracTime -= dt;
}
ISynTrigger *= trigExpDecay;
"""

threshold = """
RefracTime <= 0.0 && V >= Vthresh
"""

reset = """
V = Vreset;
RefracTime = TauRefrac;
"""
