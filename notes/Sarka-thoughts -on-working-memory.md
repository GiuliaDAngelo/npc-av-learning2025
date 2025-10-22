## Thoughs on Working memory 
### **Can we bind DINO embeddings directly with SSP representations of position and rotation?** 
**DINOv2 embeddings** - 384 element vectors, non-unitary, might be correlated in the representational space (not strictly close to orthogonal)  
**VSA binding conditions:** 
- The vectors must be unitary in the SSP sense:
  $|\mathcal{F}( \text{DINO} )[k]| = 1,$ for each frequency $k$ of the phase spectrum
- Current implementation:
  ```python
	# normalize dino embedding so its norm is 1
	dino_embedding /= np.linalg.norm(dino_embedding) + 1e-10
  ```
- To make the embedding SSP-unitary:
  ```python
  def to_unitary_ssp(vec, n=None):
      v = vec.astype(np.float64).ravel()
      if n is None: n = v.size
      V = np.fft.rfft(v, n=n)
      V_unit = V / (np.abs(V) + 1e-9)        # unit magnitude per bin
      u = np.fft.irfft(V_unit, n=n)
      u /= (np.linalg.norm(u) + 1e-9)         # optional
      return u
  
  u = to_unitary_ssp(dino_embedding, n=SSP_DIM)
    ```

**Non-orthogonality of DINO embeddings:** 
As DINO embedings don't originate from the SSP space, they can be correlated (not almost-orthogonal as SSP) $\rightarrow$ more similar to each other. 
Non-orthogonality of DINO embeddings can affect working memory performance by increasing interference during binding and memory retrieval. (Storing multiple patches in the same memory might result in overlap of the features. $\rightarrow$ Lower capacity of the memory.).

**DINO embeddings:** semantic richness $\rightarrow$ high representational quality;
increased interference compared to random SSP vectors.

Binding the DINO patches with SSP positions should orthogonalize the vectors: cos similarity $\rightarrow 0$. Unless we need to unbind (DINO-patch, SSP-position, SSP-rotation) precisely, it should be ok.  
