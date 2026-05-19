import numpy as np
from pygenn import init_postsynaptic, init_weight_update, create_neuron_model, init_toeplitz_connectivity
from utils.semdCode import code, threshold, reset


#################################
# ##### Kernel definition ##### #
#################################
def gaussian(size, sigma):
    x = np.linspace(-size//2, size//2, size)
    y = np.linspace(-size//2, size//2, size)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x**2+y**2)/(2*sigma**2))
    kernel = (kernel-kernel.min())/(kernel.max()-kernel.min())
    return kernel


def blockOMS(pixels, kernel, stimulus, params, model):
    #####################
    # ##### Scene ##### #
    #####################
    ##### Size input #####
    sizeInRow, sizeInCol = pixels
    ##### Size kernel #####
    kernelRow, kernelCol = kernel
    ##### Size output #####
    sizeOutRow, sizeOutCol = (sizeInRow-kernelRow+1), (sizeInCol-kernelCol+1)

    ###################################
    # ##### GeNN stimulus spike ##### #
    ###################################
    timeEnd = np.cumsum([len(spike) for spike in stimulus])
    timeStart = np.concatenate(([0], timeEnd[:-1]))
    timeSpike = np.concatenate(stimulus)/params['timeWindow']

    #################################
    # ##### Neuron population ##### #
    #################################
    popSource = model.add_neuron_population(
        pop_name='stim',
        num_neurons=sizeInRow*sizeInCol,
        neuron='SpikeSourceArray',
        params={}, vars={'startSpike': timeStart, 'endSpike': timeEnd}
    )
    popSource.extra_global_params['spikeTimes'].set_init_values(timeSpike)
    popSource.spike_recording_enabled = True

    paramLifCent = {
        'C': params['tauMCent'],  # nF
        'TauM': params['tauMCent'],  # ms
        'Ioffset': 0.0,  # nA
        'Vrest': 0.0,  # -65.0,  # mV
        'Vthresh': params['thVCent'],  # -50.0,  # mV
        'Vreset': 0.0,  # -70.0,  # mV
        'TauRefrac': 0.0,  # ms
    }
    varLifCent = {
        'V': paramLifCent['Vrest'],  # mV
        'RefracTime': 0.0,  # ms
    }
    popCenter = model.add_neuron_population(
        pop_name='cent',
        num_neurons=sizeOutRow*sizeOutCol,
        neuron='LIF',
        params=paramLifCent, vars=varLifCent
    )
    popCenter.spike_recording_enabled = True

    paramLifSurr = {
        'C': params['tauMSurr'],  # nF
        'TauM': params['tauMSurr'],  # ms
        'Ioffset': 0.0,  # nA
        'Vrest': 0.0,  # -65.0,  # mV
        'Vthresh': params['thVSurr'],  # -50.0,  # mV
        'Vreset': 0.0,  # -70.0,  # mV
        'TauRefrac': 0.0,  # ms
    }
    varLifSurr = {
        'V': paramLifSurr['Vrest'],  # mV
        'RefracTime': 0.0,  # ms
    }
    popSurrou = model.add_neuron_population(
        pop_name='surr',
        num_neurons=sizeOutRow*sizeOutCol,
        neuron='LIF',
        params=paramLifSurr, vars=varLifSurr
    )
    popSurrou.spike_recording_enabled = True

    # ##### Synaptic connection ##### #
    kernelCent = gaussian(8, 1)/params['scalingCent']
    kernelSurr = gaussian(8, 4)/params['scalingSurr']
    paramsConv = {
        'conv_kh': kernelRow, 'conv_kw': kernelCol,
        'conv_ih': sizeInRow, 'conv_iw': sizeInCol, 'conv_ic': 1,
        'conv_oh': sizeOutRow, 'conv_ow': sizeOutCol, 'conv_oc': 1
    }

    model.add_synapse_population(
        pop_name='stimExcitCent', matrix_type='TOEPLITZ',
        source=popSource, target=popCenter,
        postsynaptic_init=init_postsynaptic('DeltaCurr'),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': kernelCent.flatten()}),
        connectivity_init=init_toeplitz_connectivity('Conv2D', paramsConv)
    )
    model.add_synapse_population(
        pop_name='stimExcitSurr', matrix_type='TOEPLITZ',
        source=popSource, target=popSurrou,
        postsynaptic_init=init_postsynaptic('DeltaCurr'),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': kernelSurr.flatten()}),
        connectivity_init=init_toeplitz_connectivity('Conv2D', paramsConv)
    )

    return {'popSource': popSource, 'popCenter': popCenter, 'popSurrou': popSurrou, 'pixelsInput': (sizeInRow, sizeInCol), 'pixelsOutput': (sizeOutRow, sizeOutCol)}


def blockSEMD(pixels, stimulus, model, params):
    sizeInRow, sizeInCol = pixels

    # ##### GeNN stimulus spike ##### #
    timeEnd = np.cumsum([len(spike) for spike in stimulus])
    timeStart = np.concatenate(([0], timeEnd[:-1]))
    timeSpike = np.concatenate(stimulus)/params['timeWindow']

    # ##### Neuron population ##### #
    popSource = model.add_neuron_population(
        pop_name='stim',
        num_neurons=sizeInRow*sizeInCol,
        neuron='SpikeSourceArray',
        params={}, vars={'startSpike': timeStart, 'endSpike': timeEnd}
    )
    popSource.extra_global_params['spikeTimes'].set_init_values(timeSpike)
    popSource.spike_recording_enabled = True

    semdModel = create_neuron_model(
        class_name='semd',
        params=['C', 'TauM', 'Vrest', 'Vreset', 'Vthresh', 'Ioffset', 'TauRefrac', 'TauSynTrig'],
        vars = [('V', 'scalar'), ('RefracTime', 'scalar'), ('ISynTrigger', 'scalar')],
        derived_params=[('ExpTC', lambda pars, dt: np.exp(-dt/pars['TauM'])), ('Rmembrane', lambda pars, dt: pars['TauM']/pars['C']), ('trigExpDecay', lambda pars, dt: np.exp(-dt/pars['TauSynTrig'])), ('trigInit', lambda pars, dt: (pars['TauSynTrig']*(1.0-np.exp(-dt/pars['TauSynTrig'])))*(1.0/dt))],
        sim_code=code, threshold_condition_code=threshold, reset_code=reset,
        additional_input_vars=[('ISynFac', 'scalar', 0.0)]
    )
    semdParams = {
        'C': 0.25,
        'TauM': 10.0,
        'Vrest': 0.0,
        'Vreset': 0.0,
        'Vthresh': 5.0,
        'Ioffset': 0.0,
        'TauRefrac': 0.0,
        'TauSynTrig': 3.0
    }
    semdVars = {
        'V': 0.0,
        'RefracTime': 0.0,
        'ISynTrigger': 0.0
    }

    popLR = model.add_neuron_population(
        pop_name='LR',
        num_neurons=sizeInRow*(sizeInCol-1),
        neuron=semdModel,
        params=semdParams, vars=semdVars
    )
    popLR.spike_recording_enabled = True

    popRL = model.add_neuron_population(
        pop_name='RL',
        num_neurons=sizeInRow*(sizeInCol-1),
        neuron=semdModel,
        params=semdParams, vars=semdVars
    )
    popRL.spike_recording_enabled = True

    popTB = model.add_neuron_population(
        pop_name='TB',
        num_neurons=(sizeInRow-1)*sizeInCol,
        neuron=semdModel,
        params=semdParams, vars=semdVars
    )
    popTB.spike_recording_enabled = True

    popBT = model.add_neuron_population(
        pop_name='BT',
        num_neurons=(sizeInRow-1)*sizeInCol,
        neuron=semdModel,
        params=semdParams, vars=semdVars
    )
    popBT.spike_recording_enabled = True

    # ##### Synaptic connection ##### #
    pixel = np.arange(0, sizeInRow*sizeInCol, 1, dtype=np.uint32).reshape((sizeInRow, sizeInCol))

    synHorFac = pixel[:, :-1].flatten()
    synHorTri = pixel[:, 1:].flatten()
    synHor = np.arange(0, sizeInRow*(sizeInCol-1), 1, dtype=np.uint32)

    facLR = model.add_synapse_population(
        pop_name='facLR', matrix_type='SPARSE',
        source=popSource, target=popLR,
        postsynaptic_init=init_postsynaptic('ExpCurr', {'tau': 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': np.ones_like(synHor)}),
    )
    facLR.post_target_var = 'ISynFac'
    facLR.set_sparse_connections(synHorFac, synHor)
    triLR = model.add_synapse_population(
        pop_name='triLR', matrix_type='SPARSE',
        source=popSource, target=popLR,
        postsynaptic_init=init_postsynaptic('DeltaCurr'),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': np.ones_like(synHor)*20}),
    )
    triLR.set_sparse_connections(synHorTri, synHor)

    facRL = model.add_synapse_population(
        pop_name='facRL', matrix_type='SPARSE',
        source=popSource, target=popRL,
        postsynaptic_init=init_postsynaptic('ExpCurr', {'tau': 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': np.ones_like(synHor)}),
    )
    facRL.post_target_var = 'ISynFac'
    facRL.set_sparse_connections(synHorTri, synHor)
    triRL = model.add_synapse_population(
        pop_name='triRL', matrix_type='SPARSE',
        source=popSource, target=popRL,
        postsynaptic_init=init_postsynaptic('DeltaCurr'),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': np.ones_like(synHor)*20}),
    )
    triRL.set_sparse_connections(synHorFac, synHor)


    synVerFac = pixel[:-1, :].T.flatten()
    synVerTri = pixel[1:, :].T.flatten()
    synVer = np.arange(0, (sizeInRow-1)*sizeInCol, 1, dtype=np.uint32).reshape((sizeInRow-1, sizeInCol)).T.flatten()

    facTB = model.add_synapse_population(
        pop_name='facTB', matrix_type='SPARSE',
        source=popSource, target=popTB,
        postsynaptic_init=init_postsynaptic('ExpCurr', {'tau': 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': np.ones_like(synVer)}),
    )
    facTB.post_target_var = 'ISynFac'
    facTB.set_sparse_connections(synVerFac, synVer)
    triTB = model.add_synapse_population(
        pop_name='triTB', matrix_type='SPARSE',
        source=popSource, target=popTB,
        postsynaptic_init=init_postsynaptic('DeltaCurr'),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': np.ones_like(synVer)*20}),
    )
    triTB.set_sparse_connections(synVerTri, synVer)

    facBT = model.add_synapse_population(
        pop_name='facBT', matrix_type='SPARSE',
        source=popSource, target=popBT,
        postsynaptic_init=init_postsynaptic('ExpCurr', {'tau': 5.0}),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': np.ones_like(synVer)}),
    )
    facBT.post_target_var = 'ISynFac'
    facBT.set_sparse_connections(synVerTri, synVer)
    triBT = model.add_synapse_population(
        pop_name='triBT', matrix_type='SPARSE',
        source=popSource, target=popBT,
        postsynaptic_init=init_postsynaptic('DeltaCurr'),
        weight_update_init=init_weight_update('StaticPulse', {}, {'g': np.ones_like(synVer)*20}),
    )
    triBT.set_sparse_connections(synVerFac, synVer)

    return {'popSource': popSource, 'popLR': popLR, 'popRL': popRL, 'popTB': popTB, 'popBT': popBT, 'pixelsOutput': (sizeInRow, sizeInCol)}
