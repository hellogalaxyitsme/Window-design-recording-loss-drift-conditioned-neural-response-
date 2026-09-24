"""Verified Axion metadata/memmap access and a fixed negative-peak detector."""
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from pyaxion.axis_reader import AxisFile,ReturnDimension

def open_voltage(path,verify=True):
    """Use Axion metadata to map continuous int16 samples without loading a plate."""
    with AxisFile(str(path)) as af:
        ds=af.raw_voltage
        if ds is None: ds=af.broad_band_high
        if ds is None: raise ValueError('No continuous voltage dataset')
        assert ds.num_samples_per_block==1 and ds.num_datasets_per_block==1 and ds.block_header_size==0
        assert int(ds.sample_type)==0, 'Only the verified int16 sample type is supported'
        channels=ds.channel_array.channels; nc=len(channels)
        assert nc==ds.num_channels_per_block and ds.data_region_length%(2*nc)==0
        n=int(ds.data_region_length//(2*nc)); fs=float(ds.sampling_frequency); scale=float(ds.voltage_scale)
        assert abs(n/fs-float(ds.duration))<1e-6 and abs(fs-12500)<1e-6
        assert ds.data_region_start+ds.data_region_length<=Path(path).stat().st_size
        raw=np.memmap(path,dtype='<i2',mode='r',offset=int(ds.data_region_start),shape=(n,nc))
        mapping=[]
        for i,c in enumerate(channels):
            mapping.append(dict(column=i,well=f'{chr(64+int(c.well_row))}{int(c.well_column)}',
                well_row=int(c.well_row),well_column=int(c.well_column),
                electrode_column=int(c.electrode_column),electrode_row=int(c.electrode_row),
                electrode=int(c.electrode_column)*10+int(c.electrode_row),hardware_channel=int(c.channel_index)))
        if verify:
            sample=ds.load_raw_data(wells='A1',electrode=[11,12],timespan=[0,2],dimension=ReturnDimension.BYELECTRODE)
            checked=0
            for record in sample.flat:
                if record is None or not hasattr(record,'get_voltage_vector'): continue
                column=next(i for i,c in enumerate(channels) if c==record.channel)
                np.testing.assert_array_equal(raw[:len(record.data),column],record.data)
                np.testing.assert_allclose(raw[:len(record.data),column].astype(float)*scale,
                    record.get_voltage_vector(),rtol=1e-14,atol=1e-15)
                checked+=1
            assert checked==2
        meta=dict(path=str(path),samples=n,channels=nc,sampling_frequency=fs,voltage_scale=scale,
            duration_s=n/fs,data_start=int(ds.data_region_start),data_length=int(ds.data_region_length),
            header_version=f'{af.header_version_major}.{af.header_version_minor}',
            acquisition_description=ds.description,plate_metadata=af.metadata,
            direct_memmap_reader_comparison='passed' if verify else 'not_requested')
    return raw,mapping,meta


def detect_negative_peaks(voltage_uv,fs,minimum_multiplier=4.,refractory_ms=1.):
    """Fixed amplitude threshold; larger thresholds select nested retained peaks.

    Noise is the full-recording MAD scale. The reference peak set uses the lowest
    threshold and minimum peak separation, then higher thresholds filter heights.
    No activity-based trial or well selection is performed.
    """
    v=np.asarray(voltage_uv)
    if v.ndim!=1 or not np.all(np.isfinite(v)): raise ValueError('Invalid voltage trace')
    median=float(np.median(v)); noise=float(np.median(np.abs(v-median))/.67448975)
    if noise<=0: return np.array([],dtype=np.int64),np.array([],float),dict(valid=False,median_uv=median,noise_uv=noise)
    peak,properties=find_peaks(median-v,height=minimum_multiplier*noise,distance=max(1,int(np.ceil(fs*refractory_ms/1000))))
    return peak,properties['peak_heights'],dict(valid=True,median_uv=median,noise_uv=noise)
