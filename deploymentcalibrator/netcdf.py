import os
import numpy as np
from netCDF4 import Dataset
import arrow

class NetCDF_HZG(object):
    ''' Minimal implementation of NetCDF file adhering to HZG specs

    Parameters
    ----------
    filename : string
         name of netcdf file
    title : string
         title attribute
    source : string
         source attribute
    originator : string
         originator attribute
    contact : string
         contact attribute
    crs : string ('WSG84')
         coordinate system attribute


    Note
    ----
    This class can (should) be used using a context manager.

    Example
    -------
    >>> with NetCDF_HZG("test.nc", **conf) as nc:
    >>> nc.add_parameter("latitude", "degree north", velocity.t, velocity.lat)
    >>> nc.add_parameter("longitude", "degree east", velocity.t, velocity.lon)
    >>> nc.add_parameter("eastward current", "m/s" , velocity.t, velocity.z, velocity.u)
    >>> nc.add_parameter("northward current", "m/s" , velocity.t, velocity.z, velocity.v)
    >>> nc.add_parameter("upward current", "m/s" , velocity.t, velocity.z, velocity.w)
    '''
    def __init__(self, filename, mode='w', title="", source="", originator="", contact="", crs='WGS84'):
        self.dataset = Dataset(filename, mode=mode)
        self.dims = {}
        self.dataset.title = title
        self.dataset.source = source
        self.dataset.originator = originator
        self.dataset.contact = contact
        self.dataset.creation_date=arrow.utcnow().format("YYYY-MM-DDTHH:MM:SS")

    @staticmethod
    def get_default_conf():
        ''' Returns a prefilled configuration dictionary '''
        conf = dict(title="tbd",
                    source="tbd",
                    originator="Lucas Merckelbach",
                    contact="lucas.merckelbach@hzg.de")
        return conf
        
        
    def __enter__(self, *p):
        return self

    def __exit__(self, *p):
        self.dataset.close()
        
    def _get_time_name(self, name):
        f = list(os.path.split(name))
        f[-1]='time'
        s = "/".join(f)
        return s
            
    def _check_for_time_dimension(self,t, name, time_dimension):
        tstr = self._get_time_name(name)
        if not time_dimension in self.dims.keys():
            dim = self.dataset.createDimension(time_dimension, size=None)
            self.dims[time_dimension] = dim
            var = self.dataset.createVariable(tstr, "f8", dimensions=(time_dimension,))
            var.units = 'seconds since 1-1 1970 00:00:00'
            var.standard_name = 'time'
            var[:] = t

    def _check_for_z_dimension(self, z):
        if not "Z" in self.dims.keys():
            dim = self.dataset.createDimension("Z", size=len(z))
            self.dims["Z"] = dim
            var = self.dataset.createVariable('z', "f8", dimensions=("Z",))
            var.units = 'm'
            var.standard_name = 'depth'
            var.long_name = 'water depth relative to sea surface'
            var.positive='down'
            var[:] = z

    def add_meta_variable(self, name, unit, value, dtype='f4'):
        v = self.dataset.createVariable(name, dtype, dimensions=())
        v.unit = unit
        v[...] = value
            
    def add_parameter(self, name, unit, *v, standard_name=None, time_dimension=None):
        ''' Add a parameter
        
        Parameters
        ----------
        name : string 
            name of variable
        unit : string
            unit of variable
        *v : list of arrays, length 2 or 3
             (time, values), or (time, z, values)
        '''
        time_dimension = time_dimension or "T"
        
        if len(v) == 2:
            self._check_for_time_dimension(v[0], name, time_dimension)
            var = self.dataset.createVariable(name, "f8", dimensions=(time_dimension,))
            var.units = unit
            var[:] = v[1]
        elif len(v) == 3:
            self._check_for_time_dimension(v[0], time_dimension)
            self._check_for_z_dimension(v[1])
            var = self.dataset.createVariable(name, "f8", dimensions=(time_dimension, "Z"))
            var.units = unit
            var[:] = v[2]
        else:
            raise ValueError("Variable type not supported.")
        if not standard_name is None:
            var.standard_name = standard_name

    def close(self):
        ''' Closes netcdf file'''
        self.dataset.close()

    
class GliderFlightNetCDF(NetCDF_HZG):

    def __init__(self, filename, **conf):
        super().__init__(filename, **conf)

    def write_glider_flight_parameters(self, GM, calibration_result):
        '''
        '''
        group = 'glider_flight'
        tdim = 'Tgf'
        for p in GM.parameters:
            if p in calibration_result.keys():
                continue # need to write this as time variable.
            unit = GM.parameter_units[p]
            value = GM.__dict__[p]
            self.add_meta_variable(f"{group}/{p}", unit, value)
        for p in calibration_result.keys():
            if p == 't':
                continue
            unit = GM.parameter_units[p]
            self.add_parameter(f"{group}/{p}", unit, calibration_result['t'], calibration_result[p], time_dimension=tdim)
        
    def write_thermal_lag_coefs(self, thermal_lag_coefs):
        group="thermal_lag_coefs"
        units = ["-", "s", "s"]
        names = ["alpha", "beta", "tau"]
        for p, v, u in zip(names, thermal_lag_coefs, units):
            self.add_meta_variable(f"{group}/{p}", u, v)

    def write_model_results(self, model_result):
        parameters = "u w U alpha pitch ww heading depth lat lon density SA CT pot_density buoyancy_change".split()
        units = ["m s^{-1}", "m s^{-1}", "m s^{-1}","rad", "rad", "m s^{-1}", "rad", "m", "decimal degree", "decimal degree", 
                 "kg m^{-3}", "kg kg^{-1}","degree Celcius", "kg m^{-3}", "cc"]
        long_names = ["horizontal velocity relative to water (in flight direction)",
                      "vertical velocity relative to water",
                      "speed through water", 
                      "angle of attack",
                      "pitch angle",
                      "vertical water velocity",
                      "heading angle",
                      "depth", 
                      "latitude",
                      "longitude",
                      "in-situ density",
                      "absolute salinity",
                      "conservative temperature",
                      "potential density",
                      "buoyancy_change"]
        t = model_result.t
        for p, u, ln, v in zip(parameters, units, long_names, model_result[1:]):
            self.add_parameter(p, u, t, v, standard_name=ln)
