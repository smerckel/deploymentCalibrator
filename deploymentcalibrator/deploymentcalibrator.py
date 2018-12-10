from collections import namedtuple
import glob
import os

import numpy as np
from scipy.interpolate import interp1d

import dbdreader
from profiles.ctd import ThermalLag

from . import easy_gsw

class DeploymentCalibrator(object):
    '''A class to calibrate the glider flight model for a full deployment

    Parameters
    ----------

    glider  : string
              name of glider
    path    : string
              path where glider files (dbd and ebd) reside
    interval: float
              time interval for which glider data files are aggregated to optimize
              for glider flight coefficients.

    thermal_lag_coefs : tuple of floats
                        if the ThermalLag class can be imported, the CTD data will be corrected
                        for thermal lag effects
    pitch_correct_coefs: tuple of float (default: (1.0, 0.0))
                         the pitch reported by the glider is adjusted with the two coefficients
                         given. The first coefficient is a scaling factor, the second an offset.

    This class provides an interface to calibrate the glider flight
    model for a full deployment.  The calibrated model can be used to
    generate glider flight velocities accounting for time varying
    glider flight coefficients (notably the drag coefficient).

    Example
    -------

    A typical use is:
    
    * define a flight model

    * create an instance of the DeploymentCalibrator with optional
      pitch correction factors and thermal lag correction factors
    * Calibrate the model for specified parameters, returning time 
      series for the parameters with a resolution determined by the
      interval setting
    * construct interpolating functions of the time series
    * compute the glider flight velocities for given flight model 
      and interpolating functions

    >>> import gliderflight
    >>> GM = gliderflight.SteadyStateCalibrate(rho0=1006)
    >>> GM.define(ah=3.8, Cd1=10.5, mg=70.1, Vg=70/1006, Cd0=0.15)
    >>> pitch_correction_coefs = (0.8308, -0.006)
    >>> mc = DeploymentCalibrator(glider='comet', path="/home/lucas/gliderdata/latvia201711_test/hd",
                                  interval=86400,
                                  pitch_correction_coefs=pitch_correction_coefs)
    >>> r = mc.calibrate(glider_model=GM, parameters="Cd0 Vg".split(), min_depth=10, max_depth=40)
    >>> coef_functions = dict(Cd0=mc.construct_ifun(r, 'Cd0'),
                              Vg=mc.construct_ifun(r, 'Vg'))
    >>> model_result = mc.compute_glider_velocity(GM, coef_functions)

    The results are stored in the named tuple model_result

    '''

    def __init__(self, glider, path, interval,
                 thermal_lag_coefs=(0.011, 0.073),
                 pitch_correction_coefs=(1.,0.)):
        self.glider = glider
        self.path = path
        self.interval = interval
        self.pitch_correction_coefs = pitch_correction_coefs
        self.thermal_lag_coefs = thermal_lag_coefs
        self.data_cache = {}
        
    def get_filename_pattern(self):
        ''' Get a pattern for matching glider data filenames (assuming dbd files).
        
        Returns
        -------
            p : pattern for consumption by glob.glob()
        '''
        p = os.path.join(self.path, "{}*.dbd".format(self.glider))
        return p
    
    def get_filename_list_per_interval(self):
        ''' Bin filenames
        
        Returns
        -------
        binned_filenames : a list of tuples, each containing the average time of an interval, and 
                           a list of filenames that have their opening times falling in the this interval.

        This method conveniently bins all filenames based on opening times. The interval is set in the 
        constructor of this class (usually something like 1 day).
        '''
        p = self.get_filename_pattern()
        ps = dbdreader.DBDPatternSelect()
        binned_filenames = ps.bins(pattern=p, width=self.interval)
        return binned_filenames

    def thermal_lag_correction(self, data, lat, lon):
        ''' Applies a thermal lag correction

        Parameters
        ----------
        data : dictionary
               a data dictionary
        lat : float
              mean latitude of the deployment
        lon : float
              mean longitude of the deployment
        
        Returns
        -------
        None

        The methed updates the conductivity in the data dictionary
        '''
        alpha, beta = self.thermal_lag_coefs
        tl_data = data.copy()
        ctd_tl = ThermalLag(tl_data)
        ctd_tl.interpolate_data(dt = 1)
        ctd_tl.split_profiles()
        ctd_tl.apply_thermal_lag_correction(alpha, beta,
                                            Cparameter="C", Tparameter="T",
                                            lon=lon, lat=lat)
        Ccor = np.interp(data['time'], ctd_tl.data['time'], ctd_tl.data['Ccor'])/10
        data['C']=Ccor

    def add_density(self, data, lat, lon):
        ''' Adds density to the data dictionary

        data : dictionary
               a data dictionary
        lat : float
              mean latitude of the deployment
        lon : float
              mean longitude of the deployment
        
        Returns
        -------
        None

        The method adds 'density' to the data dictionary
        '''
        C=data['C']
        T=data['T']
        D=data['D']
        density = easy_gsw.density_from_C_t(C, T, D, lon, lat)
        data['density']=density
        
    def calibrate_segment(self, glider_model, fns, min_depth, max_depth, parameters):
        ''' Calibrate the model for a data segment

        Not to be called directly
        '''
        data = self.get_data_dictionary(filenames=fns)
        glider_model.set_input_data(**data)
        glider_model.OR(data['pressure']*10<min_depth)
        glider_model.OR(data['pressure']*10>max_depth)
        calibration_result = glider_model.calibrate(*parameters, verbose=True)
        return calibration_result
    
        
    def calibrate(self, glider_model,
                  min_depth=10,
                  max_depth=40, parameters=["Cd0", "Vg"]):
        ''' Calibrate the glider model

        Parameters
        ----------
        glider_model : gliderflight GliderModel
                       a flight model, either steady state or dynamic.
        min_depth    : float
                       minimum depth
        max_depth    : float
                       maximum depth
        parameters   : list of parameter strings
                       names of model coefficients that should be optimised.
        
        Returns
        -------
        d : dictionary of array of floats for t (time) and parameters
            time is centred time of the calibration intervals
                
        The calibration method minimises a cost function. All data points that have
        a depth shallower than min_depth or a depth deeper than max_depth are excluded.
        '''
       
        binned_filenames = self.get_filename_list_per_interval()
        N_segments = len(binned_filenames)

        d = dict([(t, np.zeros(N_segments, float)) for t in parameters + ["t"]])

        for i, (tm, fns) in enumerate(binned_filenames):
            r = self.calibrate_segment(glider_model, fns, min_depth, max_depth, parameters)
            d['t'][i] = tm
            for j, p in enumerate(parameters):
                d[p][i] = glider_model.__dict__[p]
        return d

    def get_data_dictionary(self, pattern=None, filenames=[]):
        if not pattern is None:
            filenames = dbdreader.DBDList(glob.glob(pattern))
            filenames.sort()
        key = "".join(filenames)
        try:
            data = self.data_cache[key]
        except KeyError:
            pass
        else:
            print("Returning cached data")
            return data
        # if we get here, there were no data in the cache.
        dbds = dbdreader.MultiDBD(pattern=pattern, filenames=filenames, include_paired=True)
        tmp = dbds.get_sync("sci_ctd41cp_timestamp", "sci_water_cond sci_water_temp sci_water_pressure m_pitch m_ballast_pumped m_heading".split())
        t, tctd, C, T, P, pitch, buoyancy_change, heading = tmp.compress(tmp[2]>0, axis=1)
        _, lat, lon = dbds.get_sync("m_gps_lat", ["m_gps_lon"])
        lat = np.median(lat)
        lon = np.median(lon)
        a, b = self.pitch_correction_coefs
        pitch = a*pitch + b
        data = dict(time = tctd,
                    pressure = P,
                    C = C*10,
                    T = T,
                    D = P*10,
                    pitch = pitch,
                    buoyancy_change=buoyancy_change,
                    heading = heading)
        #self.thermal_lag_correction(data, lat, lon)
        self.add_density(data, lat, lon)
        dbds.close()
        self.data_cache[key] = data
        return data
    
    def compute_glider_velocity(self, glider_model, coef_functions={}):
        p = self.get_filename_pattern()
        data = self.get_data_dictionary(pattern=p)
        for k, f in coef_functions.items():
            glider_model.define(k=f)
        model_result = glider_model.solve(data)
        Modelresultxtd = namedtuple("Modelresultxtd", "t u w U alpha pitch ww heading".split())
        model_resultxtd = Modelresultxtd(*model_result, data['heading'])
        return model_resultxtd 
    
    def construct_ifun(self, calibration_result, parameter):
        t = calibration_result['t']
        x = calibration_result[parameter]
        return interp1d(t, x, bounds_error=False, fill_value=(x.min(), x.max()))


if 0:
    import gliderflight
    DM = gliderflight.DynamicCalibrate(rho0=1006, dt=0.1)
    GM = gliderflight.SteadyStateCalibrate(rho0=1006)
    GM.define(ah=3.8, Cd1=10.5, mg=70.1, Vg=70/1006, Cd0=0.15)


    pitch_correction_coefs = (0.8308, -0.006)
    mc = DeploymentCalibrator(glider='comet', path="/home/lucas/gliderdata/latvia201711_test/hd",
                              interval=86400,
                              pitch_correction_coefs=pitch_correction_coefs)


    calibration_results = mc.calibrate(glider_model=GM, parameters="Cd0 Vg".split(), min_depth=10, max_depth=40)
    DM.copy_settings(GM)
    DM.define(Vg=calibration_results['Vg'].mean())
    calibration_results = mc.calibrate(glider_model=GM, parameters="Cd0".split(), min_depth=10, max_depth=40)

    coef_functions = dict(Cd0=mc.construct_ifun(calibration_results, 'Cd0'))

    model_result = mc.compute_glider_velocity(GM, coef_functions)
