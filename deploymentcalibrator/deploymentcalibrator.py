from collections import namedtuple
import glob
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint

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
                 pitch_correction_coefs=(1.,0.),
                 **kwds):
        self.glider = glider
        self.path = path
        self.interval = interval
        self.pitch_correction_coefs = pitch_correction_coefs
        self.thermal_lag_coefs = thermal_lag_coefs
        self.dbd_kwds = kwds
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

        The method adds 'density' to the data dictionary
        '''
        C=data['C']
        T=data['T']
        D=data['D']
        density = easy_gsw.density_from_C_t(C, T, D, lon, lat)
        data['density']=density


    def rate_model(self, y, t, tau, pump_on_fun):
        """Define the right-hand side of equation dy/dt = a*y"""
        delta = (pump_on_fun(t)-y)
        if delta>0:
            _tau=1
        else:
            _tau=tau
        f = delta/_tau
        return f

        
    def is_pumping(self, t, b, tau, limit=0.1):
        ''' Is the glider pumping, allowing for a fade out time of tau? 

        Parameters
        ----------
        t : array
            time in seconds
        b : array
            buoyancy drive in cc
        tau : float
              time allowed for the effect of pumping to fade
        limit : float (default 0.1)
                threshold to determine when the pump is moving (db/dt > limit)
        
        Returns
        -------
        y : array
            indication of the pump moving, where 0 <= y <= 1. At y=0.5 we would have
            waited about tau seconds after switching of the pump.
        '''
        dbdt = np.gradient(b)/np.gradient(t)
        pump_on = (np.abs(dbdt)>limit).astype(int)
        pump_on_fun = interp1d(t, pump_on, fill_value=0, bounds_error=False)
        y0=0.
        y = odeint(self.rate_model, y0, t, hmax=5, args=(tau,pump_on_fun))
        return y.squeeze()


    def calibrate_segment(self, glider_model, fns, parameters, min_depth=None, max_depth=None, seconds_to_discard_after_pumping=None):
        ''' Calibrate the model for a data segment

        Not to be called directly
        '''
        data = self.get_data_dictionary(filenames=fns)
        glider_model.set_input_data(**data)
        if not min_depth is None:
            glider_model.OR(data['pressure']*10<min_depth)
        if not max_depth is None:
            glider_model.OR(data['pressure']*10>max_depth)
        if not seconds_to_discard_after_pumping is None:
            pumping = self.is_pumping(data['time'], data['buoyancy_change'], seconds_to_discard_after_pumping)
            glider_model.OR(pumping>0.5)
        calibration_result = glider_model.calibrate(*parameters, verbose=True)
        return calibration_result
    
        
    def calibrate(self, glider_model,parameters=["Cd0", "Vg"],
                  min_depth=10,
                  max_depth=None,
                  seconds_to_discard_after_pumping=None):
    
        '''Calibrate the glider model

        Parameters
        ----------
        glider_model : gliderflight GliderModel
                       a flight model, either steady state or dynamic.
        parameters   : list of parameter strings
                       names of model coefficients that should be optimised.
        min_depth    : float or None
                       minimum depth
        max_depth    : float or None
                       maximum depth
        seconds_to_discard_after_pumping : float or None
                       all data points gathered within this amount of
                       seconds after the pump stops starting from the
                       beginning of pummping are discarded.
        
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
            print("Processing segment %d of a total of %d."%(i+1, N_segments))
            try:
                r = self.calibrate_segment(glider_model, fns, parameters, min_depth, max_depth, seconds_to_discard_after_pumping)
            except ValueError as e:
                if e.args[0] == "All selected data files were banned.":
                    continue
                else:
                    raise(e)
            d['t'][i] = tm
            for j, p in enumerate(parameters):
                d[p][i] = glider_model.__dict__[p]
        return d

    def get_data_dictionary(self, pattern=None, filenames=[]):
        '''Reads dbd files and puts selected data into a dictionary

        Parameters
        ----------
        pattern   : None | string
                  A path, possibly containing wildcards, to select one or more files
        filenames : None | list of strings
                  A list of filenames

        Returns
        -------
        data : dictionary with data


        This method reads dbd files and stores selected parameters
        into a dictionary. The filenames to be read can be specified,
        either by a list of filenames or a pattern with wild
        cards. The method recognizes when a dictionary is requested
        multiple times for the same set of files, and returns cached
        data in such instance.
        '''
        if not pattern is None:
            filenames = dbdreader.DBDList(glob.glob(pattern))
            filenames.sort()
        key = "".join(filenames)
        try:
            data = self.data_cache[key]
        except KeyError:
            pass
        else:
            return data
        # if we get here, there were no data in the cache.
        dbds = dbdreader.MultiDBD(pattern=None, filenames=filenames, include_paired=True,
                                  **self.dbd_kwds)
        tmp = dbds.get_sync("sci_ctd41cp_timestamp", "sci_water_cond sci_water_temp sci_water_pressure m_pitch".split())
        t, tctd, C, T, P, pitch = tmp.compress(tmp[2]>0, axis=1)

        # for extracting the heading we have to proceed with a bit of
        # care. Simple interpolation (by including m_heading in the
        # get_sync() method above, can result in jumpy values around
        # North crossings. This creates problems when the glider is
        # moving north and many crossings occur during a profile.

        t_hdg, hdg = dbds.get("m_heading")
        x = np.cos(hdg)
        y = np.sin(hdg)
        xi = np.interp(tctd, t_hdg, x)
        yi = np.interp(tctd, t_hdg, y)
        heading = np.arctan2(yi, xi)

        # Let's figure out which buoyancy variable to use. Try m_ballast_pumped first.
        _buoyancy_change = dbds.get("m_ballast_pumped")
        if np.allclose(_buoyancy_change[1], 0, rtol=1e-5):
            _buoyancy_change = dbds.get("m_de_oil_vol")
        # interpolate to t
        buoyancy_change = np.interp(t, *_buoyancy_change)
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
        '''Compute glider velocity using given glider model and, optionally,
           interpolating functions for model coefficients.

        Parameters
        ----------
        glider_model : an instance of a gliderflight glider model 
                       (SteadyStateGliderModel or DynamicGliderModel)
        coef_functions : a dictionary of interpolating functions
        
        Returns
        -------
        model_result : a named tuple with glider flight model results

        This method takes a glider flight model and computes, with the given settings
        the glider flight velocities. An optional dictionary with interpolating functions
        for one or more coefficients can be supplied. These functions override the constant
        values of named coefficients. 

        The result returned is a named tuple, as defined in the gliderflight module, but with
        heading as extra field, so that the results retured contain

        t : time in seconds since epoch
        u : horizontal velocity through water m/s
        w : vertical velocity through water m/s
        U : incident water velocity m/s
        alpha : angle of attack rad
        pitch : pitch rad
        ww : vertical water velocity m/s (dh/dt - w)
        heading : heading rad
        '''
        p = self.get_filename_pattern()
        data = self.get_data_dictionary(pattern=p)
        for k, f in coef_functions.items():
            glider_model.define(k=f)
        model_result = glider_model.solve(data)
        Modelresultxtd = namedtuple("Modelresultxtd", "t u w U alpha pitch ww heading depth".split())
        model_resultxtd = Modelresultxtd(*model_result, data['heading'], data['pressure']*10)
        return model_resultxtd 
    
    def construct_ifun(self, calibration_result, parameter):
        '''Constructor of interpolating function

        Parameters
        ----------
        calibration_result : dictionary with results from the calibrate() method
        parameter : name of parameter to construct an interpolating function for
        
        Returns
        -------
        ifun : interpolating function (scipy.interp1d)

        This method can be used to construct a suitable interpolating
        function for the coef_functions dictionary that can be
        supplied to the method compute_glider_velocity(). It takes the
        input from the calibration() method, which calibrates the data
        in time intervals, producing a time series for one or more
        calibrated model coefficients.
        '''
        t = calibration_result['t']
        x = calibration_result[parameter]
        ifun = interp1d(t, x, bounds_error=False, fill_value=(x.min(), x.max()))
        return ifun
