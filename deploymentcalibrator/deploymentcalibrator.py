from collections import namedtuple, defaultdict
import glob
import os
import pickle
from logging import getLogger

import arrow
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint

import dbdreader
from profiles.ctd import ThermalLag, iterprofiles

from . import easy_gsw

logger = getLogger("DeploymentCalibrator")

Modelresultxtd = namedtuple("Modelresultxtd", "t u w U alpha pitch ww z heading depth lat lon density SA CT pot_density buoyancy_change C T P Craw".split())

class DeploymentCalibrator(object):
    '''A class to calibrate the glider flight model for a full deployment

    This class provides an interface to calibrate the glider flight
    model for a full deployment.  The calibrated model can be used to
    generate glider flight velocities accounting for time varying
    glider flight coefficients (notably the drag coefficient).

    Parameters
    ----------

    glider  : str
              name of glider
    path    : str
              path where glider files (dbd and ebd) reside
    interval: float
              time interval for which glider data files are aggregated to optimize
              for glider flight coefficients.

    thermal_lag_coefs : None or tuple of floats (alpha, beta, tau)
             if the ThermalLag class can be imported, the CTD data will be corrected
             for thermal lag effects.

    pitch_correction_coefs: None or tuple of float 
             the pitch reported by the glider is adjusted with the two coefficients
             given. The first coefficient is a scaling factor, the second an offset.

    segments : None or int or tuple
        None: all segments are processed
        int:  positive first n segments; negative last n segments
        tuple: all segments between first and second element are processed.

    neutral_buoyancy_drive : float (0.0)
        assumed buoyancy drive when glider is neutral. A non-zero value would be used for a
        glider with an extended pump and ballasted for a non-zero pump value, intentionally or
        nonintentionally. Effects only the calibration result when up and down cast samples are
        weighted.

    **kwds :
        optional keywords passed to MultiDBD()


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
                 thermal_lag_coefs=None, 
                 pitch_correction_coefs=None,
                 segments=None,
                 neutral_buoyancy_drive=0,
                 **kwds):
        self.glider = glider
        self.path = path
        self.interval = interval
        self.pitch_correction_coefs = pitch_correction_coefs
        self.thermal_lag_coefs = thermal_lag_coefs
        self.segments = segments
        self.neutral_buoyancy_drive = neutral_buoyancy_drive
        self.dbd_kwds = kwds
        self.data_cache = {}
        self.info = {}
        
    def get_filename_pattern(self):
        ''' Get a pattern for matching glider data filenames (assuming dbd files).
        
        Returns
        -------
            p : pattern for consumption by glob.glob()
        '''
        p = os.path.join(self.path, "{}*.dbd".format(self.glider))
        return p
    
    def get_filename_list_per_interval(self, interval=None):
        ''' Bin filenames
        
        Parameters
        ----------
        interval : float or None
            interval duration. If None, self.interval is used.

        Returns
        -------
        binned_filenames : a list of tuples, each containing the average time of an interval, and 
                           a list of filenames that have their opening times falling in the this interval.

        This method conveniently bins all filenames based on opening times. The interval is set in the 
        constructor of this class (usually something like 1 day).
        '''
        interval = interval or self.interval
        p = self.get_filename_pattern()
        ps = dbdreader.DBDPatternSelect()
        binned_filenames = ps.bins(pattern=p, binsize=interval)
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
        if self.thermal_lag_coefs is None:
            return
        alpha, beta, tau = self.thermal_lag_coefs
        tl_data = data.copy()
        ctd_tl = ThermalLag(tl_data)
        ctd_tl.interpolate_data(dt = 1)
        #ctd_tl.split_profiles()
        if tau:
            ctd_tl.apply_short_time_mismatch(tau)
        ctd_tl.apply_thermal_lag_correction(alpha, beta,
                                            Cparameter="C", Tparameter="T",
                                            lon=lon, lat=lat)
        Ccor = np.interp(data['time'], ctd_tl.data['time'], ctd_tl.data['Ccor'])
        # Fix any leading 0s
        idx = np.where(Ccor!=0)[0]
        data['C'][idx]=Ccor[idx]
        data['Craw'] = tl_data['C']
        
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
        density, pDensity = easy_gsw.density_from_C_t(C, T, D, lon, lat)
        SA = easy_gsw.SA_from_C_t(C, T, D, lon, lat)
        CT = easy_gsw.CT_from_C_t(C, T, D, lon, lat)
        data['density']=density
        data['pot_density']=pDensity
        data['density']=density
        data['SA'] = SA
        data['CT'] = CT

    def rate_model(self, y, t, tau, pump_on_fun):
        """Define the right-hand side of equation dy/dt = a*y"""
        delta = (pump_on_fun(t)-y)
        if delta>0:
            _tau=1
        else:
            _tau=tau
        f = delta/_tau
        return f

        
    def is_pumping(self, t, b, tau, limit=0.75):
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
        is_pumping : array of bool
        '''
        dbdt = np.gradient(b)/np.gradient(t)
        pump_on = (np.abs(dbdt)>limit).astype(int)
        pump_on_fun = interp1d(t, pump_on, fill_value=0, bounds_error=False)
        y0=0.
        y = odeint(self.rate_model, y0, t, hmax=5, args=(tau,pump_on_fun)).squeeze()
        is_pumping = y>0.36
        return is_pumping


    def __balance_up_and_down_casts(self, gm):
        threshold = 50 #cc
        minimum_datapoints=500
        
        c_up = np.logical_and(gm.input_data['buoyancy_change']>self.neutral_buoyancy_drive + threshold, ~gm.mask)
        c_dw = np.logical_and(gm.input_data['buoyancy_change']<self.neutral_buoyancy_drive - threshold, ~gm.mask)

        w_up = c_up.compress(c_up).shape[0]
        w_dw = c_dw.compress(c_dw).shape[0]
        if w_up<minimum_datapoints or w_dw<minimum_datapoints:
            weights = None
        else:
            w_up = w_up/(w_up+w_dw)
            w_dw = 1 - w_up
            weights = w_up * c_up.astype(int) + w_dw * c_dw.astype(int)
        return weights
        
        
    def create_profile_based_mask(self, data):
        ps = iterprofiles.ProfileSplitter(data)
        ps.split_profiles()
        mask = np.ones(data["time"].shape, int)
        slices = ps.get_casts().slices
        for s in slices:
            mask[s] = False
        return mask.astype(bool), ps.nop
        
    def calibrate_segment(self, glider_model, fns, parameters, min_depth=None, max_depth=None,
                          seconds_to_discard_after_pumping=None, balance_up_down=False, constraints=('dhdt')):
        ''' Calibrate the model for a data segment

        Not to be called directly
        '''
        try:
            data = self.get_data_dictionary(filenames=fns)
        except dbdreader.DbdError as e:
            errno, mesg = e.args
            if errno==dbdreader.DBD_ERROR_ALL_FILES_BANNED:
                data = None # no data to return
            else:
                # something unexpected happened. Raise the error again
                raise e
        if data is None: # no data, return None
            return None
        
        glider_model.set_input_data(**data)

        glider_model.mask, n_profiles = self.create_profile_based_mask(data)

        if not min_depth is None:
            glider_model.OR(data['pressure']*10<min_depth)
        if not max_depth is None:
            glider_model.OR(data['pressure']*10>max_depth)
        if not seconds_to_discard_after_pumping is None:
            is_pumping = self.is_pumping(data['time'], data['buoyancy_change'], seconds_to_discard_after_pumping)
            glider_model.OR(is_pumping)
        if balance_up_down:
            weights = self.__balance_up_and_down_casts(glider_model)
        else:
            weights = None
        # if all data are masked, return None
        if np.all(glider_model.mask):
            logger.warning(f"All data in this segment are masked. Max depth: {glider_model.input_data['pressure'].max()*10:.1f} m. Number of profiles found: {n_profiles}.")
            return None
        logger.info(f"Calibrating for {n_profiles} profiles.")
        calibration_result = glider_model.calibrate(*parameters, constraints=constraints, weights=weights, verbose=True)
        return calibration_result
    
        
    def calibrate(self, glider_model,parameters=["Cd0", "Vg"],
                  calibrated_parameters={},
                  min_depth=10,
                  max_depth=None,
                  seconds_to_discard_after_pumping=None,
                  balance_up_down = False,
                  output_filename=None,
                  constraints=('dhdt')):
        '''Calibrate the glider model

        Parameters
        ----------
        glider_model : gliderflight GliderModel
                       a flight model, either steady state or dynamic.
        parameters   : list of parameter strings
                       names of model coefficients that should be optimised.
        calibrated_parameters : dict of parameter/values
                       Optionally provides calibration parameters obtained in previous cycle
                       as starting point.
        min_depth    : float or None
                       minimum depth
        max_depth    : float or None
                       maximum depth
        seconds_to_discard_after_pumping : float or None
                       all data points gathered within this amount of
                       seconds after the pump stops starting from the
                       beginning of pummping are discarded.
        balance_up_down : bool
                       If set True, then the calibration accounts for weighting the 
                       up and down cast samples evenly.
        output_filename : string or None
                       filename for pickled file to store intermediate results.

        Returns
        -------
        d : dictionary of array of floats for t (time) and parameters
            time is centred time of the calibration intervals
                
        The calibration method minimises a cost function. All data points that have
        a depth shallower than min_depth or a depth deeper than max_depth are excluded.

        '''
        undef_params = glider_model.undefined_parameters()
        if undef_params:
            s = "All glider model parameters must be defined.\n"
            s +="Missing parameters: "+ " ".join(undef_params)
            raise ValueError(s)
        binned_filenames = self.get_binned_filenames()
        N_segments = len(binned_filenames)

        d = defaultdict(lambda : list())

        for i, (tm, fns) in enumerate(binned_filenames):
            #if i+1 >3:
            #    continue
            mesg = f"({self.glider}) Processing segment {i+1}/{N_segments}. "
            T0 = arrow.get(tm)
            T1 = arrow.get(tm+self.interval)
            fmt="HH:MM DD/MM/YY"
            mesg += f"{T0.format(fmt)} -> {T1.format(fmt)} ({len(fns)})"
            logger.info(mesg)
            if not fns:
                continue
            # set any calibrated parameters from a previous loop
            for k, v in calibrated_parameters.items():
                if k == 't':
                    continue # we should not set t (time)
                logger.info("Setting %s=%.3f from list."%(k,v[i]))
                glider_model.__dict__[k] = v[i]
            try:
                r = self.calibrate_segment(glider_model, fns, parameters, min_depth, max_depth,
                                           seconds_to_discard_after_pumping, balance_up_down, constraints)
            except ValueError as e:
                if e.args[0] == "All selected data files were banned.":
                    continue
                else:
                    raise(e)
            if r is None: # no calibration results returned because of no data.
                continue
            d['t'].append(tm)
            for j, p in enumerate(parameters):
                d[p].append(glider_model.__dict__[p])
        d = dict( (k, np.array(v)) for k,v in d.items()) # return a normal dictionary with lists converted to numpy arrays
        if output_filename:
            self.write_calibration_results(output_filename, d)
        return d
    
    
    def write_calibration_results(self, filename, d):
        ''' Writes calibration results dictionary to a pickled file
        
        Parameters
        ----------
        filename : string
            name of pickled file
        d        : dictionary
            dictionary with arrays of calibration results (computed so far)
        
        This method is mainly for diagnostics and to continue failed computations.
        '''
        if not filename.endswith(".pck"):
            filename+=".pck"
        with open(filename, "wb") as fp:
            pickle.dump(d, fp)
                
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
        data : dictionary with data or None

        None is returned if insufficient data.


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
        self.info['filenames'] = filenames
        if not len(filenames):
            raise ValueError(f'({self.glider}) Did not find any files for this glider.')
        try:
            data = self.data_cache[key]
            logger.info("Using cached data")
        except KeyError:
            pass
        else:
            return data
        # if we get here, there were no data in the cache.
        dbds = dbdreader.MultiDBD(pattern=None, filenames=filenames, complement_files=True,
                                  **self.dbd_kwds)
        self.info['missions'] = dbds.mission_list
        self.info['files_opened'] = dbds.filenames
        tctd, C, T, P, pitch = dbds.get_CTD_sync("m_pitch")
        logger.info(f"Opened {len(dbds.filenames)} files.")
        if not len(tctd) or tctd.ptp()<300: #no data or less than 5 minutes of data, return None
            logger.info("Skipping this segment for cabration due to insufficient data.")
            return None
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
        buoyancy_change = np.interp(tctd, *_buoyancy_change)
        t_gps, lat, lon = dbds.get_sync("m_gps_lat", "m_gps_lon")
        if not self.pitch_correction_coefs is None:
            a, b = self.pitch_correction_coefs
            pitch = a*pitch + b
        data = dict(time = tctd,
                    pressure = P,
                    C = C*10,
                    T = T,
                    D = P*10,
                    pitch = pitch,
                    buoyancy_change=buoyancy_change,
                    heading = heading,
                    lat = np.interp(tctd, t_gps, lat),
                    lon = np.interp(tctd, t_gps, lon))
        self.thermal_lag_correction(data, np.median(lat), np.median(lon))
        self.add_density(data, np.median(lat), np.median(lon))
        dbds.close()
        self.data_cache[key] = data
        return data

    def compute_glider_velocity(self, glider_model, coef_fun = {}):
        '''Compute glider velocity using given glider model and, optionally,
           interpolating functions for model coefficients.

        Parameters
        ----------
        glider_model : {SteadyStateGliderModel, DynamicGliderModel}
             an instance of a gliderflight glider model 

        coeficients : dictionary of arrays
            A dictionary with time and value arrays for all calibration parameters, for example
            coefficients = {'t':[...], 'Cd0':[...], 'Vg':[...]}
 
        
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
        # using cached data and string them together.
        binned_filenames = self.get_binned_filenames()
        data = None
        for i, (tm, fns) in enumerate(binned_filenames):
            try:
                _data = self.get_data_dictionary(filenames = fns)
            except ValueError:
                continue
            except dbdreader.DbdError as e:
                if e.value == dbdreader.DBD_ERROR_ALL_FILES_BANNED:
                    continue
                else:
                    raise(e)
            else:
                if _data is None:
                    continue
            if data is None:
                data = _data.copy()
            else:
                for k in data.keys():
                    data[k] = np.hstack((data[k], _data[k]))

        glider_model.mask=None
        glider_model.ensure_monotonicity(data)
        for k, f in coef_fun.items():
            glider_model.__dict__[k] = f(data["time"])
        model_result = glider_model.solve(data)
        # We have Craw only when the thermal correction is applied.
        if not 'Craw' in data:
            data['Craw'] = data['C']
        model_resultxtd = Modelresultxtd(*model_result, data['heading'], data['pressure']*10,
                                         data['lat'], data['lon'], data['density'], data['SA'],
                                         data['CT'], data['pot_density'], data['buoyancy_change'],
                                         data['C'], data['T'], data['pressure'], data['Craw'])
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

    def get_binned_filenames(self):
        ''' Returns binned filenames

        Filenames are binned conform the given interval duration. If a limit is set for the number
        of segements (self.max_segments), then the list is truncated accordingly.
        
        Returns
        -------
        list of string of filenames
        '''
        binned_filenames = self.get_filename_list_per_interval()
        N_segments = len(binned_filenames)
        if not self.segments is None:
            if isinstance(self.segments, tuple):
                s = slice(*self.segments)
                mesg = f"({self.glider}) Processing segments {self.segments[0]}-{self.segments[1]} out of {N_segments} only."
                binned_filenames = binned_filenames[s]
            elif self.segments > 0:
                mesg = f"({self.glider}) Processing segments first {self.segments} out of {N_segments} only."
                binned_filenames = binned_filenames[:self.segments]
            else:
                mesg = f"({self.glider}) Processing segments last {self.segments} out of {N_segments} only."
                binned_filenames = binned_filenames[-self.segments:]
            logger.info(mesg)
        return binned_filenames



