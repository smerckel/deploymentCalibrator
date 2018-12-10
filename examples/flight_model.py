import numpy as np
import gliderflight
from deploymentcalibrator import DeploymentCalibrator

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
