# from scipy.interpolate import RectBivariateSpline as Bi
import numpy as np


# TP profile configuration inputs
# 1) Number of central lattitude points
# 2) Number of pressure level points
# 3) Temperature grid

# Disc Averaging configuration inputs
# 1) Triplets: lattitude, longitude, emission zenith, weight

# Desired output
# weight, emission zenith, TP profile, abundance


"""
# model input TP grid
TP_lon_pres = np.array([
  [1000, 2000, 2500, 1500],
  [800, 1800, 2300, 1300],
  [700, 1700, 2200, 1200],
  [600, 1600, 2100, 1100],
])

# lat, lon, stellar zenith, emission zenith, stellar azimuth, weight
disc_table =  np.array([[0.00000000e+00, 2.15175527e+01, 4.26667143e+01, 6.23131549e+01,
         7.34273121e+01, 6.23131549e+01, 4.26667143e+01, 2.15175527e+01,
         0.00000000e+00, 0.00000000e+00, 2.22415968e+01, 3.77666255e+01,
         3.77666255e+01, 2.22415968e+01, 0.00000000e+00, 0.00000000e+00],
        [7.34273121e+01, 7.21455550e+01, 6.71756517e+01, 5.21295595e+01,
         0.00000000e+00, 3.07870440e+02, 2.92824348e+02, 2.87854445e+02,
         2.86572688e+02, 4.00880859e+01, 3.42536146e+01, 1.45799408e+01,
         3.45420059e+02, 3.25746385e+02, 3.19911914e+02, 0.00000000e+00],
        [1.06572688e+02, 1.06572688e+02, 1.06572688e+02, 1.06572688e+02,
         1.06572688e+02, 1.06572688e+02, 1.06572688e+02, 1.06572688e+02,
         1.06572688e+02, 1.39911914e+02, 1.39911914e+02, 1.39911914e+02,
         1.39911914e+02, 1.39911914e+02, 1.39911914e+02, 1.80000000e+02],
        [7.34273121e+01, 7.34273121e+01, 7.34273121e+01, 7.34273121e+01,
         7.34273121e+01, 7.34273121e+01, 7.34273121e+01, 7.34273121e+01,
         7.34273121e+01, 4.00880859e+01, 4.00880859e+01, 4.00880859e+01,
         4.00880859e+01, 4.00880859e+01, 4.00880859e+01, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.80000000e+02],
        [1.92259500e-02, 3.84519000e-02, 3.84519000e-02, 3.84519000e-02,
         3.84519000e-02, 3.84519000e-02, 3.84519000e-02, 3.84519000e-02,
         1.92259500e-02, 5.62805200e-02, 1.12561030e-01, 1.12561030e-01,
         1.12561030e-01, 1.12561030e-01, 5.62805200e-02, 1.29579660e-01]]))
"""

mock_pressure_grid = np.array([1.00000000e+01, 6.15848211e+00, 3.79269019e+00, 2.33572147e+00,
       1.43844989e+00, 8.85866790e-01, 5.45559478e-01, 3.35981829e-01,
       2.06913808e-01, 1.27427499e-01, 7.84759970e-02, 4.83293024e-02,
       2.97635144e-02, 1.83298071e-02, 1.12883789e-02, 6.95192796e-03,
       4.28133240e-03, 2.63665090e-03, 1.62377674e-03, 1.00000000e-03])

mock_longitude_grid = np.array([0,90,180,270])

mock_TP_grid = np.array([[3000.        , 2000.        , 2500.        , 2800.        ],
                        [2894.73684211, 1973.68421053, 2431.57894737, 2710.52631579],
                        [2789.47368421, 1947.36842105, 2363.15789474, 2621.05263158],
                        [2684.21052632, 1921.05263158, 2294.73684211, 2531.57894737],
                        [2578.94736842, 1894.73684211, 2226.31578947, 2442.10526316],
                        [2473.68421053, 1868.42105263, 2157.89473684, 2352.63157895],
                        [2368.42105263, 1842.10526316, 2089.47368421, 2263.15789474],
                        [2263.15789474, 1815.78947368, 2021.05263158, 2173.68421053],
                        [2157.89473684, 1789.47368421, 1952.63157895, 2084.21052632],
                        [2052.63157895, 1763.15789474, 1884.21052632, 1994.73684211],
                        [1947.36842105, 1736.84210526, 1815.78947368, 1905.26315789],
                        [1842.10526316, 1710.52631579, 1747.36842105, 1815.78947368],
                        [1736.84210526, 1684.21052632, 1678.94736842, 1726.31578947],
                        [1631.57894737, 1657.89473684, 1610.52631579, 1636.84210526],
                        [1526.31578947, 1631.57894737, 1542.10526316, 1547.36842105],
                        [1421.05263158, 1605.26315789, 1473.68421053, 1457.89473684],
                        [1315.78947368, 1578.94736842, 1405.26315789, 1368.42105263],
                        [1210.52631579, 1552.63157895, 1336.84210526, 1278.94736842],
                        [1105.26315789, 1526.31578947, 1268.42105263, 1189.47368421],
                        [1000.        , 1500.        , 1200.        , 1100.        ]])


def interpolate_to_location(longitude,lattitude,longitude_grid,profile_grid,index=0.25):
    """
    Interpolate a grid of profiles specified at a list of longitudes to a location
    specified by input longitude and lattitude, assuming a parametrised lattitude dependence.

    Parameters
    ----------
    longitude : real
        in degrees
    lattitude : real
        in degrees
    longitude_grid : ndarray
        longitudes for which the profiles are specified
    profile_grid : ndarray
        The profile for each of the longitude in the longitude grid
        dimension = pressure x longitude
    index : real
        parameter specifying the lattitudinal dependen of a profile; assume profiles
        varies as cos^index(lattitude)

    Returns
    -------
    interpolated_profile : ndarray
        Interpolated profile at the location specified by input lattitude and longitude.
    """
    longitude = longitude%360
    assert lattitude <=90 and lattitude >=0
    # finder lower and upper bounds in the longitude array
    low_index = np.where(longitude<=longitude_grid)[0][-1]
    high_index = np.where(longitude>=longitude_grid)[0][0]

    low_longitude = longitude_grid[low_index]
    high_longitude = longitude_grid[high_index]
    low_profile = profile_grid[:,low_index]
    high_profile = profile_grid[:,high_index]

    interpolated_profile = low_profile*(high_longitude-longitude)/(high_longitude-low_longitude)\
        + high_profile*(longitude-low_longitude)/(high_longitude-low_longitude)

    interpolated_profile = interpolated_profile * np.cos(np.pi*lattitude/180) ** index

    return interpolated_profile

#def disc_average(phase,longitude_grid,profile_grid,index=0.25):


"""
def disc_integrate(lattitudes,longitudes,emission_angle,weight):

    radiance = 0
    for lat in lattitudes:
"""


"""
def interp_pressure_longitude(longitude,pressure,pressure_grid,longitude_grid,T_pres_long):

    f = interp2d(pressure_grid,longitude_grid,T_pres_long)
    interp_T_pres_long = f(pressure,longitude)

    return interp_T_pres_long

def interp_location(longitude,lattitude,pressure,pressure_grid,longitude_grid,T_pres_long,index=0.25):

    interp_T_pres_long = interp_pressure_longitude(longitude,pressure,pressure_grid,longitude_grid,T_pres_long)

    interp_T_pres_long_lat = np.zeros((len(pressure),len(longitude),len(lattitude)))

    for index, lat in enumerate(lattitude):
        interp_T_pres_long_lat[:,index,index] = interp_T_pres_long[:,index] * np.cos(lat*np.pi/180) ** index

    return interp_T_pres_long_lat

def disc_average(interp_T_pres_long_lat,long_lat_weight,nav):

    for i in range(nav):
        TP = interp_T_pres_long_lat[:,i,i]
"""

"""
def model_30(

        input_longitude, input_pressure, input_lattitude,
             longitude_grid, pressure_grid, temperature_grid, index=0.25):
    f = interp2d(longitude_grid,pressure_grid,temperature_grid)
    interp_temperature_grid = f(input_longitude, input_pressure)
    output_lon_lat_pre_grid = np.zeros((len(input_longitude),len(input_lattitude),len(input_pressure)))
    for index, lat in enumerate(input_lattitude):
        output_lon_lat_pre_grid[index,index,:] = interp_temperature_grid[index,:] * np.cos(lat*np.pi/180)**index
    return output_lon_lat_pre_grid
"""

input_lattitude, input_longitude = np.loadtxt('test_angles.dat',unpack=True,usecols=(0,1))
temperature_grid = np.loadtxt('test_TP.dat')
