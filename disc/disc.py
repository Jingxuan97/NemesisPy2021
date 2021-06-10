#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this is the version with stellar angles.

"""
For an input orbital phase, the code ouputs locations on the disc (in terms of
latitude/longtitude pairs) and weights for interpolation and disc averaging.
Solar zenith and azimuth angles are also generated, but not currently used as
reflected sunlight is not included.

Note on geometry:
    We are interested in observing a close-orbiting transitting exoplanet.
    Therefore, we can make the following assumptions:
        1) We are observing the planet's orbit edge on;
        2) The planet is tidally locked, i.e. it's in synchronous rotation and always
            present the same hemisphere to the star.
    We define the north pole of the planet such that when viewed directly above the
    north pole the planet is rotating anticlockwise.

    The orbital phase is defined such that:
        0=nightside,
        45=quarter - dayside on left (when viewed from north pole )
        90=half-lit - dayside on left, (when viewed from north pole )
        180=dayside,
        270=half-lit - dayside on right, (when viewed from north pole )
        315=quarter - dayside on right. (when viewed from north pole )

    The latitude on the planet is defined such that the substellar point is 180,
    the antistellar point is 0, east is in the direction of the planet's self rotation.
"""

import numpy as np

def arctan(y,x):
    """
    Calculates the argument of point (x,y) in the range [0,2pi).

    Parameters
    ----------
    """
    if(x == 0.0):
        if(y == 0.0): ang=0.0
        else:
            if(y > 0.0): ang = 0.5*np.pi
            else: ang = 1.5*np.pi
        return ang
    ang=np.arctan(y/x)
    if(y > 0.0):
        if(x > 0.0): ang = ang # 1st quadrant
        else: ang = ang+np.pi # 2nd quadrant
    else:
        if(x > 0.0): ang = ang+2*np.pi # 4th quadrant
        else: ang = ang+np.pi # 3rd quadrant
    return ang

def rotatey(v, phi):
    """
    Rotate a vector v about the y-axis by an angle phi

    Parameters
    ----------
        v : ndarray
            A real 3D vector to rotate
        phi : real
            Angle to rotate the vector by (radians)

    Returns
    -------
        v1 : ndarray
            Rotated vector
    """
    a = np.zeros([3,3])
    a[0,0] = np.cos(phi)
    a[0,2] = np.sin(phi)
    a[1,1] = 1.
    a[2,0] = -np.sin(phi)
    a[2,2] = np.cos(phi)
    v1 = np.matmul(a,v)
    v1 = np.around(v1,10)
    return v1

def rotatez(v, phi):
    """
    Rotate a vector v about the z-axis by an angle phi

    Parameters
    ----------
        v : ndarray
            A real 3D vector to rotate
        phi : real
            Angle to rotate the vector by (radians)

    Returns
    -------
        v1 : ndarray
            Rotated vector
    """
    a = np.zeros([3,3])
    a[0,0] = np.cos(phi)
    a[0,1] = -np.sin(phi)
    a[1,0] = np.sin(phi)
    a[1,1] = np.cos(phi)
    a[2,2] = 1
    v1 = np.matmul(a,v)
    v1 = np.around(v1,10)
    return v1

def thetasol_gen_azi_latlong(xphase,rho,alpha):
    """
    Calculate the stellar zenith angle, stellar aziumuth angle, lattitude and
    longtitude for a point on a TRANSITING TIDALLY LOCKED planet's surface
    illuminated by its star at variable phase angle.

    Orbital phase (stellar phase angle) increases from 0 at primary transit
    to 180 at secondary eclipse.

    Planetary longitude is 180E at the substellar point and 0E at the antistellar
    point. East is in the direction of the planet's self-rotation.

    Note on the geometry used in this routine:
        Imagine a frame centred at the planet at the moment of primary transit.
        At this point, the stellar phase is 0.
        We are viewing the orbital plane edge on; WLOG assume the orbit
        is anticlockwise, then
            x-axis points towards 3 o'clock.
            y-axis points towards the star.
            z-axis points towards north.
            theta is measured from the z-axis conventionally.
            phi is measured anticlockwise from x-axis.
        Now imagine the star orbiting around the planet; our frame moves with the
        centre of the planet but is not rotating. In this routine we assume
        the planetary surface is a perfect spherical shell with a dimensionless
        radius 1.

    Parameters
    ----------
    xphase : real
        Stellar phase/orbital phase in degrees.
        0=parimary transit and increase to 180 at secondary eclipse.
    rho	: real
        Fractional radius of required position on disc.
    alpha	: real
        Position angle of point on visible disc (degrees), measured
        anticlockwise from 3 o'clock position

    Returns
    -------
    thetasol : real
        Computed solar zenith angle (radians)
    azi	: real
        Computed solar azimuth angle (radians). Uses convention that
        forward scatter = 0.
    lat	: real
        Latitude
    lon	: real
        Longitude
    """
    xphase = xphase%360
    dtr = np.pi/180.
    assert rho <=1, "Fractional radius should be less or equal to 1"

    ### calculate solar direction vector using spherical polars (r,theta,phi)
    thetas = np.pi/2.               # star lie in planet's equitorial plane
    phi_star = 90.0 + xphase
    x1 = np.sin(thetas)*np.cos(phi_star*dtr)
    y1 = np.sin(thetas)*np.sin(phi_star*dtr)
    z1 = np.cos(thetas)
    v1 = np.array([x1,y1,z1])

    ### calculate sample position vector using spherical polars (r=1,theta,phi)
    theta = np.arccos(rho*np.sin(alpha*dtr)) # planetary zenith angle of spot on surface
    if np.sin(theta) != 0.0:
        cos_phi = rho*np.cos(alpha*dtr)/abs(np.sin(theta)) # changed
        phi = (-np.arccos(cos_phi))%(2*np.pi) # azimuth angle of spot on surface / on our side
    else:
        phi = 0.0 # sin(theta) = 0 at north polt
    x2 = np.sin(theta)*np.cos(phi)
    y2 = np.sin(theta)*np.sin(phi)
    z2 = np.cos(theta)
    v2 = np.array([x2,y2,z2])

    ### calculate angle between solar position vector and local normal
    # i.e. thetasol solar zenith angle
    inner_product = np.sum(v1*v2)
    thetasol = np.arccos(inner_product)
    thetasol = np.around(thetasol, 10)

    ### calculate latitude and longitude of the spot
    # (sub-stellar point = 180E, anti-stellar point = 0E, longtitudes in the direction of self-rotation)
    lat = np.around(90.-theta*180/np.pi, 10)
    lon = (phi/dtr - (phi_star+180))%360

    ### calculate emission viewing angle direction vector (-y axis) (Observer direction vecto)
    x3 = 0.
    y3 = -1.0
    z3 = 0.0
    v3 = np.array([x3,y3,z3])

    ### calculate azimuth angle
    # Rotate frame clockwise by phi about z (v2 is now x-axis)
    v1A=rotatez(v1,-phi)
    v2A=rotatez(v2,-phi)
    v3A=rotatez(v3,-phi)

    # Rotate frame clockwise by theta about y (v2 is now z-axis )
    v1B=rotatey(v1A,-theta)
    v2B=rotatey(v2A,-theta)
    v3B=rotatey(v3A,-theta)

    # thetsolB=np.arccos(v1B[2])
    # thetobsB=np.arccos(v3B[2])
    phisolB=arctan(v1B[1], v1B[0])
    phiobsB=arctan(v3B[1], v3B[0])

    azi = abs(phiobsB-phisolB)
    if(azi > np.pi):
        azi=2*np.pi-azi

    # Ensure azi meets convention where azi=0 means forward-scattering
    azi = np.pi-azi
    return thetasol, azi, lat, lon

def subdiscweightsv3(xphase, nmu=3):
    """
    Python routine for setting up geometry and weights for observing a planet
    at a variable stellar phase angle xphase.

    Code splits disc into a number of rings using Gauss-Lobatto quadrature and then
    does azimuth integration using trapezium rule.

    Orbital phase (stellar phase angle) increases from 0 at primary transit
    to 180 at secondary eclipse.

    Planetary longitude is 180E at the substellar point and 0E at the antistellar
    point. East is in the direction of the planet's self-rotation.

    Parameters
    ----------
    xphase : real
        Stellar phase/orbital phase in degrees.
        0=parimary transit and increase to 180 at secondary eclipse.
    nmu	: integer
        Number of zenith angle ordinates

    Output variables
    nav	: integer
        Number of FOV points
    wav	: ndarray
        FOV-averaging table:
        0th row is lattitude, 1st row is longitude, 2nd row is stellar zenith
        angle, 3rd row is emission zenith angle, 4th row is stellar azimuth angle,
        5th row is weight.
    """
    assert nmu <=5, "Currently cannot do more than 5 quadrature rings"
    xphase = xphase%360
    dtr = np.pi/180
    delR = 1./nmu
    nsample = 1000             # large array size to hold calculations
    tablat = np.zeros(nsample) # latitudes
    tablon = np.zeros(nsample) # longitudeds
    tabzen = np.zeros(nsample) # zenith angle in quadrature scheme
    tabsol = np.zeros(nsample) # solar zenith angle
    tabazi = np.zeros(nsample) # solar azimuth angle (scattering phase angle?)
    tabwt = np.zeros(nsample)  # weight of each sample

    if nmu == 2:
        mu = [0.447213595499958,1.000000]                   # cos zenith angle
        wtmu = [0.8333333333333333,0.166666666666666666]    # corresponding weights
    if nmu == 3:
        mu = [0.28523151648064509,0.7650553239294646,1.0000]
        wtmu = [0.5548583770354863,0.3784749562978469,0.06666666666666666]
    if nmu == 4:
        mu = [0.2092992179024788,0.5917001814331423,0.8717401485096066,1.00000]
        wtmu = [0.4124587946587038,0.3411226924835043,0.2107042271435060,0.035714285714285]
    if nmu == 5:
        mu = [0.165278957666387,0.477924949810444,0.738773865105505,0.919533908166459,1.00000000000000]
        wtmu = [0.327539761183898,0.292042683679684,0.224889342063117,0.133305990851069,2.222222222222220E-002]

    # define limb of planet
    thet = np.arange(361)
    xx = np.around(np.cos(thet*dtr), 14)
    zz = np.around(np.sin(thet*dtr), 14)

    # define terminator
    zt = np.linspace(-1,1,201)              # r cos theta (z coordinates of the terminator)
    angle = np.arccos(zt)+np.pi/2.          # theta
    r1 = np.around(np.cos(angle),14)        # r sin theta
    xt = r1 * np.around(np.cos((xphase)*np.pi/180.), 14) # r sin theta sin xphase (x coordinates of the terminator)

    if (xphase > 180.0):
        xt = -xt # flip after phase = 180

    rr = np.sqrt(xt**2+zt**2)   # radial coordinate of the determinator
    rmin = min(rr)              # least radius (on x axis )

    isample = 0
    for imu in range(0, nmu):       # quadrature rings
        r = np.sqrt(1.-mu[imu]**2)  # quadrature radius (from small to large)
        circumh = np.pi*r	        # half the circumference
        xx = np.around(r*np.cos(thet*dtr), 14)
        zz = np.around(r*np.sin(thet*dtr), 14)

        if r > rmin:  # quadrature ring intersects terminator.
            # find the intersection and place a sample point there
            ikeep = np.where(rr<=r)
            ikeep = ikeep[0]
            ir = np.array([ikeep[0], ikeep[-1]])    # index of two intersectionns
            xr = xt[ir]                             # coordinate of intersection
            zr = zt[ir]
            if zr[1] > 0:                           # take the intersection in the upper hemisphere
                phi = arctan(zr[1],xr[1])/dtr
            else:
                phi = arctan(zr[0],xr[0])/dtr

            # split the quadrature rings with sample points
            nphi1 = int(0.5+circumh*(phi/180.0)/delR) # round up; separation ~ R/nmu
            nphi2 = int(0.5+circumh*((180.-phi)/180.0)/delR)

            # at least 1 point either side of the intersection
            if(nphi1 < 2):
                nphi1=2
            if(nphi2 < 2):
                nphi2=2

            nphi = nphi1+nphi2-1 # intersection point double counted
            phi1 = phi*np.arange(nphi1)/(nphi1-1)
            phi2 = phi+(180.-phi)*np.arange(nphi2)/(nphi2-1)
            phi2 = phi2[1:(nphi2)]
            phix = np.concatenate([phi1,phi2])

        else:   # quadrature ring does not intersect terminator
            if(circumh > 0.0):
                nphi = int(0.5+circumh/delR)
                phix = 180*np.arange(nphi)/(nphi-1)
            else:
                nphi=1

        if(nphi > 1):

            sum = 0.
            for iphi in np.arange(0,nphi):
                xphi = phix[iphi]
                xp = r*np.cos(xphi*dtr)
                yp = r*np.sin(xphi*dtr)

                thetasol, xazi, xlat, xlon = thetasol_gen_azi_latlong(xphase,r,xphi)

                # trapezium rule weights
                if(iphi == 0):
                    wt = (phix[iphi+1]-phix[iphi])/2.0
                else:
                    if(iphi == nphi-1):
                        wt = (phix[iphi]-phix[iphi-1])/2.0
                    else:
                        wt = (phix[iphi+1]-phix[iphi-1])/2.0


                wtazi= wt/180.                                  # sample azimuthal weight
                sum = sum+wtazi
                tablat[isample] = xlat                          # sample lattitude
                tablon[isample] = xlon                          # sample longitude
                tabzen[isample] = np.arccos(mu[imu])/dtr        # sample emission zenith angle
                tabsol[isample] = thetasol/dtr                  # sample stellar zenith angle
                tabazi[isample] = xazi/dtr                      # sample stellar azimuth angle
                tabwt[isample] = 2*mu[imu]*wtmu[imu]*wtazi      # sample weight
                isample = isample+1

        else:
            xphi = 0.
            thetasol,xazi, xlat,xlon = thetasol_gen_azi_latlong(xphase,r,xphi)
            if(tabzen[isample] == 0.0):
                xazi = 180.
            tablat[isample] = xlat
            tablon[isample] = xlon
            tabzen[isample] = np.arccos(mu[imu])/dtr
            tabsol[isample] = thetasol/dtr
            tabazi[isample] = xazi
            tabwt[isample] = 2*mu[imu]*wtmu[imu]
            isample = isample+1

    nav = isample
    wav = np.zeros([6,isample])
    sum=0.
    for i in np.arange(0,isample):
        wav[0,i]=tablat[i]              # 0th array is lattitude
        wav[1,i]=tablon[i]%360          # 1st array is longitude
        wav[2,i]=tabsol[i]              # 2nd array is stellar zenith angle
        wav[3,i]=tabzen[i]              # 3rd array is emission zenith angle
        wav[4,i]=tabazi[i]              # 4th array is stellar azimuth angle
        wav[5,i]=tabwt[i]               # 5th array is weight
        sum = sum+tabwt[i]

    for i in range(isample):            # normalise weights so they add up to 1
        wav[5,i]=wav[5,i]/sum

    return nav, np.around(wav,8)


"""
nav, table = subdiscweightsv3(22.5*2, 5)
np.savetxt('test.dat',table.T,fmt='%.6e')
"""
