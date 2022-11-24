import xarray as xr
import numpy as np
import itertools


def shift(lo,la,clo,cla):
    xc = (90-cla)*np.cos(clo*np.pi/180)
    yc = (90-cla)*np.sin(clo*np.pi/180)
    xx = (90-la)*np.cos(lo*np.pi/180)
    yy = (90-la)*np.sin(lo*np.pi/180)
    x = xx + xc
    y = yy + yc
    r = np.sqrt(x**2  +  y**2)
    phi = np.arctan2(y,x) 
    nla = 90 - r
    nlo = phi *180/np.pi #+180
    return nlo, nla

def calc_centre_90(dset,year):
    """sets the SI center to the Northpole at lat=lon=90"""
    la=90.0
    lo=90.0
    return lo,la

def calc_centre_sat(dset,year):
    """calculates the SI center to the Northpole for the satellite data
    (siconc values in [0,1]).
    Takes the geographical mean point where there is sea ice (siconc value = 1)
    If that's not possile sets center to North pole at lat=lon=90
    dset: xarray dataset, with coordinates time, lat in [-180,180], lon in [-90,90] 
            and variable siconc in [0,1] in the ocean and nan on land
    year: specify the year (or time index if more than September data is available)
    """
    
    southb = np.where(dset.lat>60)[0][0] #first index where lat> 60 degrees N
    t=year 
    
    x=dset['siconc'].isel(time=t)[southb:,::]

    maxlats=np.where(x==1.0)[0]
    maxlons=np.where(x==1.0)[1]


    xxs=np.zeros_like(maxlats)
    yys=np.zeros_like(maxlons)
    if maxlats.size !=0:
        for j,i in enumerate(maxlons):
            longi=float(x.lon[i])
            lati = float(x.lat[maxlats[j]])
            ##cirular average" -> transform into cartesion
            xx = (90-lati)*np.cos(longi*np.pi/180)
            yy = (90-lati)*np.sin(longi*np.pi/180)

            xxs[j]=xx
            yys[j]=yy
        
        mx = np.mean(xxs)
        my = np.mean(yys)

        r = np.sqrt(mx**2  +  my**2)
        phi = np.arctan2(my,mx)
        nla = 90 - r
        nlo = phi *180/np.pi
        if nlo < 0:           #transform to "our" system where degrees east don't exist
                nlo = 360 + nlo         
    else:
        nlo = 90.0
        nla = 90.0

    return nlo, nla

def get_border_sat(dset,year,thlong,mlongi,mlati,conc_threshold = 0.5, exGr=False): #get border with moving center
    """ calculates the borderpoints of a (satellite) dataset
    dset: xarray dataset, with coordinates time, lat in [-180,180], lon in [-90,90] 
            and variable siconc in [0,1] in the ocean and nan on land
    year: specify the year (or time index if more tha September data is available)
    thlong: defines how many borderpoints to calculate, look at every "thlong"th direction from the center
    mlongi: longitude of the center
    mlati: latitude of the center
    conc_threshold: concentration threshold over which we consider to have sea ice in a cell
        Note:in the sat data, non-binary values come from interpolation during regridding, not actual siconc values!
    exGr: flag if SI close to Greenland should be ignored, default: False
    """
    
    t=year

    southb = np.where(dset.lat>60)[0][0] #first index where lat> 60 degrees N, only consider the Arctic

    longitudes = dset.lon.data[::thlong]
    latitudes = dset.lat.data[southb-int(np.abs(90-mlati)):]

    limpoints=np.zeros((len(longitudes),2))

    for i,lo in enumerate(longitudes):
        points = np.zeros((len(latitudes),2))
        values = np.zeros(len(latitudes))

        
        for j,la in enumerate(latitudes):
            if mlati != 90.0: #if center isn't at the north pole
                nlo,nla = shift(lo,la,mlongi,mlati)
                nnlo = dset.lon.sel(lon=nlo,method="nearest")
                nnla = dset.lat.sel(lat=nla,method="nearest")
            else:
                nnlo = lo
                nnla = la
            points[j]=[nnlo,nnla]
            
            sic = dset["siconc"].isel(time=t).sel(lat=nnla,lon=nnlo)
            values[j]=sic

        if np.where(values>=conc_threshold)[0].size !=0:
            #if there is any sea ice in the given direction
            border=np.where(values>=conc_threshold)[0][0]

            ############ADDITIONAL STUFF TO EXCLUDE GREENLAND!!!!!###########################
            #get the number of repeats of nan-values (no sea) between 70 and 80 degrees N
            if exGr: #if flag is put to exclude Greenland explicitly
                repeats=[(x, len(list(y))) for x, y in itertools.groupby(np.isnan(values[np.where(points[:,1] >=70)[0]]))] #find number of repeats of each element
                max_reps=np.zeros(10)
                ki=0
                for k in repeats:
                    if k[0]==True: #for the true elements (i.e. if they are nones in the orig. array)
                        max_reps[ki]=k[1]
                        ki+=1


                if np.max(max_reps)>=7:#8:#10: #if there is a big landmass over 70 degrees N(i.e. Greenland)
                    last_green=0
                    for gi,gv in enumerate(values[:-10]):
                        if np.isnan(gv) and np.isnan(values[gi:gi+10]).all():
                            last_green=gi
                    border=last_green+10
                    if np.where(values[border:]>15.0)[0].size !=0: #if there are positive points towards the center
                        while np.isnan(values[border]) or values[border]<15.0:
                            border+=1
                    else:
                        lcord = [mlongi,mlati]
                ############### back to normal: ##################################

            #calculate the "real" border:
            if np.prod(values[border:]>=conc_threshold) == 1:
                #if furthes point after continuous ice
                #print("CONTINUOUS ICE")
                lcord = points[border]
            else:
                o=values[border:]>=conc_threshold
                badpoints=values[border:][np.where(o==False)[0]]
                if np.prod(np.isnan(badpoints)==1):#if there's only land in the way (and no ice-free water)
                    #print("island")
                    #Ignore the island and do as before:
                    lcord = points[border]
                else:
                    #print("water in between")
                    if np.sum(o)>=0: #if there are positive points
                        m=-1
                        while np.isnan(badpoints[m]):
                            m-=1
                        probvals= np.sort(np.concatenate((np.where(values[border:]<=conc_threshold)[0],np.where(np.isnan(values[border:]))[0]),axis=None))
                        #if ice point found:
                        if border+probvals[m]+1 <=len(values)-1:
                            lcord = points[border+probvals[m]+1]
                        else: #there is no ice point... use the center
                            lcord = [mlongi,mlati]

        else: # no sea ice in that direction
            lcord=[mlongi,mlati]

        limpoints[i]=lcord
       
    return limpoints




def calc_centre(dset,year,epsi=0.2, month = 8, period = 12):
    """calculates the SI center to the Northpole for the regridded CMIP6 data
     (siconc values in [0,100]).
    Takes the geographical mean point of the cells with highest sea ice concentration 
    (where siconc >= epsi* maximal concentration)and where there is sea ice (siconc >= 15%)
    If that's not possile sets center to North pole at lat=lon=90
    dset: xarray dataset, with coordinates time, 
        lat in [-180,180], lon in [-90,90] 
        and variable siconc in [0,100] in the ocean and nan on land
    year: specify the year
    month: specify which month to look at, default: 8 (september)
    period: take every "period"th time point, default: 12 (months -> every year)
    """

    southb = np.where(dset.lat>60)[0][0] #first index where lat> 60 degrees N

    t=month+year*period
    
    x=dset['siconc'].isel(time=t)[southb:,::]
    maxlats=np.where((x>=np.max(x)-epsi*np.max(x)) & (x>=15.0))[0]
    maxlons=np.where((x>=np.max(x)-epsi*np.max(x)) & (x>=15.0))[1]


    xxs=np.zeros_like(maxlats)
    yys=np.zeros_like(maxlons)
    if maxlats.size !=0:
        for j,i in enumerate(maxlons):
            longi=float(x.lon[i])
            lati = float(x.lat[maxlats[j]])
            #"cirular average" --> transform into cartesian
            xx = (90-lati)*np.cos(longi*np.pi/180)
            yy = (90-lati)*np.sin(longi*np.pi/180)

            xxs[j]=xx
            yys[j]=yy
        
        mx = np.mean(xxs)
        my = np.mean(yys)

        r = np.sqrt(mx**2  +  my**2)
        phi = np.arctan2(my,mx)
        nla = 90 - r
        nlo = phi *180/np.pi #+180
        if nlo < 0:           #transform to "our" system wher degrees east don't exist
                nlo = 360 + nlo         

    else:
        nlo = 90.0
        nla = 90.0

    return nlo, nla






def get_border(dset,year,thlong,mlongi,mlati,exGr=False, month = 8, period=12): 
    """ calculates the borderpoints of a (regridded) CMIP6 dataset in lat/lon gridding, 
    we consider to have no sea ice in a cell, if siconc < 15%
    dset: xarray dataset, with coordinates time, lat in [-180,180], lon in [-90,90] 
            and variable siconc in [0,1] in the ocean and nan on land
    year: specify the year
    thlong: defines how many borderpoints to calculate, look at every "thlong"th direction from the center
    mlongi: longitude of the center
    mlati: latitude of the center
    exGr: flag if SI close to Greenland should be ignored, default: False
    month: specify which month to look at, default: 8 (september)
    period: take every "period"th time point, default: 12 (months -> every year)
    """
    
    t=month+period*year

    southb = np.where(dset.lat>60)[0][0] #first index where lat> 60 degrees N

    longitudes = dset.lon.data[::thlong]
    latitudes = dset.lat.data[southb-int(np.abs(90-mlati)):]

    limpoints=np.zeros((len(longitudes),2))

    for i,lo in enumerate(longitudes):
        points = np.zeros((len(latitudes),2))
        values = np.zeros(len(latitudes))
        for j,la in enumerate(latitudes):
            nlo,nla = shift(lo,la,mlongi,mlati)
            
            nnlo = dset.lon.sel(lon=nlo,method="nearest")
            nnla = dset.lat.sel(lat=nla,method="nearest")
            points[j]=[nnlo,nnla]

            sic = dset["siconc"].isel(time=t).sel(lat=nnla,lon=nnlo)

            values[j]=sic

        if np.where(values>=15.0)[0].size !=0:
            #if there is any sea ice in the given direction

            border=np.where(values>=15.0)[0][0]

            ############ADDITIONAL STUFF TO EXCLUDE GREENLAND!!!!!###########################
            #get the number of repeats of nan-values (no sea) between 70 and 80 degrees N
            if exGr: #if flag is put to exclude Greenland explicitly
                repeats=[(x, len(list(y))) for x, y in itertools.groupby(np.isnan(values[np.where(points[:,1] >=70)[0]]))] #find number of repeats of each element
                max_reps=np.zeros(10)
                ki=0
                for k in repeats:
                    if k[0]==True: #for the true elements (i.e. if they are nones in the orig. array)
                        max_reps[ki]=k[1]
                        ki+=1


                if np.max(max_reps)>=7:#8:#10: #if there is a big landmass over 70 degrees N(i.e. Greenland)
                    #print("GREENLAND")
                    last_green=0
                    for gi,gv in enumerate(values[:-10]):
                        if np.isnan(gv) and np.isnan(values[gi:gi+10]).all():
                            last_green=gi
                    border=last_green+10
                    if np.where(values[border:]>15.0)[0].size !=0: #if there are positive points towards the center
                        while np.isnan(values[border]) or values[border]<15.0:
                            border+=1
                    else:
                        lcord = [mlongi,mlati]
            ################# BACK TO NORMAL ############################

            #calculate the "real" border:
            if np.prod(values[border:]>=15.0) == 1: 
                #if furthes point after continuous ice
                #print("CONTINUOUS ICE")
                lcord = points[border]
            else:
                o=values[border:]>=15.0
                badpoints=values[border:][np.where(o==False)[0]]
                if np.prod(np.isnan(badpoints)==1):#if there's only land in the way (and no ice-free water)
                    #print("island")
                    #Ignore the island and do as before:
                    #take last point where sea ice is
                    lcord = points[border] 
                else:
                    #print("water in between")
                    if np.sum(o)>=0: #if there are positive points
                        m=-1
                        while np.isnan(badpoints[m]):
                            m-=1
                        probvals= np.sort(np.concatenate((np.where(values[border:]<=15.0)[0],np.where(np.isnan(values[border:]))[0]),axis=None))
                        #if ice point found:
                        if border+probvals[m]+1 <=len(values)-1:
                            lcord = points[border+probvals[m]+1]
                        else: #there is no ice point... use the center
                            lcord = [mlongi,mlati]


        else: # no sea ice in that direction
            lcord=[mlongi,mlati]

        limpoints[i]=lcord
    #try using the points on the grid...
    #lll = [dset.siconc.isel(time = t).sel(lon = limpoints.T[0], lat = limpoints.T[1], method="nearest").lon.data, 
    #            dset.siconc.isel(time = t).sel(lon = limpoints.T[0], lat = limpoints.T[1], method="nearest").lat.data]      
    return limpoints
    #return lll
