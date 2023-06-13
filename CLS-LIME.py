###V2.0
###This script must be put in the file directory with the pertinent
###data you want to plot. Use Excel files from CLS and .txt general report for ECHEM
#When setting up:
# 1- change all the pertinent values in __inti__ function
# 2- Make sure time calculation is right. If spread on different days, you need a -1440s element.

#imports os package and pandas package
import os
import pandas
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
import numpy
import seaborn
from celluloid import Camera #Makes it easier to animate plots
import matplotlib.animation as pltanimation #for animation
import scipy #for interpolation

class Main:
    def __init__(self):
        self.dir = os.getcwd()
        #Plot_XRF params
        self.timestart = 0  # Time you want to start plotting
        self.timeend = 460  # 0-100 is the first "charge", 360-460 is the last "discharge"
        self.plotstart = 0  # depth (in microns) to start plotting.
        self.plotend = 300  # depth (in microns) to end plotting.
        self.startTotalMinutesEchem = 1 #time (mins) difference between echem and xrf file. positive means echem started first.
        self.headerXRF = 48 #number of lines to cut from beginning of XRF file. This can annoyingly change. 48 for test file.
        self.steps = 101 # Number of points in 1 full scan. 101 for test data
        self.newSteps = self.steps
        self.stepDistance = 3  # distance (microns) between those points
        self.reduce = 3  # Cuts the data by a factor of this value (for concentration gradient only)
        self.numberOfAverages = 1 #Number of lines you want to take to use as the 1M average
        self.calibrationConcentration = 1.0 #OCV concentration in molar
        #Heatmap params
        self.timestep = 180 #in seconds (for heatmap) you can find this in XRF file. 180 for test file
        #Plotting params
        self.heatmapYTickReduction = 10 #By how much to reduce y axis tick count (1 is by nothing)
        self.ticksize = 16 #was 14 then 18 for new paper
        self.labelsize = 18  # was 14



        self.verts = [] #Empty list for 3d waterfall plot
        self.z3d = [] #Empty list for z values (time) for 3d waterfall plot

    def get_echem_files(self):  #Get all sheets of all files into dataframes and concatenates them into a single dataframe
        fileDF = pandas.DataFrame()
        allDF = pandas.DataFrame()

        os.chdir(f'{self.dir}\Echem')

        listdir = os.listdir()  # Get current directory

        fileList = list()  # Create empty list for all echem files

        for file in listdir:  # Append all files to fileList
            if file.endswith('.txt'):
                fileList.append(file)

        for i in range(len(fileList)):  # For ECLab text files
            dfEchemFile = pandas.read_csv(fileList[i], delimiter="\t", engine='python', encoding='latin1')  # Read it directly into pandas Dataframe
            # ECLabData = numpy.loadtxt(fileList[i], skiprows=3)  # Data as numpy array
            # dfEchemFile = pandas.DataFrame(ECLabData, columns=columns)
            if i == 0:  # For the first fileDF produced, call it self.dfEchem
                self.dfEchem = dfEchemFile
            else:  # For all following fileDFs produced, concatenate them to dfEchem and add last time value of previous file for continuous time
                # First substract first value of time from all others (in case file does not start at 0s)
                dfEchemFile['time/s'] = dfEchemFile['time/s'] - dfEchemFile.iloc[0, dfEchemFile.columns.get_loc('time/s')]
                # Get last time value from dfEchem. -1 is last row index and .columns.get_loc gets appropriate column index
                # iloc is faster than loc so better to do it like this
                dfEchemFile['time/s'] = dfEchemFile['time/s'] + self.dfEchem.iloc[-1, self.dfEchem.columns.get_loc('time/s')]
                self.dfEchem = pandas.concat([self.dfEchem, dfEchemFile], ignore_index=True)

        self.dfEchem['Ewe/V'] = self.dfEchem['Ewe/V'] * 1000  # Convert V to mV if necessary
        self.dfEchem['time/s'] = self.dfEchem['time/s'] / 60  # Convert s to mins if necessary
        self.dfEchem['time/s'] = self.dfEchem['time/s'] - self.dfEchem['time/s'][0] #Normalize to first timepoint
        #self.dfEchem = self.dfEchem.rename(columns={"Ewe/V": "Vol(mV)", "control/mA": "Cur(mA)", "Capacity/mA.h": "Cap(mAh)","cycle number": "Cycle ID"}, errors="raise") #Rename certain columns
        self.dfEchem = self.dfEchem.rename(columns={"time/s": "Elapsed Time", "Ewe/V": "Vol(mV)", "<I>/mA": "Cur(mA)", "Capacity/mA.h": "Cap(mAh)", "cycle number": "Cycle ID"}, errors="raise")  # Rename certain columns

        print('Upload Echem Complete')


    def plot_echem(self):
        Poty = self.dfEchem['Vol(mV)'] #Potential
        Poty = Poty/1000 #convert to V
        Potx = self.dfEchem['Elapsed Time'] #Elapsed Echem Time
        Cury = self.dfEchem['Cur(mA)']  # Current
        Curx = self.dfEchem['Elapsed Time']  # Elapsed Echem Time


        #Create Heatmap Object
        self.figHeatmap = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(4, 3, height_ratios=[0.05, 1, 0.4, 0.4])
        self.axColorbar = plt.subplot(gs[1]) #Here I assign each axis a zone of the greated gs object
        self.axHeatmap = plt.subplot(gs[3:6])
        self.axEchem = plt.subplot(gs[6:9])
        axCur = plt.subplot(gs[9:12])

        #self.figHeatmap, (self.axColorbar, self.axHeatmap, self.axEchem, axCur) = plt.subplots(4, 1, gridspec_kw={'height_ratios':[0.05, 1, 0.4, 0.4], 'width_ratios':[0.25, 1, 1, 1]}) #Do 4x1 so colorbar has own spot
        #axCur = self.axEchem.twinx() #A second axis for the echem plot
        self.axEchem.plot(Potx, Poty, 'k-', linewidth=5)
        axCur.plot(Curx, Cury, 'r-', linewidth=5)
        axCur.set_xlabel('Time (Minutes)', fontsize=self.labelsize)
        axCur.set_ylabel('I (mA)', fontsize=self.labelsize, color='k')
        self.axEchem.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) #Remove Potential x ticks
        axCur.tick_params(axis='both', labelsize=self.ticksize, colors='k') #Set axCur label size and colours
        self.axEchem.set_ylabel('E (mV vs. Li)', fontsize=self.labelsize)
        self.axEchem.tick_params(labelsize=self.ticksize)
        self.axEchem.set_xlim(self.timestart, self.timeend)
        self.axEchem.set_ylim(-0.15, 0.15)
        self.axEchem.yaxis.set_major_locator(plt.MaxNLocator(3)) #Choose # of ticks on y
        axCur.set_ylim(-0.5, 0.5)
        axCur.set_xlim(self.timestart, self.timeend)
        axCur.yaxis.set_major_locator(plt.MaxNLocator(5)) #Choose # of ticks on y
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1) #Adjust spacing of suplots
        # Hide the right and top spines
        self.axEchem.spines['right'].set_visible(False)
        self.axEchem.spines['top'].set_visible(False)
        axCur.spines['right'].set_visible(False)
        axCur.spines['top'].set_visible(False)

    def echem_animation(self): #Makes an animation of Echem vs time to match with XRD/XRF animations
        #Total frames is the amount of frames required to emulate XRF/XRD scan/position animation
        TotalFrames = len(self.verts) #Total frames must be = to # of frames from XRF animation

        #interpolate the data
        Poty, Potx = self.allDF.iloc[:, 1], self.allDF['Elapsed Time']   # Potential and Elapsed Echem Time
        Cury, Curx = self.allDF.iloc[:, 2], self.allDF['Elapsed Time']  # Current and Elapsed Echem Time
        EchemInterp = scipy.interpolate.interp1d(Potx, Poty)
        CurInterp = scipy.interpolate.interp1d(Curx, Cury)
        pointsNum = TotalFrames #Used to be used for a multiplier. No need here.
        xInterval = numpy.linspace(self.timestart, self.timeend, pointsNum)
        PotyInterp, CuryInterp = EchemInterp(xInterval), CurInterp(xInterval)

        #plt.plot(xInterval, PotyInterp, 'k.')
        #plt.show()

        # Second make the plots and animate using celluloid
        figGifechem, (axGifechem, axGifCur) = plt.subplots(2, 1)

        #Modify plot
        axGifechem.set_xlabel('Time (Minutes)', fontsize=16)
        #axGifechem.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) #Remove Potential x ticks
        axGifechem.set_ylabel('E (V vs. Li)', fontsize=16)
        axGifCur.set_xlabel('Time (Minutes)', fontsize=16)
        axGifCur.set_ylabel('I (mA)', fontsize=16)
        axGifCur.set_xlim(self.timestart, self.timeend)
        axGifCur.set_ylim(-8.5, 8.5)
        axGifCur.tick_params(labelsize=self.ticksize)
        axGifechem.tick_params(labelsize=self.ticksize)
        axGifechem.set_xlim(self.timestart, self.timeend)
        axGifechem.set_ylim(2.3, 4.3)
        #axGifechem.yaxis.set_major_locator(plt.MaxNLocator(3)) #Choose # of ticks on y
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1) #Adjust spacing of suplots
        # Hide the right and top spines
        #axGifechem.spines['right'].set_visible(False)
        #axGifechem.spines['top'].set_visible(False)
        figGifechem.set_size_inches(7, 6)
        figGifechem.tight_layout()

        #setup camera
        camera = Camera(figGifechem)  # Apply celluloid's camera function to the fig

        #plot
        for i in range(len(xInterval)):  # For all element in dataset
            # Plot
            print(i)
            axGifechem.plot(xInterval[0:i], PotyInterp[0:i], 'k-', linewidth=5)
            axGifCur.plot(xInterval[0:i], CuryInterp[0:i], 'r-', linewidth=5)
            # Snapshot
            camera.snap()  # take a snapshot of the figure in the current state

        #Animate and save
        animation = camera.animate()  # Animate these snapshots
        ffmpegwriter = pltanimation.writers['ffmpeg']
        writer = ffmpegwriter(fps=15)
        animation.save('celluloid_gif_echem.gif', writer=writer) #'Pillow'


    def get_xrf_times(self): #Put all the XRF folders in a subfolder called 'XRF'
        os.chdir(f'{self.dir}\XRF')

        self.fileList = os.listdir()

        self.elapsedTimeDictionary = dict()

        for i in range(len(self.fileList)):
            file = self.fileList[i]  # Takes file "i" from fileList

            with open(file) as f:
                #header = f.readline()  # Header is first line
                #header_fixed = header[1:]  # remove '#' from header
                #fileDF = pandas.read_csv(f, sep='\s+', names=header)

                fileDF = pandas.read_fwf(f) #Gonna have to fix this more elegantly.
                #Cannot use first line as header cause we get "duplicate header" error

            if i == 0:
                startTime = fileDF.iloc[0, 0][19:]  # Gets the timestamp of experiment beginning
                startDay = int(fileDF.iloc[0, 0][16:19])  # Gets the day the experiment began
                startHour = int(startTime[:-6])  # Gets the hour from the start time
                startMinute = int(startTime[3:-3])  # Gets the minutes from the start time
                startSeconds = int(startTime[6:])  # Gets the seconds from the start time
                startTotalMinutes = (startHour * 60) + startMinute + (startSeconds / 60)
                #startTotalMinutes = (startHour * 60) + startMinute + (startSeconds / 60) + 120  # +120 is for time difference and 1440 is for 1 day difference
                #self.elapsedTimeDictionary["file" + str(i) + "elapsedTime"] = (startTotalMinutes) # Start time does not takes into consideration the Echem data started first
                self.elapsedTimeDictionary["file" + str(i) + "elapsedTime"] = 0 #Set initial XRF time to 0
            else:
                time = fileDF.iloc[0, 0][19:]  # Gets the timestamp of experiment
                day = int(fileDF.iloc[0, 0][16:19])  # Gets the day the experiment
                hour = int(time[:-6])  # Gets the hour from the time
                minute = int(time[3:-3])  # Gets the minutes from the time
                seconds = int(time[6:])  # Gets the seconds from the time
                totalMinutes = (hour * 60) + minute + (seconds / 60) + ((day-startDay)*1440) #120 is for time difference between echem and xrf data
                #totalMinutes = (hour * 60) + minute + (seconds / 60) + 120  # +120 is for time difference
                elapsedTime = totalMinutes - startTotalMinutes  # Calculates elapsed time. Does not Takes into consideration Echem time started first
                #elapsedTime = (startTotalMinutes) + totalMinutes - startTotalMinutes
                self.elapsedTimeDictionary["file" + str(i) + "elapsedTime"] = elapsedTime

        print('ECHEM TIMES:')
        print(self.dfEchem['Elapsed Time'])
        print('XRF FILE START TIMES:')
        print(self.elapsedTimeDictionary)

    def concatenate_xrf_files(self):
        # Import all excel files in folder and create a dictionary of them
        fileDictionary = dict()

        for i in range(len(self.fileList)):
            file = self.fileList[i]  # Takes file "i" from fileList
            fileDF = pandas.read_table(file, sep='\s+', header=self.headerXRF)  # Imports the file into a DF. !!!ALL COLUMNS ARE SHIFTED TO THE LEFT BECAUSE OF THE #!!! 'Z' is actually 'time'
            fileDF['Elapsed Time'] = (fileDF['Z'] / 60) + self.elapsedTimeDictionary["file" + str(i) + "elapsedTime"] + self.startTotalMinutesEchem # Adds a column with the elapsed time (relative to start time of first file) for each measurment individually
            #Save fileDF to a dictionary
            fileDictionary["file" + str(i) + "df"] = fileDF
            print("Processed file " + str(i) + " XRF Data")

        # Concatenate all dataframes
        self.allXRFDF = pandas.DataFrame()

        for i in range(len(self.fileList)):
            self.allXRFDF = pandas.concat([self.allXRFDF, fileDictionary["file" + str(i) + "df"]], ignore_index=True)

        #Sort firstly by elapsed time, and then in second priority by the depth coordinates
        self.allXRFDF.sort_values(by=['Elapsed Time', '#'], axis=0, ascending=[True, True], inplace=True)
        self.allXRFDF.reset_index(inplace=True) #When sorting, the inserted values will have fucky indexes so we reset it here

        #print(self.allXRFDF)

    def plot_xrf(self):
        j = 0 #What is the starting value to take for 1 scan
        count = 0 #Used to limit the amount of curves plotted

        self.timeSliceDF = self.allXRFDF[(self.allXRFDF['Elapsed Time'] >= self.timestart) & (self.allXRFDF['Elapsed Time'] <= self.timeend)] #Takes only points between certain times

        self.ally = self.timeSliceDF['AsKa1'].tolist() #Add all y values from df to a list in order to plot

        self.roundedDown = len(self.timeSliceDF) - len(self.timeSliceDF) % self.newSteps  # This is the length of the data rounded down to nearest multiple of steps. % is the modulus of steps
        self.roundedDownFromBeg = len(self.allXRFDF['AsKa1']) - len(self.timeSliceDF) % self.newSteps #Same but for all data (used to do calib from beginning of data collection instead)

        # Choose colours for plotting
        self.colour_subsection = numpy.linspace(0.1, 0.9, self.roundedDown // self.newSteps)  # Splits colourmap an equal # of sections related to # of curves
        self.coloursList = [cm.autumn_r(x) for x in self.colour_subsection]

        #Setup Figure
        self.figGradient, (self.axXRF, self.axEchem2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[1, 0.25]})


        self.colourListShort = [] #Empty list to hold only the used colours

        '''Do the 1 point calibration calculations (from start, or from first time)'''
        averagey = self.one_point_calibration()
        #averagey = self.one_point_calibration_from_start()

        #Plot Gradient XRF Data (only plot a fraction of them based on value of self.reduce)
        for i in range(self.newSteps, self.roundedDown, self.newSteps): #i is the end boundary for the y values plotted
            x = list(range(self.newSteps)) # Takes x axis with as many numbers as there are steps
            x = [float(i)*self.stepDistance for i in x] #Converts x to float values and multiplies by step size (in microns)
            y = self.ally[j:i]
            y = numpy.asarray(y) #convert y values to array
            y = y * (self.calibrationConcentration/averagey) #do the 1 point calibration

            if count % self.reduce == 0: #Checks if modulus of count by reduce is 0. If no, continue loop
                colour = self.coloursList[int(i/self.newSteps)] #colour list index needs to be divided to work
                self.colourListShort.append(colour) #This is only the colours that are actually used. I keep them for the 3d plot
                self.axXRF.plot(x,y, linewidth=3, color=colour)
                self.get_3d_values(x,y,i) #Gets values for waterfall plot
                self.plot_linescans_on_echem(i) #This is a function to plot the linescan colors on the echem to indicate location in time
            else:
                pass
            # Change j and increase count
            j = i #change the next startpoint to the old endpoint
            count = count + 1

        # Plot Echem2
        self.axEchem2Cur = self.axEchem2.twinx()
        Poty = self.dfEchem['Vol(mV)'] #Potential
        Poty = Poty/1000  #convert to V
        Potx = self.dfEchem['Elapsed Time']  # Elapsed Echem Time
        Cury = self.dfEchem['Cur(mA)'] #Potential
        self.axEchem2.plot(Potx, Poty, 'k-', linewidth=3)
        self.axEchem2Cur.plot(Potx, Cury, 'r-', linewidth=3)


        #Customize figure Gradient
        self.axXRF.set_xlabel('Depth (\u03bcm)', fontsize=22)
        #self.axXRF.set_ylabel('As K\u03B1', fontsize=16)
        self.axXRF.set_ylabel('Li$^+$ Concentration (M)', fontsize=22)
        self.axXRF.set_xlim(self.plotstart, self.plotend)
        self.axXRF.set_ylim(0, 2)
        self.axXRF.tick_params(axis='both', which='major', labelsize=self.ticksize)
        self.axEchem2.tick_params(axis='both', which='major', labelsize=self.ticksize)
        self.axEchem2.set_ylabel('E (mV vs. Li)', fontsize=22)
        self.axEchem2.set_xlabel('Time (Minutes)', fontsize=22)
        self.axEchem2.tick_params(labelsize=self.ticksize)
        self.axEchem2.set_xlim(self.timestart, self.timeend)
        self.axEchem2.set_ylim(-0.1, 0.1)
        self.axEchem2Cur.set_ylabel('I (mA)', fontsize=22)
        self.axEchem2Cur.tick_params(labelsize=self.ticksize)
        #self.axEchem2.set_xlim(215, 370)
        # Hide the right and top spines
        self.axEchem2.spines['right'].set_visible(False)
        self.axEchem2.spines['top'].set_visible(False)
        # Adjust spacing of subplots
        self.figGradient.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)

        self.figGradient.set_size_inches(8, 9)

    def one_point_calibration(self): #Converts counts to concentration for gradient plot

        allxArray = numpy.empty((0, self.newSteps))  # Empty arrays with the proper size to accept x and y
        allyArray = numpy.empty((0, self.newSteps))  # These are for the one_point_calibration function

        j = 0  # What is the starting value to take for 1 scan
        count = 0  # Used to limit the amount of curves plotted

        for i in range(self.newSteps, self.roundedDown, self.newSteps): #i is the end boundary for the y values plotted
            xcalib = list(range(self.newSteps)) # Takes x axis with as many numbers as there are steps
            xcalib = [float(i)*self.stepDistance for i in xcalib] #Converts x to float values and multiplies by step size (in microns)
            ycalib = self.ally[j:i]

            #Get all values in a single array for x and y
            xArray = numpy.asarray([xcalib])  # These arrays are for the on_point_calibration function
            yArray = numpy.asarray([ycalib])
            allxArray = numpy.append(allxArray, xArray, axis=0)
            allyArray = numpy.append(allyArray, yArray, axis=0)

            #Change j and increase count
            j = i  # change the next startpoint to the old endpoint
            count = count + 1

        #Average the values for the OCV lines and return the values in an array
        averagey = allyArray[:(self.numberOfAverages)]  # The average of n first lines (all should be at 1M).
        averagey = numpy.average(averagey, axis=0)
        return averagey #which is a list of average As Ka values for the first (numberOfAverages) curves

    def one_point_calibration_from_start(self): #Same as above, but using first lines from beginning of measurement time

        allxArray = numpy.empty((0, self.newSteps))  # Empty arrays with the proper size to accept x and y
        allyArray = numpy.empty((0, self.newSteps))  # These are for the one_point_calibration function

        j = 0  # What is the starting value to take for 1 scan
        count = 0  # Used to limit the amount of curves plotted

        for i in range(self.newSteps, self.roundedDownFromBeg, self.newSteps):  # i is the end boundary for the y values plotted
            xcalib = list(range(self.newSteps))  # Takes x axis with as many numbers as there are steps
            xcalib = [float(i) * self.stepDistance for i in xcalib]  # Converts x to float values and multiplies by step size (in microns)
            ycalib = self.allXRFDF['AsKa1'].tolist() # This line makes the average with the first n linescans from time 0
            ycalib = ycalib[j:i] 

            # Get all values in a single array for x and y
            xArray = numpy.asarray([xcalib])  # These arrays are for the on_point_calibration function
            yArray = numpy.asarray([ycalib])
            allxArray = numpy.append(allxArray, xArray, axis=0)
            allyArray = numpy.append(allyArray, yArray, axis=0)

            # Change j and increase count
            j = i  # change the next startpoint to the old endpoint
            count = count + 1

        # Average the values for the OCV lines and return the values in an array
        averagey = allyArray[0:(self.numberOfAverages)]  # The average of n first lines (all should be at 1M).
        averagey = numpy.average(averagey, axis=0)
        return averagey  # which is a list of average As Ka values for the first (numberOfAverages) curves

    def plot_linescans_on_echem(self, i):
        yline = numpy.linspace(2.3, 4.7, 11) #Take 10 points linearly from 2.3V to 4.3V on y axis of heatmap
        correctIndex = self.timeSliceDF.index[0] #This finds the first index value to correct i with. Because timeSliceDF is a chopped DF, the index doesn't start at 0, it starts at this value
        xline = 0*yline + self.timeSliceDF.loc[i+correctIndex, 'Elapsed Time']  # look for the elapsed time at index i, where i represents the end boundary index for the scan. Make as function of yline so we get an equal # of values out
        self.axEchem2.plot(xline, yline, color=self.coloursList[int(i/self.newSteps)], linewidth=3, linestyle='--')

    def get_3d_values(self, x, y, i): #Get values for 3d plot in an 2d array for x (Depth) and y (As Ka), a 1d array for z (time)
        # Get time value (z)
        correctIndex = self.timeSliceDF.index[0]  # This finds the first index value to correct i with. Because timeSliceDF is a chopped DF, the index doesn't start at 0, it starts at this value
        z3d_i = self.timeSliceDF.loc[i + correctIndex, 'Elapsed Time']  # look for the elapsed time at index i, where i represents the end boundary index for the scan. Make as function of yline so we get an equal # of values out
        self.z3d.append(z3d_i)

        self.z3d = self.z3d
        y3d = y #Use x and y from the loop in get_xrf()
        x3d = x

        #y3d = x[50:]  # Use this if you want to reduce how deep the plots go (i.e to cut out separator)
        #x3d = y[50:]

        y3d[0], y3d[-1] = 0.3, 0.3 #Change first and last values of x to get a nice looking polygon
        self.verts.append(list(zip(x3d, y3d))) #Zip joins the 2 into tuples and then the whole thing in verts list

    def plot_3d_gradient(self): #x is AsKa, y is depth and i is the itiration in get_xrf
        self.fig3d = plt.figure()
        ax3d = self.fig3d.gca(projection='3d')

        #poly = PolyCollection(self.verts, edgecolor=self.colourListShort, facecolors='w', linewidths=2) #If you want white face
        poly = PolyCollection(self.verts, edgecolor=self.colourListShort, facecolors=self.colourListShort) #Converts vertices into polygons
        #poly.set_alpha(0) #If you want invisible face
        ax3d.add_collection3d(poly, zs=self.z3d, zdir='y')

        #Modify 3d plot
        ax3d.set_xlabel('Depth (\u03bcm)', fontsize=16)
        stopDepth = (self.depthend-self.depthstart)*1000 #If starting at 0um, what is the final depth value
        ax3d.set_xlim3d(0, stopDepth)
        #ax3d.set_xlabel('As K\u03B1 (counts)', fontsize=16)
        ax3d.set_zlabel('Conc. As (M)', fontsize=16)
        ax3d.set_zlim3d(0.3, 2.5)
        ax3d.set_ylabel('Time (minutes)', fontsize=16)
        ax3d.set_ylim3d(self.timestart, self.timeend)
        # Hide the right and top spines
        ax3d.spines['right'].set_visible(False)
        ax3d.spines['top'].set_visible(False)

        self.fig3d.set_size_inches(7, 6)

    def gif_3d_gradient(self): #make an animation of the 3d plot
        # Second make the plots and animate using celluloid
        figGif = plt.figure()
        axGif = figGif.gca(projection='3d')

        # Modify 3d plot
        axGif.set_xlabel('Depth (\u03bcm)', fontsize=16)
        stopDepth = (self.depthend - self.depthstart) * 1000  # If starting at 0um, what is the final depth value
        axGif.set_xlim3d(0, stopDepth)
        # ax3d.set_xlabel('As K\u03B1 (counts)', fontsize=16)
        axGif.set_zlabel('Conc. As (M)', fontsize=16)
        axGif.set_zlim3d(0.3, 2.5)
        axGif.set_ylabel('Time (minutes)', fontsize=16)
        axGif.set_ylim3d(self.timestart, self.timeend)
        # Hide the right and top spines
        axGif.spines['right'].set_visible(False)
        axGif.spines['top'].set_visible(False)
        axGif.view_init(elev=16, azim=73) #Standard elevation is 30 degrees

        figGif.set_size_inches(7, 6)

        camera = Camera(figGif)  # Apply celluloid's camera function to the fig

        #convert verts to a list
        vertslist = [self.verts]
        for i in range(len(vertslist[0])):  # For all element in dataset
            try:
                # Plot
                print(i)
                #print(vertslist[0][0:i])
                #print(z3d[0:i])
                poly = PolyCollection(vertslist[0][0:i], edgecolor='k', facecolors=self.colourListShort[0:i])  # Converts vertices into polygons
                axGif.add_collection3d(poly, zs=self.z3d[0:i], zdir='y')
                # Snapshot
                camera.snap()  # take a snapshot of the figure in the current state
            except ValueError:
                next

        animation = camera.animate()  # Animate these snapshots
        ffmpegwriter = pltanimation.writers['ffmpeg']
        writer = ffmpegwriter(fps=15)
        animation.save('celluloid_gif.gif', writer=writer) #'Pillow'

    def heatmap(self):
        #Get time values
        #xAxis = list()  # Time values
        #for i in range(self.timestart*60, self.timeend*60, self.timestep):
            #xAxis.append(i/60) #Get time in minutes

        #Get depth values
        yAxis = list(range(self.newSteps))  # Takes y axis with as many numbers as there are steps
        yAxis = [float(i) * self.stepDistance for i in yAxis]  # Converts y to float values and multiplies by step size (in microns)
        #yAxis = [float(i) / 1000 for i in yAxis] #Divide all values by 1000 to scale down yAxis according to label
        yAxis = [float(i) for i in yAxis] #Use this to keep in microns instead of mm

        allStepsList = list()

        #Get heatmap data
        print(len(self.timeSliceDF['Elapsed Time']))
        for i in range(0, len(self.timeSliceDF['Elapsed Time']), self.newSteps):
            timeForStep = self.timeSliceDF.loc[self.timeSliceDF.index[i], 'Elapsed Time']  # Finds the elapsed time value for each step
            stepList = self.timeSliceDF['AsKa1'][(self.timeSliceDF['Elapsed Time'] == timeForStep)]  # Save all AsKa values where elapsed time is == timeforStep
            stepList = stepList.rename(round(timeForStep)) #Changes name (which will be column name)
            stepList = stepList.reset_index(drop=True) #Resets index so they all have the same
            allStepsList.append(stepList) #Append to a list
            #allStepsList.append(stepList) #Append to a list
            print(f'Processing step {i/self.newSteps} for Heatmap')
            heatmapDF = pandas.concat(allStepsList, axis=1, ignore_index=False) #Concatenate them all according to index

        #####Do a 1-point calibration for the heatmap#####
        for i in range(0, len(heatmapDF)):
            averageAtOCV = numpy.average(heatmapDF.iloc[i,0:1]) #0:x is the curves to be used for normalization
            heatmapDF.iloc[i] = (heatmapDF.iloc[i]/averageAtOCV)*self.calibrationConcentration

        #Plot heatmap
        self.colour_subsection = numpy.linspace(0, 1,
                                                1000)  # Splits colourmap an equal # of sections related to concentration accuracy
        cmap = [cm.jet(x) for x in self.colour_subsection]
        #cmap = seaborn.cubehelix_palette(8) #A sequential color palette

        HM = seaborn.heatmap(heatmapDF, vmin=0.0, vmax=2.0, cmap=cmap, ax=self.axHeatmap, cbar_ax=self.axColorbar, #vmin 0.01 and vmax 0.19 OR 0.15 and 0.5 for sep OR 0.1 - 3.0 for concentration
                             cbar_kws={"orientation": "horizontal", 'label': 'As K\u03B1 (Normalized counts)'}, xticklabels=0,
                             yticklabels=self.heatmapYTickReduction, mask=heatmapDF.isnull())  # The ax ties in to the subplot command in plot_echem

        #Modify Colorbar details
        #self.axColorbar.set_xlabel('As K\u03B1 (Normalized counts)', fontsize=16)
        self.axColorbar.set_xlabel('Li$^+$ concentration (M)', fontsize=18) #was font 16 for paper fig
        self.axColorbar.xaxis.tick_top() #Set colourbar ticks to top
        self.axColorbar.xaxis.set_label_position('top') #Set colourbar label to top
        self.axColorbar.tick_params(axis='x', which='major', labelsize=self.ticksize - 4)

        #Modify Heatmap Details
        HM.set(xlabel='', xticklabels=[], yticklabels=yAxis[0::self.heatmapYTickReduction]) #The 0::x just takes every nth element of list
        HM.tick_params(labelsize=self.ticksize)
        #HM.set_ylabel(ylabel='Depth (10\u00b2 \u03bcm)', fontsize=16)
        HM.set_ylabel(ylabel='Depth (\u03bcmm)', fontsize=self.labelsize)
        #HM.set_ylabel(ylabel='Depth (mm)', fontsize=self.ticksize)
        HM.set_yticklabels(HM.get_yticklabels(), rotation=0) #Make sure labels are not rotated
        HM.set_ylim(self.plotend/self.stepDistance, self.plotstart/self.stepDistance) #depth limits. Divide by step size because the ylim is by # of datapoints
        #self.axHeatmap.get_yaxis().set_major_formatter(plt.ScalarFormatter())  # Sets y axis to a scalar formatter
        #self.axHeatmap.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) #Changes the formatter to use sci notation
        self.figHeatmap.set_size_inches(10, 7.5)


    def save_figs(self):
        self.figHeatmap.savefig('Figure Heatmap', dpi=1000)
        self.figGradient.savefig('Figure Gradient', dpi=1000)
        #self.fig3d.savefig('Figure 3d', dpi=1000)

m = Main()
m.get_echem_files()
m.plot_echem()
m.get_xrf_times()
m.concatenate_xrf_files()
m.plot_xrf()
#m.gif_3d_gradient()
m.heatmap()
#m.plot_3d_gradient()
#m.echem_animation() #must run plot_3d gradient first to get right number of frames
#m.save_figs()

plt.tight_layout()
plt.show()
